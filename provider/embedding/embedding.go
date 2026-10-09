// Package embedding creates text embeddings through the OpenAI-compatible
// embeddings API.
package embedding

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/runforyou-ai/einorun/provider/apierr"
	"github.com/runforyou-ai/einorun/provider/internal/httpx"
	"github.com/runforyou-ai/einorun/provider/vendor"
)

// BatchSize is the number of inputs per request, the smallest batch that every
// supported vendor accepts.
const BatchSize = 20

// maxResponseBytes bounds a single response.
const maxResponseBytes = 32 << 20

// ErrDimensionMismatch reports vectors whose dimension differs from the one
// requested.
var ErrDimensionMismatch = errors.New("embedding: dimension mismatch")

// Endpoint is an OpenAI-compatible endpoint, as returned by
// vendor.Preset.CompatibleURL, and its API key (empty for services without
// credentials).
type Endpoint struct {
	BaseURL string
	APIKey  string
}

// Result is the vectors in input order and the input tokens the vendor
// reported (0 when it reported none).
type Result struct {
	Vectors     [][]float32
	InputTokens int
}

// Client creates embeddings.
type Client struct {
	HTTP httpx.Doer
}

// NewClient returns a client with a five-minute timeout per request that
// never follows redirects.
func NewClient() *Client {
	return &Client{HTTP: httpx.NewClient(5 * time.Minute)}
}

// Embed creates a vector of dimension for each input, in batches of BatchSize.
// Failures of the API are classified with apierr; vectors of the wrong
// dimension fail with ErrDimensionMismatch.
func (c *Client) Embed(ctx context.Context, endpoint Endpoint, model string, dimension int, inputs []string) (Result, error) {
	url, err := vendor.AppendPath(endpoint.BaseURL, "embeddings")
	if err != nil {
		return Result{}, apierr.InvalidConfigError(err)
	}
	result := Result{Vectors: make([][]float32, 0, len(inputs))}
	for start := 0; start < len(inputs); start += BatchSize {
		batch := inputs[start:min(start+BatchSize, len(inputs))]
		var output struct {
			Data []struct {
				Index     int       `json:"index"`
				Embedding []float32 `json:"embedding"`
			} `json:"data"`
			Usage struct {
				PromptTokens int `json:"prompt_tokens"`
			} `json:"usage"`
		}
		err := httpx.Call(ctx, c.HTTP, httpx.Request{
			Method: http.MethodPost, URL: url, APIKey: endpoint.APIKey, MaxBytes: maxResponseBytes,
			JSON: map[string]any{"model": model, "input": batch, "dimensions": dimension},
		}, &output)
		if err != nil {
			return Result{}, err
		}
		if len(output.Data) != len(batch) {
			return Result{}, apierr.New(apierr.StageCapability, apierr.Protocol, fmt.Errorf("got %d vectors for %d inputs", len(output.Data), len(batch)))
		}
		// Vendors identify inputs by index; restore the input order.
		ordered := make([][]float32, len(batch))
		for _, item := range output.Data {
			if item.Index < 0 || item.Index >= len(batch) || ordered[item.Index] != nil {
				return Result{}, apierr.New(apierr.StageCapability, apierr.Protocol, fmt.Errorf("invalid vector index %d", item.Index))
			}
			if len(item.Embedding) != dimension {
				return Result{}, fmt.Errorf("%w: got %d, want %d", ErrDimensionMismatch, len(item.Embedding), dimension)
			}
			ordered[item.Index] = item.Embedding
		}
		result.Vectors = append(result.Vectors, ordered...)
		result.InputTokens += output.Usage.PromptTokens
	}
	return result, nil
}
