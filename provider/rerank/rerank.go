// Package rerank scores documents by their relevance to a query.
package rerank

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/runforyou-ai/einorun/provider/apierr"
	"github.com/runforyou-ai/einorun/provider/internal/httpx"
	"github.com/runforyou-ai/einorun/provider/vendor"
)

// maxResponseBytes bounds a single response.
const maxResponseBytes = 4 << 20

// Endpoint is a rerank service: its request format, base URL and API key
// (empty for services without credentials). Use the brand's preset to choose
// Protocol.
type Endpoint struct {
	Protocol vendor.RerankProtocol
	BaseURL  string
	APIKey   string
}

// Score is the relevance of the document at Index in the request.
type Score struct {
	Index     int
	Relevance float64
}

// Result is the scores and the tokens the vendor reported (0 when it reported
// none).
type Result struct {
	Scores      []Score
	InputTokens int
}

// Client scores documents.
type Client struct {
	HTTP httpx.Doer
}

// NewClient returns a client with a one-minute timeout per request that never
// follows redirects.
func NewClient() *Client {
	return &Client{HTTP: httpx.NewClient(time.Minute)}
}

// Rerank scores documents against query and returns at most topN scores.
// Failures of the API are classified with apierr; a response without any score
// is a protocol failure.
func (c *Client) Rerank(ctx context.Context, endpoint Endpoint, model, query string, documents []string, topN int) (Result, error) {
	url, err := URL(endpoint.Protocol, endpoint.BaseURL)
	if err != nil {
		return Result{}, apierr.InvalidConfigError(err)
	}
	var payload any
	if endpoint.Protocol == vendor.RerankDashScope {
		payload = map[string]any{
			"model":      model,
			"input":      map[string]any{"query": query, "documents": documents},
			"parameters": map[string]any{"return_documents": false, "top_n": topN},
		}
	} else {
		payload = map[string]any{"model": model, "query": query, "documents": documents, "top_n": topN}
	}
	type item struct {
		Index          int     `json:"index"`
		RelevanceScore float64 `json:"relevance_score"`
	}
	var decoded struct {
		Results []item `json:"results"`
		Output  struct {
			Results []item `json:"results"`
		} `json:"output"`
		Usage struct {
			TotalTokens int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := httpx.Call(ctx, c.HTTP, httpx.Request{Method: http.MethodPost, URL: url, APIKey: endpoint.APIKey, JSON: payload, MaxBytes: maxResponseBytes}, &decoded); err != nil {
		return Result{}, err
	}
	results := decoded.Results
	if endpoint.Protocol == vendor.RerankDashScope {
		results = decoded.Output.Results
	}
	if len(results) == 0 {
		return Result{}, apierr.New(apierr.StageCapability, apierr.Protocol, errors.New("no scores in response"))
	}
	scores := make([]Score, 0, len(results))
	for _, r := range results {
		if r.Index < 0 || r.Index >= len(documents) {
			return Result{}, apierr.New(apierr.StageCapability, apierr.Protocol, fmt.Errorf("invalid document index %d", r.Index))
		}
		scores = append(scores, Score{Index: r.Index, Relevance: r.RelevanceScore})
	}
	return Result{Scores: scores, InputTokens: decoded.Usage.TotalTokens}, nil
}

// URL returns the rerank endpoint for a base URL in the given format.
func URL(protocol vendor.RerankProtocol, baseURL string) (string, error) {
	parsed, err := vendor.ParseURL(baseURL)
	if err != nil {
		return "", err
	}
	path := strings.TrimSuffix(parsed.Path, "/")
	if protocol == vendor.RerankDashScope {
		for _, suffix := range []string{"/compatible-mode/v1", "/api/v1", "/v1"} {
			path = strings.TrimSuffix(path, suffix)
		}
		path += "/api/v1/services/rerank/text-rerank/text-rerank"
	} else {
		path += "/rerank"
	}
	parsed.Path, parsed.RawPath = path, ""
	return parsed.String(), nil
}
