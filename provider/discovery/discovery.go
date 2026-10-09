// Package discovery reads the models a vendor endpoint offers.
package discovery

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/runforyou-ai/einorun/llm"
	"github.com/runforyou-ai/einorun/provider/apierr"
	"github.com/runforyou-ai/einorun/provider/internal/httpx"
	"github.com/runforyou-ai/einorun/provider/probe"
	"github.com/runforyou-ai/einorun/provider/vendor"
)

// ModelType is what a model is used for.
type ModelType string

// Model types.
const (
	Chat      ModelType = "chat"
	Embedding ModelType = "embedding"
	Rerank    ModelType = "rerank"
	Decision  ModelType = "decision"
)

// Model is one model an endpoint offers. Fields the endpoint does not report
// are zero; callers fill them in.
type Model struct {
	ID   string
	Name string
	Type ModelType
	// Inputs are the input modalities besides text.
	Inputs          []llm.Modality
	ContextWindow   int64
	MaxOutputTokens int64
}

// Endpoint is a vendor endpoint.
type Endpoint struct {
	Brand   vendor.Brand
	BaseURL string
	APIKey  string
}

// ErrUnsupported reports a brand whose models cannot be discovered.
var ErrUnsupported = errors.New("discovery: brand does not support model discovery")

// ollamaDetailConcurrency bounds concurrent Ollama detail requests.
const ollamaDetailConcurrency = 4

// DefaultTimeout bounds each request made with the default client.
const DefaultTimeout = time.Minute

// Discover reads the models of endpoint. Brands whose preset does not support
// discovery fail with ErrUnsupported. A nil client means one that never
// follows redirects and times out after DefaultTimeout.
func Discover(ctx context.Context, client httpx.Doer, endpoint Endpoint) ([]Model, error) {
	if client == nil {
		client = httpx.NewClient(DefaultTimeout)
	}
	switch endpoint.Brand {
	case vendor.Ollama:
		return discoverOllama(ctx, client, endpoint)
	case vendor.OpenRouter:
		return discoverOpenRouter(ctx, client, endpoint)
	case vendor.OpenAICompatible:
		return discoverCompatible(ctx, client, endpoint)
	}
	return nil, fmt.Errorf("%w: %s", ErrUnsupported, endpoint.Brand)
}

// discoverOllama lists installed models and reads their capabilities and
// context window; a model whose details cannot be read keeps its basic entry.
func discoverOllama(ctx context.Context, client httpx.Doer, endpoint Endpoint) ([]Model, error) {
	url, err := probe.OllamaURL(endpoint.BaseURL, "api/tags")
	if err != nil {
		return nil, apierr.InvalidConfigError(err)
	}
	var list struct {
		Models []struct {
			Model string `json:"model"`
			Name  string `json:"name"`
		} `json:"models"`
	}
	if err := httpx.Call(ctx, client, httpx.Request{URL: url, APIKey: endpoint.APIKey}, &list); err != nil {
		return nil, err
	}
	models := make([]Model, 0, len(list.Models))
	for _, item := range list.Models {
		id := strings.TrimSpace(item.Model)
		if id == "" {
			id = strings.TrimSpace(item.Name)
		}
		if id != "" {
			models = append(models, Model{ID: id, Name: id, Type: Chat})
		}
	}
	showURL, err := probe.OllamaURL(endpoint.BaseURL, "api/show")
	if err != nil {
		return nil, apierr.InvalidConfigError(err)
	}
	var wg sync.WaitGroup
	limit := make(chan struct{}, ollamaDetailConcurrency)
	for i := range models {
		wg.Add(1)
		limit <- struct{}{}
		go func() {
			defer func() { <-limit; wg.Done() }()
			var detail struct {
				Capabilities []string `json:"capabilities"`
				Parameters   string   `json:"parameters"`
			}
			request := httpx.Request{Method: http.MethodPost, URL: showURL, APIKey: endpoint.APIKey, JSON: map[string]string{"model": models[i].ID}}
			if err := httpx.Call(ctx, client, request, &detail); err != nil {
				slog.WarnContext(ctx, "discovery: reading Ollama model details failed", "model", models[i].ID, "error", err)
				return
			}
			applyOllamaDetail(&models[i], detail.Capabilities, detail.Parameters)
		}()
	}
	wg.Wait()
	return models, nil
}

// applyOllamaDetail fills in type, modalities and limits from Ollama's
// capabilities and Modelfile parameters.
func applyOllamaDetail(model *Model, capabilities []string, parameters string) {
	for _, capability := range capabilities {
		switch capability {
		case "embedding":
			model.Type = Embedding
		case "vision":
			model.Inputs = []llm.Modality{llm.Image}
		}
	}
	// The context window is the num_ctx Ollama actually loads.
	for line := range strings.SplitSeq(parameters, "\n") {
		fields := strings.Fields(line)
		if len(fields) < 2 || fields[0] != "num_ctx" {
			continue
		}
		if n, err := strconv.ParseInt(fields[1], 10, 64); err == nil && n > 0 {
			model.ContextWindow = n
		}
		break
	}
	if model.Type == Chat {
		// Local models are limited by their context only.
		model.MaxOutputTokens = model.ContextWindow
	}
}

// discoverCompatible reads an OpenAI-compatible model list, which only has
// identifiers.
func discoverCompatible(ctx context.Context, client httpx.Doer, endpoint Endpoint) ([]Model, error) {
	url, err := vendor.AppendPath(endpoint.BaseURL, "models")
	if err != nil {
		return nil, apierr.InvalidConfigError(err)
	}
	var list struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := httpx.Call(ctx, client, httpx.Request{URL: url, APIKey: endpoint.APIKey}, &list); err != nil {
		return nil, err
	}
	if list.Data == nil {
		return nil, apierr.New(apierr.StageCapability, apierr.Protocol, errors.New("response has no data array"))
	}
	models := make([]Model, 0, len(list.Data))
	for _, item := range list.Data {
		if id := strings.TrimSpace(item.ID); id != "" {
			models = append(models, Model{ID: id, Name: id, Type: Chat})
		}
	}
	return models, nil
}

// discoverOpenRouter reads OpenRouter's catalog of every output type and keeps
// chat, embedding, rerank and decision models that accept text.
func discoverOpenRouter(ctx context.Context, client httpx.Doer, endpoint Endpoint) ([]Model, error) {
	url, err := vendor.AppendPath(endpoint.BaseURL, "models")
	if err != nil {
		return nil, apierr.InvalidConfigError(err)
	}
	var list struct {
		Data []struct {
			ID            string `json:"id"`
			Name          string `json:"name"`
			ContextLength int64  `json:"context_length"`
			Architecture  struct {
				InputModalities  []string `json:"input_modalities"`
				OutputModalities []string `json:"output_modalities"`
			} `json:"architecture"`
			TopProvider struct {
				MaxCompletionTokens int64 `json:"max_completion_tokens"`
			} `json:"top_provider"`
		} `json:"data"`
	}
	if err := httpx.Call(ctx, client, httpx.Request{URL: url + "?output_modalities=all", APIKey: endpoint.APIKey}, &list); err != nil {
		return nil, err
	}
	if list.Data == nil {
		return nil, apierr.New(apierr.StageCapability, apierr.Protocol, errors.New("response has no data array"))
	}
	models := make([]Model, 0, len(list.Data))
	for _, item := range list.Data {
		id := strings.TrimSpace(item.ID)
		if id == "" || len(item.Architecture.OutputModalities) != 1 {
			continue
		}
		model := Model{ID: id, Name: strings.TrimSpace(item.Name), ContextWindow: item.ContextLength}
		if model.Name == "" {
			model.Name = id
		}
		switch item.Architecture.OutputModalities[0] {
		case "text":
			model.Type, model.MaxOutputTokens = Chat, item.TopProvider.MaxCompletionTokens
		case "embeddings":
			model.Type = Embedding
		case "rerank":
			model.Type = Rerank
		case "decisions":
			model.Type = Decision
		default:
			continue
		}
		text := false
		for _, modality := range item.Architecture.InputModalities {
			switch m := llm.Modality(modality); m {
			case "text":
				text = true
			case llm.Image, llm.Audio, llm.Video:
				if !slices.Contains(model.Inputs, m) {
					model.Inputs = append(model.Inputs, m)
				}
			}
		}
		if text {
			models = append(models, model)
		}
	}
	return models, nil
}
