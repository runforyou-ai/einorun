package provider_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/runforyou-ai/einorun/llm"
	"github.com/runforyou-ai/einorun/provider/apierr"
	"github.com/runforyou-ai/einorun/provider/discovery"
	"github.com/runforyou-ai/einorun/provider/embedding"
	"github.com/runforyou-ai/einorun/provider/probe"
	"github.com/runforyou-ai/einorun/provider/rerank"
	"github.com/runforyou-ai/einorun/provider/vendor"
)

// serve answers requests by path; it records the request bodies and headers.
func serve(t *testing.T, routes map[string]func(body map[string]any) (int, any)) (*httptest.Server, *[]*http.Request) {
	t.Helper()
	var mu sync.Mutex
	var requests []*http.Request
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		requests = append(requests, r)
		mu.Unlock()
		route, ok := routes[r.URL.Path]
		if !ok {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		status, response := route(body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_ = json.NewEncoder(w).Encode(response)
	}))
	t.Cleanup(ts.Close)
	return ts, &requests
}

func TestEmbedding(t *testing.T) {
	ts, requests := serve(t, map[string]func(map[string]any) (int, any){
		"/v1/embeddings": func(body map[string]any) (int, any) {
			inputs := body["input"].([]any)
			data := make([]any, 0, len(inputs))
			for i := len(inputs) - 1; i >= 0; i-- { // reversed order
				data = append(data, map[string]any{"index": i, "embedding": []float32{float32(i), 0}})
			}
			return http.StatusOK, map[string]any{"data": data, "usage": map[string]any{"prompt_tokens": len(inputs)}}
		},
	})
	inputs := make([]string, 25)
	result, err := embedding.NewClient().Embed(context.Background(), embedding.Endpoint{BaseURL: ts.URL + "/v1", APIKey: "k"}, "m", 2, inputs)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Vectors) != 25 || result.Vectors[1][0] != 1 || result.Vectors[21][0] != 1 || result.InputTokens != 25 {
		t.Fatalf("result %+v", result)
	}
	if len(*requests) != 2 || (*requests)[0].Header.Get("Authorization") != "Bearer k" {
		t.Fatalf("requests %d", len(*requests))
	}
	if _, err := embedding.NewClient().Embed(context.Background(), embedding.Endpoint{BaseURL: ts.URL + "/v1"}, "m", 3, []string{"a"}); !errors.Is(err, embedding.ErrDimensionMismatch) {
		t.Fatalf("dimension: %v", err)
	}
}

func TestEmbeddingStatus(t *testing.T) {
	ts, _ := serve(t, map[string]func(map[string]any) (int, any){
		"/embeddings": func(map[string]any) (int, any) { return http.StatusUnauthorized, map[string]any{"error": "bad key"} },
	})
	_, err := embedding.NewClient().Embed(context.Background(), embedding.Endpoint{BaseURL: ts.URL}, "m", 2, []string{"a"})
	classified, ok := apierr.As(err)
	if !ok || classified.Kind != apierr.Unauthorized || classified.Status != http.StatusUnauthorized {
		t.Fatalf("error %v", err)
	}
}

func TestRerank(t *testing.T) {
	ts, _ := serve(t, map[string]func(map[string]any) (int, any){
		"/v1/rerank": func(map[string]any) (int, any) {
			return http.StatusOK, map[string]any{"results": []any{map[string]any{"index": 1, "relevance_score": 0.9}}, "usage": map[string]any{"total_tokens": 7}}
		},
		"/api/v1/services/rerank/text-rerank/text-rerank": func(body map[string]any) (int, any) {
			if body["input"] == nil {
				return http.StatusBadRequest, nil
			}
			return http.StatusOK, map[string]any{"output": map[string]any{"results": []any{map[string]any{"index": 0, "relevance_score": 0.5}}}}
		},
	})
	client := rerank.NewClient()
	result, err := client.Rerank(context.Background(), rerank.Endpoint{Protocol: vendor.RerankCompatible, BaseURL: ts.URL + "/v1"}, "m", "q", []string{"a", "b"}, 1)
	if err != nil || result.Scores[0] != (rerank.Score{Index: 1, Relevance: 0.9}) || result.InputTokens != 7 {
		t.Fatalf("compatible %+v %v", result, err)
	}
	result, err = client.Rerank(context.Background(), rerank.Endpoint{Protocol: vendor.RerankDashScope, BaseURL: ts.URL + "/compatible-mode/v1"}, "m", "q", []string{"a"}, 1)
	if err != nil || result.Scores[0].Index != 0 {
		t.Fatalf("dashscope %+v %v", result, err)
	}
	if _, err := client.Rerank(context.Background(), rerank.Endpoint{Protocol: vendor.RerankCompatible, BaseURL: ts.URL + "/v1"}, "m", "q", []string{"a"}, 1); err == nil {
		t.Fatal("accepted an out-of-range index")
	}
}

func TestProbe(t *testing.T) {
	ts, requests := serve(t, map[string]func(map[string]any) (int, any){
		"/v1/models":     func(map[string]any) (int, any) { return http.StatusOK, map[string]any{"data": []any{}} },
		"/api/tags":      func(map[string]any) (int, any) { return http.StatusOK, map[string]any{"models": []any{}} },
		"/v1beta/models": func(map[string]any) (int, any) { return http.StatusOK, map[string]any{"nothing": 1} },
	})
	ctx := context.Background()
	if err := probe.Run(ctx, nil, probe.Endpoint{Brand: vendor.OpenAI, BaseURL: ts.URL + "/v1", APIKey: "k"}); err != nil {
		t.Fatal(err)
	}
	if err := probe.Run(ctx, nil, probe.Endpoint{Brand: vendor.Anthropic, BaseURL: ts.URL, APIKey: "k"}); err != nil {
		t.Fatal(err)
	}
	last := (*requests)[len(*requests)-1]
	if last.Header.Get("X-Api-Key") != "k" || last.Header.Get("Authorization") != "" {
		t.Fatalf("anthropic headers %v", last.Header)
	}
	if err := probe.Run(ctx, nil, probe.Endpoint{Brand: vendor.Ollama, BaseURL: ts.URL + "/v1"}); err != nil {
		t.Fatal(err)
	}
	err := probe.Run(ctx, nil, probe.Endpoint{Brand: vendor.Google, BaseURL: ts.URL, APIKey: "k"})
	if classified, ok := apierr.As(err); !ok || classified.Kind != apierr.Protocol {
		t.Fatalf("google: %v", err)
	}
	err = probe.Run(ctx, nil, probe.Endpoint{Brand: vendor.TypeSafe, BaseURL: ts.URL + "/missing"})
	if classified, ok := apierr.As(err); !ok || classified.Kind != apierr.NotFound {
		t.Fatalf("missing: %v", err)
	}
}

func TestDiscovery(t *testing.T) {
	ts, _ := serve(t, map[string]func(map[string]any) (int, any){
		"/api/tags": func(map[string]any) (int, any) {
			return http.StatusOK, map[string]any{"models": []any{map[string]any{"model": "llava"}, map[string]any{"name": "nomic"}}}
		},
		"/api/show": func(body map[string]any) (int, any) {
			if body["model"] == "llava" {
				return http.StatusOK, map[string]any{"capabilities": []string{"completion", "vision"}, "parameters": "num_ctx 8192\nstop x"}
			}
			return http.StatusOK, map[string]any{"capabilities": []string{"embedding"}}
		},
		"/api/v1/models": func(map[string]any) (int, any) {
			return http.StatusOK, map[string]any{"data": []any{
				map[string]any{"id": "a/chat", "name": "Chat", "context_length": 1000,
					"architecture": map[string]any{"input_modalities": []string{"text", "image", "file"}, "output_modalities": []string{"text"}},
					"top_provider": map[string]any{"max_completion_tokens": 100}},
				map[string]any{"id": "a/img", "architecture": map[string]any{"input_modalities": []string{"text"}, "output_modalities": []string{"image"}}},
				map[string]any{"id": "a/emb", "architecture": map[string]any{"input_modalities": []string{"text"}, "output_modalities": []string{"embeddings"}}},
			}}
		},
	})
	ctx := context.Background()
	models, err := discovery.Discover(ctx, nil, discovery.Endpoint{Brand: vendor.Ollama, BaseURL: ts.URL})
	if err != nil || len(models) != 2 {
		t.Fatalf("ollama %+v %v", models, err)
	}
	if models[0].ContextWindow != 8192 || models[0].MaxOutputTokens != 8192 || !slices.Equal(models[0].Inputs, []llm.Modality{llm.Image}) || models[1].Type != discovery.Embedding {
		t.Fatalf("ollama details %+v", models)
	}
	models, err = discovery.Discover(ctx, nil, discovery.Endpoint{Brand: vendor.OpenRouter, BaseURL: ts.URL + "/api/v1"})
	if err != nil || len(models) != 2 || models[0].MaxOutputTokens != 100 || !slices.Equal(models[0].Inputs, []llm.Modality{llm.Image}) || models[1].Type != discovery.Embedding {
		t.Fatalf("openrouter %+v %v", models, err)
	}
	if _, err := discovery.Discover(ctx, nil, discovery.Endpoint{Brand: vendor.Anthropic, BaseURL: ts.URL}); !errors.Is(err, discovery.ErrUnsupported) {
		t.Fatalf("unsupported: %v", err)
	}
}

func TestDeadlineIsTimeout(t *testing.T) {
	release := make(chan struct{})
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
		case <-release:
		}
	}))
	defer ts.Close()
	defer close(release)
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	_, err := embedding.NewClient().Embed(ctx, embedding.Endpoint{BaseURL: ts.URL}, "m", 2, []string{"a"})
	if classified, ok := apierr.As(err); !ok || classified.Kind != apierr.Timeout {
		t.Fatalf("deadline: %v", err)
	}
	canceled, cancelNow := context.WithCancel(context.Background())
	cancelNow()
	if _, err := embedding.NewClient().Embed(canceled, embedding.Endpoint{BaseURL: ts.URL}, "m", 2, []string{"a"}); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancel: %v", err)
	}
}

func TestRedirectsAreNotFollowed(t *testing.T) {
	var followed bool
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { followed = true }))
	defer target.Close()
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, target.URL+r.URL.Path, http.StatusFound)
	}))
	defer ts.Close()
	err := probe.Run(context.Background(), nil, probe.Endpoint{Brand: vendor.Anthropic, BaseURL: ts.URL, APIKey: "k"})
	if followed || err == nil {
		t.Fatalf("redirect followed %v err %v", followed, err)
	}
	if _, err := discovery.Discover(context.Background(), nil, discovery.Endpoint{Brand: vendor.OpenAICompatible, BaseURL: ts.URL}); followed || err == nil {
		t.Fatalf("discovery followed %v err %v", followed, err)
	}
}

func TestTimeoutsHideCredentials(t *testing.T) {
	release := make(chan struct{})
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
		case <-release:
		}
	}))
	defer ts.Close()
	defer close(release)
	client := &embedding.Client{HTTP: &http.Client{Timeout: 50 * time.Millisecond}}
	_, err := client.Embed(context.Background(), embedding.Endpoint{BaseURL: ts.URL + "/v1?api_key=secret"}, "m", 2, []string{"a"})
	if classified, ok := apierr.As(err); !ok || classified.Kind != apierr.Timeout || strings.Contains(err.Error(), "secret") {
		t.Fatalf("error %v", err)
	}
}

func TestTransportErrorsHideCredentials(t *testing.T) {
	_, err := embedding.NewClient().Embed(context.Background(), embedding.Endpoint{BaseURL: "http://user:secret@127.0.0.1:1/v1?key=secret"}, "m", 2, []string{"a"})
	if err == nil || strings.Contains(err.Error(), "secret") {
		t.Fatalf("error %v", err)
	}
}
