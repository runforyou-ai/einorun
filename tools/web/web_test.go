package web

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/runforyou-ai/einorun/llm"
)

func TestSearch(t *testing.T) {
	var got SearchRequest
	searcher := SearcherFunc(func(_ context.Context, request SearchRequest) (SearchResult, error) {
		got = request
		return SearchResult{Items: []SearchItem{{Title: "Go", URL: "https://go.dev"}}}, nil
	})
	result, err := NewSearchTool(searcher, Options{}).InvokableRun(context.Background(), `{"query":" golang ","count":50,"recency":"week"}`)
	if err != nil {
		t.Fatal(err)
	}
	if got != (SearchRequest{Query: "golang", Count: MaxCount, Recency: RecencyWeek}) {
		t.Fatalf("request %+v", got)
	}
	var decoded SearchResult
	if err := json.Unmarshal([]byte(result), &decoded); err != nil || len(decoded.Items) != 1 || decoded.Items[0].URL != "https://go.dev" {
		t.Fatalf("result %s %v", result, err)
	}
}

func TestSearchDefaults(t *testing.T) {
	var got SearchRequest
	searcher := SearcherFunc(func(_ context.Context, request SearchRequest) (SearchResult, error) {
		got = request
		return SearchResult{}, nil
	})
	result, err := NewSearchTool(searcher, Options{}).InvokableRun(context.Background(), `{"query":"x"}`)
	if err != nil || got.Count != DefaultCount || result != `{"items":[]}` {
		t.Fatalf("result %s request %+v err %v", result, got, err)
	}
}

func TestSearchRejectsBadArguments(t *testing.T) {
	tool := NewSearchTool(SearcherFunc(func(context.Context, SearchRequest) (SearchResult, error) {
		t.Fatal("searched")
		return SearchResult{}, nil
	}), Options{Language: llm.English})
	for _, arguments := range []string{`{"query":"  "}`, `{"query":"x","recency":"hour"}`, `not json`} {
		if _, err := tool.InvokableRun(context.Background(), arguments); err == nil {
			t.Errorf("%s: no error", arguments)
		}
	}
}

func TestFetch(t *testing.T) {
	fetcher := FetcherFunc(func(_ context.Context, address string) (Document, error) {
		return Document{URL: address, Title: "T", Content: "# T"}, nil
	})
	result, err := NewFetchTool(fetcher, Options{}).InvokableRun(context.Background(), `{"url":"HTTPS://example.com/a"}`)
	if err != nil || result != `{"url":"HTTPS://example.com/a","title":"T","content":"# T"}` {
		t.Fatalf("result %s err %v", result, err)
	}
}

func TestFetchRejectsBadURL(t *testing.T) {
	tool := NewFetchTool(FetcherFunc(func(context.Context, string) (Document, error) {
		t.Fatal("fetched")
		return Document{}, nil
	}), Options{})
	for _, address := range []string{"", "file:///etc/passwd", "example.com", "https://"} {
		if _, err := tool.InvokableRun(context.Background(), `{"url":"`+address+`"}`); err == nil {
			t.Errorf("%q: no error", address)
		}
	}
}

func TestServiceErrorsReachTheModel(t *testing.T) {
	failed := errors.New("quota exceeded")
	tool := NewFetchTool(FetcherFunc(func(context.Context, string) (Document, error) { return Document{}, failed }), Options{})
	if _, err := tool.InvokableRun(context.Background(), `{"url":"https://example.com"}`); !errors.Is(err, failed) {
		t.Fatalf("err %v", err)
	}
}

func TestUnavailable(t *testing.T) {
	_, err := NewSearchTool(nil, Options{Language: llm.Chinese}).InvokableRun(context.Background(), `{"query":"x"}`)
	if err == nil || err.Error() != "该工具当前不可用" {
		t.Fatalf("err %v", err)
	}
	_, err = NewFetchTool(nil, Options{Text: Text{Unavailable: "off"}}).InvokableRun(context.Background(), `{"url":"https://example.com"}`)
	if err == nil || err.Error() != "off" {
		t.Fatalf("err %v", err)
	}
}

func TestInfo(t *testing.T) {
	info, err := NewSearchTool(nil, Options{Language: llm.Chinese, Text: Text{Query: "关键词"}}).Info(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if info.Name != SearchToolName || !strings.HasPrefix(info.Desc, "搜索互联网") {
		t.Fatalf("info %+v", info)
	}
	params, err := info.ToJSONSchema()
	if err != nil {
		t.Fatal(err)
	}
	encoded, _ := json.Marshal(params)
	if !strings.Contains(string(encoded), `"关键词"`) || !strings.Contains(string(encoded), `"required":["query"]`) {
		t.Fatalf("schema %s", encoded)
	}
	spec := FetchSpec(nil, Options{})
	if !spec.Replayable || spec.SideEffects || spec.Tool == nil {
		t.Fatalf("spec %+v", spec)
	}
}
