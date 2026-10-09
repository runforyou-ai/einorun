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
	if err := json.Unmarshal([]byte(result), &decoded); err != nil || len(decoded.Items) != 1 || decoded.Items[0].URL != "https://go.dev" || decoded.Notice != "A search returns at most 10 results." {
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

func TestSearchLenientArguments(t *testing.T) {
	var got SearchRequest
	searcher := SearcherFunc(func(_ context.Context, request SearchRequest) (SearchResult, error) {
		got = request
		return SearchResult{}, nil
	})
	if _, err := NewSearchTool(searcher, Options{}).InvokableRun(context.Background(), `{"query":"x","count":3.0,"recency":" Week "}`); err != nil {
		t.Fatal(err)
	}
	if got != (SearchRequest{Query: "x", Count: 3, Recency: RecencyWeek}) {
		t.Fatalf("request %+v", got)
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
	result, err := NewFetchTool(fetcher, Options{}).InvokableRun(context.Background(), `{"url":" HTTPS://Example.com/a b "}`)
	if err != nil || result != `{"url":"https://Example.com/a%20b","title":"T","content":"# T"}` {
		t.Fatalf("result %s err %v", result, err)
	}
}

func TestFetchRejectsBadURL(t *testing.T) {
	tool := NewFetchTool(FetcherFunc(func(context.Context, string) (Document, error) {
		t.Fatal("fetched")
		return Document{}, nil
	}), Options{})
	for _, address := range []string{"", "file:///etc/passwd", "example.com", "https://", "https://allowed.example@evil.com/", "http://:80/"} {
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
	var none SearcherFunc
	if _, err := NewSearchTool(none, Options{}).InvokableRun(context.Background(), `{"query":"x"}`); err == nil {
		t.Fatal("nil SearcherFunc searched")
	}
	var noFetch FetcherFunc
	if _, err := NewFetchTool(noFetch, Options{}).InvokableRun(context.Background(), `{"url":"https://example.com"}`); err == nil {
		t.Fatal("nil FetcherFunc fetched")
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

func TestErrorText(t *testing.T) {
	ctx := context.Background()
	searcher := SearcherFunc(func(context.Context, SearchRequest) (SearchResult, error) { return SearchResult{}, nil })
	chinese := NewSearchTool(searcher, Options{Language: llm.Chinese})
	if _, err := chinese.InvokableRun(ctx, `{"query":"x","recency":"hour"}`); err == nil || !strings.HasPrefix(err.Error(), "recency 只能是") {
		t.Fatalf("err %v", err)
	}
	if _, err := chinese.InvokableRun(ctx, `{`); err == nil || !strings.HasPrefix(err.Error(), "参数不是有效的 JSON") {
		t.Fatalf("err %v", err)
	}
	custom := NewSearchTool(searcher, Options{Text: Text{EmptyQuery: "need a query", BadArguments: "bad: %s"}})
	if _, err := custom.InvokableRun(ctx, `{"query":""}`); err == nil || err.Error() != "need a query" {
		t.Fatalf("err %v", err)
	}
	if _, err := custom.InvokableRun(ctx, `[`); err == nil || !strings.HasPrefix(err.Error(), "bad: ") {
		t.Fatalf("err %v", err)
	}
	fetch := NewFetchTool(FetcherFunc(func(context.Context, string) (Document, error) { return Document{}, nil }), Options{Text: Text{BadURL: "no %q"}})
	if _, err := fetch.InvokableRun(ctx, `{"url":"ftp://x"}`); err == nil || err.Error() != `no "ftp://x"` {
		t.Fatalf("err %v", err)
	}
}
