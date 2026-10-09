// Package web provides optional web search and page reading tools. The host
// provides the services behind them through Searcher and Fetcher; the
// package only defines the tools the model sees.
package web

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"strings"

	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/schema"

	"github.com/runforyou-ai/einorun"
	"github.com/runforyou-ai/einorun/internal/prompt"
	"github.com/runforyou-ai/einorun/llm"
)

const (
	// SearchToolName is the name of the search tool.
	SearchToolName = "web_search"
	// FetchToolName is the name of the page reading tool.
	FetchToolName = "web_fetch"
	// DefaultCount is the number of results of a search without a count.
	DefaultCount = 5
	// MaxCount is the largest number of results of a search.
	MaxCount = 10
)

// Recency limits search results to a publication time range.
type Recency string

const (
	// RecencyDay is the last day.
	RecencyDay Recency = "day"
	// RecencyWeek is the last week.
	RecencyWeek Recency = "week"
	// RecencyMonth is the last month.
	RecencyMonth Recency = "month"
	// RecencyYear is the last year.
	RecencyYear Recency = "year"
)

// SearchRequest is one search.
type SearchRequest struct {
	Query string `json:"query"`
	// Count is between 1 and MaxCount.
	Count int `json:"count"`
	// Recency is empty for any time.
	Recency Recency `json:"recency,omitempty"`
}

// SearchItem is one search result.
type SearchItem struct {
	Title    string `json:"title"`
	URL      string `json:"url"`
	Snippet  string `json:"snippet,omitempty"`
	SiteName string `json:"siteName,omitempty"`
	// PublishedAt is the publication time as the service gives it.
	PublishedAt string `json:"publishedAt,omitempty"`
}

// SearchResult is the result of a search.
type SearchResult struct {
	Items []SearchItem `json:"items"`
	// Notice tells the model about limits of the result, such as a time range
	// the service could not apply.
	Notice string `json:"notice,omitempty"`
}

// Document is the main content of a web page as Markdown.
type Document struct {
	// URL is the address the content was read from, after redirects.
	URL     string `json:"url"`
	Title   string `json:"title,omitempty"`
	Content string `json:"content"`
}

// Searcher searches the web.
type Searcher interface {
	Search(ctx context.Context, request SearchRequest) (SearchResult, error)
}

// SearcherFunc adapts a function to Searcher.
type SearcherFunc func(ctx context.Context, request SearchRequest) (SearchResult, error)

// Search calls f.
func (f SearcherFunc) Search(ctx context.Context, request SearchRequest) (SearchResult, error) {
	return f(ctx, request)
}

// Fetcher reads the main content of a web page. The address is an http or
// https URL; the fetcher decides which addresses it may reach.
type Fetcher interface {
	Fetch(ctx context.Context, address string) (Document, error)
}

// FetcherFunc adapts a function to Fetcher.
type FetcherFunc func(ctx context.Context, address string) (Document, error)

// Fetch calls f.
func (f FetcherFunc) Fetch(ctx context.Context, address string) (Document, error) {
	return f(ctx, address)
}

// Text is the model-facing text of the tools. Empty fields use the text of
// the language.
type Text struct {
	SearchDescription string
	// Query, Count and Recency describe the search parameters.
	Query   string
	Count   string
	Recency string
	// FetchDescription describes the page reading tool and URL its parameter.
	FetchDescription string
	URL              string
	// Unavailable is the error of a call when the host provides no service.
	Unavailable string
}

// Options configures the tools.
type Options struct {
	// Language selects the default text; English when empty or unknown.
	Language llm.Language
	// Text overrides the default text field by field.
	Text Text
}

// text resolves the tools' text.
func (o Options) text() prompt.Web {
	t := prompt.WebFor(string(o.Language))
	override := func(dst *string, src string) {
		if src != "" {
			*dst = src
		}
	}
	override(&t.SearchDescription, o.Text.SearchDescription)
	override(&t.Query, o.Text.Query)
	override(&t.Count, o.Text.Count)
	override(&t.Recency, o.Text.Recency)
	override(&t.FetchDescription, o.Text.FetchDescription)
	override(&t.URL, o.Text.URL)
	override(&t.Unavailable, o.Text.Unavailable)
	return t
}

// NewSearchTool returns the search tool. A nil searcher makes every call fail
// as unavailable, so the tool can be registered before a service is set up.
func NewSearchTool(searcher Searcher, opts Options) tool.InvokableTool {
	return &searchTool{searcher: searcher, text: opts.text()}
}

// NewFetchTool returns the page reading tool. A nil fetcher makes every call
// fail as unavailable.
func NewFetchTool(fetcher Fetcher, opts Options) tool.InvokableTool {
	return &fetchTool{fetcher: fetcher, text: opts.text()}
}

// SearchSpec returns the search tool as a tool spec: replayable and without
// side effects.
func SearchSpec(searcher Searcher, opts Options) einorun.ToolSpec {
	return einorun.ToolSpec{Tool: NewSearchTool(searcher, opts), Replayable: true}
}

// FetchSpec returns the page reading tool as a tool spec: replayable and
// without side effects.
func FetchSpec(fetcher Fetcher, opts Options) einorun.ToolSpec {
	return einorun.ToolSpec{Tool: NewFetchTool(fetcher, opts), Replayable: true}
}

// searchTool is the search tool.
type searchTool struct {
	searcher Searcher
	text     prompt.Web
}

// Info describes the tool.
func (t *searchTool) Info(context.Context) (*schema.ToolInfo, error) {
	return &schema.ToolInfo{Name: SearchToolName, Desc: t.text.SearchDescription, ParamsOneOf: schema.NewParamsOneOfByParams(map[string]*schema.ParameterInfo{
		"query":   {Type: schema.String, Desc: t.text.Query, Required: true},
		"count":   {Type: schema.Integer, Desc: t.text.Count},
		"recency": {Type: schema.String, Desc: t.text.Recency, Enum: []string{string(RecencyDay), string(RecencyWeek), string(RecencyMonth), string(RecencyYear)}},
	})}, nil
}

// InvokableRun searches with the model's arguments.
func (t *searchTool) InvokableRun(ctx context.Context, arguments string, _ ...tool.Option) (string, error) {
	if t.searcher == nil {
		return "", errors.New(t.text.Unavailable)
	}
	var request SearchRequest
	if err := json.Unmarshal([]byte(arguments), &request); err != nil {
		return "", fmt.Errorf("decode arguments: %w", err)
	}
	request.Query = strings.TrimSpace(request.Query)
	if request.Query == "" {
		return "", errors.New(t.text.EmptyQuery)
	}
	switch request.Recency {
	case "", RecencyDay, RecencyWeek, RecencyMonth, RecencyYear:
	default:
		return "", fmt.Errorf(t.text.BadRecency, request.Recency)
	}
	if request.Count <= 0 {
		request.Count = DefaultCount
	}
	request.Count = min(request.Count, MaxCount)
	result, err := t.searcher.Search(ctx, request)
	if err != nil {
		return "", err
	}
	if result.Items == nil {
		result.Items = []SearchItem{}
	}
	return encode(result)
}

// fetchTool is the page reading tool.
type fetchTool struct {
	fetcher Fetcher
	text    prompt.Web
}

// Info describes the tool.
func (t *fetchTool) Info(context.Context) (*schema.ToolInfo, error) {
	return &schema.ToolInfo{Name: FetchToolName, Desc: t.text.FetchDescription, ParamsOneOf: schema.NewParamsOneOfByParams(map[string]*schema.ParameterInfo{
		"url": {Type: schema.String, Desc: t.text.URL, Required: true},
	})}, nil
}

// InvokableRun reads the page the model names.
func (t *fetchTool) InvokableRun(ctx context.Context, arguments string, _ ...tool.Option) (string, error) {
	if t.fetcher == nil {
		return "", errors.New(t.text.Unavailable)
	}
	var input struct {
		URL string `json:"url"`
	}
	if err := json.Unmarshal([]byte(arguments), &input); err != nil {
		return "", fmt.Errorf("decode arguments: %w", err)
	}
	address := strings.TrimSpace(input.URL)
	parsed, err := url.Parse(address)
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" {
		return "", fmt.Errorf(t.text.BadURL, input.URL)
	}
	document, err := t.fetcher.Fetch(ctx, address)
	if err != nil {
		return "", err
	}
	return encode(document)
}

// encode returns v as JSON.
func encode(v any) (string, error) {
	encoded, err := json.Marshal(v)
	if err != nil {
		return "", err
	}
	return string(encoded), nil
}
