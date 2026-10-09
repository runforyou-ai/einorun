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

// Recency limits search results to a publication time range: a rolling
// window ending now.
type Recency string

const (
	// RecencyDay is the last 24 hours.
	RecencyDay Recency = "day"
	// RecencyWeek is the last 7 days.
	RecencyWeek Recency = "week"
	// RecencyMonth is the last 30 days.
	RecencyMonth Recency = "month"
	// RecencyYear is the last 365 days.
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

// Fetcher reads the main content of a web page. The address is a normalized
// http or https URL with a host and without user information. The tool
// checks nothing else: the fetcher decides which addresses it may reach,
// including loopback, private and metadata addresses, numeric hosts and
// ports, and checks the parsed URL rather than prefixes of the string.
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
	// CountLimited is added to the notice of a search that asked for more
	// than MaxCount results; %d is MaxCount.
	CountLimited string
	// EmptyQuery is the error of a search without a query.
	EmptyQuery string
	// BadRecency is the error of a search with an unknown time range; %q is
	// the value.
	BadRecency string
	// BadURL is the error of a fetch of an address that is not an http or
	// https URL; %q is the address.
	BadURL string
	// BadArguments is the error of arguments that are not valid JSON; %s is
	// the decode error.
	BadArguments string
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
	override(&t.CountLimited, o.Text.CountLimited)
	override(&t.EmptyQuery, o.Text.EmptyQuery)
	override(&t.BadRecency, o.Text.BadRecency)
	override(&t.BadURL, o.Text.BadURL)
	override(&t.BadArguments, o.Text.BadArguments)
	return t
}

// NewSearchTool returns the search tool. A nil searcher (or nil
// SearcherFunc) makes every call fail as unavailable, so the tool can be
// registered before a service is set up.
func NewSearchTool(searcher Searcher, opts Options) tool.InvokableTool {
	if f, ok := searcher.(SearcherFunc); ok && f == nil {
		searcher = nil
	}
	return &searchTool{searcher: searcher, text: opts.text()}
}

// NewFetchTool returns the page reading tool. A nil fetcher (or nil
// FetcherFunc) makes every call fail as unavailable.
func NewFetchTool(fetcher Fetcher, opts Options) tool.InvokableTool {
	if f, ok := fetcher.(FetcherFunc); ok && f == nil {
		fetcher = nil
	}
	return &fetchTool{fetcher: fetcher, text: opts.text()}
}

// SearchSpec returns the search tool as a tool spec: replayable and without
// side effects, so an interrupted search may be repeated. Hosts whose
// service charges per search or has effects change the returned spec's
// traits.
func SearchSpec(searcher Searcher, opts Options) einorun.ToolSpec {
	return einorun.ToolSpec{Tool: NewSearchTool(searcher, opts), Replayable: true}
}

// FetchSpec returns the page reading tool as a tool spec: replayable and
// without side effects; hosts change the returned spec's traits as needed.
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
	// Models write counts such as 5.0 and time ranges in any case.
	var input struct {
		Query   string  `json:"query"`
		Count   float64 `json:"count"`
		Recency string  `json:"recency"`
	}
	if err := json.Unmarshal([]byte(arguments), &input); err != nil {
		return "", fmt.Errorf(t.text.BadArguments, err)
	}
	request := SearchRequest{Query: strings.TrimSpace(input.Query), Recency: Recency(strings.ToLower(strings.TrimSpace(input.Recency)))}
	if request.Query == "" {
		return "", errors.New(t.text.EmptyQuery)
	}
	switch request.Recency {
	case "", RecencyDay, RecencyWeek, RecencyMonth, RecencyYear:
	default:
		return "", fmt.Errorf(t.text.BadRecency, input.Recency)
	}
	limited := input.Count > MaxCount
	switch {
	case input.Count < 1:
		request.Count = DefaultCount
	case limited:
		request.Count = MaxCount
	default:
		request.Count = int(input.Count)
	}
	result, err := t.searcher.Search(ctx, request)
	if err != nil {
		return "", err
	}
	if result.Items == nil {
		result.Items = []SearchItem{}
	}
	if limited {
		result.Notice = strings.TrimSpace(result.Notice + " " + fmt.Sprintf(t.text.CountLimited, MaxCount))
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
		return "", fmt.Errorf(t.text.BadArguments, err)
	}
	parsed, err := url.Parse(strings.TrimSpace(input.URL))
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Hostname() == "" || parsed.User != nil {
		return "", fmt.Errorf(t.text.BadURL, input.URL)
	}
	document, err := t.fetcher.Fetch(ctx, parsed.String())
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
