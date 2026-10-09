package prompt

// Web is the model-facing text of the web tools in one language.
type Web struct {
	SearchDescription string
	Query             string
	Count             string
	Recency           string
	FetchDescription  string
	URL               string
	// Unavailable fails a call of a tool the host provides no service for.
	Unavailable string
	// CountLimited notes a search that asked for too many results. %d is the
	// largest count.
	CountLimited string
	// EmptyQuery fails a search without a query.
	EmptyQuery string
	// BadRecency fails a search with an unknown time range. %q is the value.
	BadRecency string
	// BadArguments fails a call whose arguments are not valid JSON. %s is
	// the decode error.
	BadArguments string
	// BadURL fails a fetch of an address that is not an http or https URL
	// with a host and without user information.
	// %q is the address.
	BadURL string
}

var webs = map[string]*Web{
	"zh": {
		SearchDescription: "搜索互联网上的公开信息，返回标题、链接、摘要和发布时间。",
		Query:             "搜索关键词",
		Count:             "返回的结果条数，默认 5，最多 10",
		Recency:           "只返回最近一天、一周、一月或一年内发布的结果，不传表示不限时间",
		FetchDescription:  "读取一个公开网页的正文，去掉导航和页脚等页面结构后以 Markdown 返回。",
		URL:               "网页地址，以 http:// 或 https:// 开头",
		Unavailable:       "该工具当前不可用",
		CountLimited:      "单次搜索最多返回 %d 条结果。",
		EmptyQuery:        "query 不能为空",
		BadRecency:        "recency 只能是 day、week、month 或 year，收到 %q",
		BadArguments:      "参数不是有效的 JSON：%s",
		BadURL:            "url 必须是以 http:// 或 https:// 开头的网页地址，收到 %q",
	},
	"en": {
		SearchDescription: "Search the public web. Returns titles, links, snippets and publication dates.",
		Query:             "Search query",
		Count:             "Number of results, 5 by default, at most 10",
		Recency:           "Only return results published within the last day, week, month or year; omit for any time",
		FetchDescription:  "Read the main content of a public web page as Markdown, without navigation, footers and other page chrome.",
		URL:               "Page address, starting with http:// or https://",
		Unavailable:       "This tool is not available right now",
		CountLimited:      "A search returns at most %d results.",
		EmptyQuery:        "query must not be empty",
		BadRecency:        "recency must be day, week, month or year, got %q",
		BadArguments:      "the arguments are not valid JSON: %s",
		BadURL:            "url must be a web address starting with http:// or https://, got %q",
	},
}

// WebFor returns the web tool text for language, falling back to English.
func WebFor(language string) Web {
	if w, ok := webs[language]; ok {
		return *w
	}
	return *webs["en"]
}
