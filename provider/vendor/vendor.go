// Package vendor describes model vendors: how their endpoints are addressed and
// where they differ from the OpenAI-compatible conventions. It has no
// dependencies beyond the standard library, so packages that only need these
// facts do not pull in the chat model SDKs.
package vendor

import (
	"errors"
	"fmt"
	"net/url"
	"strings"
)

// Brand is a model vendor.
type Brand string

// Supported brands.
const (
	OpenAI     Brand = "openai"
	Anthropic  Brand = "anthropic"
	Google     Brand = "google"
	DeepSeek   Brand = "deepseek"
	Alibaba    Brand = "alibaba"
	Moonshot   Brand = "moonshot"
	Zhipu      Brand = "zhipu"
	Volcengine Brand = "volcengine"
	MiniMax    Brand = "minimax"
	XAI        Brand = "xai"
	Mistral    Brand = "mistral"
	OpenRouter Brand = "openrouter"
	TypeSafe   Brand = "typesafe"
	// Ollama is a self-hosted Ollama server; it may run without credentials.
	Ollama Brand = "ollama"
	// OpenAICompatible is any self-hosted service speaking the OpenAI API; it
	// may run without credentials.
	OpenAICompatible Brand = "openai_compatible"
)

// Brands lists every supported brand.
var Brands = []Brand{
	OpenAI, Anthropic, Google, DeepSeek, Alibaba, Moonshot, Zhipu, Volcengine,
	MiniMax, XAI, Mistral, OpenRouter, TypeSafe, Ollama, OpenAICompatible,
}

// Structured is how a vendor constrains a model to produce a JSON object.
type Structured string

const (
	// StructuredNone relies on the instruction alone.
	StructuredNone Structured = "none"
	// StructuredJSONSchema uses response_format json_schema in strict mode.
	StructuredJSONSchema Structured = "json_schema"
	// StructuredJSONObject uses response_format json_object; the structure
	// relies on the instruction.
	StructuredJSONObject Structured = "json_object"
	// StructuredForcedTool forces a call to a single tool whose arguments are
	// the output.
	StructuredForcedTool Structured = "forced_tool"
	// StructuredGemini uses Gemini's response JSON schema.
	StructuredGemini Structured = "gemini"
)

// RerankProtocol is the request format of a rerank endpoint.
type RerankProtocol string

const (
	// RerankCompatible is the rerank format shared by Cohere, Jina and others.
	RerankCompatible RerankProtocol = "compatible"
	// RerankDashScope is Alibaba DashScope's native rerank format.
	RerankDashScope RerankProtocol = "dashscope"
)

// Preset is what einorun knows about a brand.
type Preset struct {
	Brand Brand
	// compatiblePath rewrites the endpoint path (without a trailing slash) to
	// the OpenAI-compatible entry point; nil keeps it.
	compatiblePath func(path string) string
	// RequiresMaxOutputTokens means every chat request must carry an output
	// limit.
	RequiresMaxOutputTokens bool
	// DisableThinkingFields are added to the request body of OpenAI-compatible
	// requests to turn thinking off.
	DisableThinkingFields map[string]any
	// Structured is the default structured-output strategy.
	Structured Structured
	// Rerank is the rerank request format.
	Rerank RerankProtocol
	// Discovery reports whether the brand's model list can be read from the
	// service with package discovery.
	Discovery bool
	// CredentialsOptional means the service may run without an API key.
	CredentialsOptional bool
}

var presets = map[Brand]Preset{
	OpenAI:    {Structured: StructuredJSONSchema},
	Anthropic: {Structured: StructuredForcedTool, RequiresMaxOutputTokens: true},
	Google:    {Structured: StructuredGemini},
	DeepSeek:  {Structured: StructuredJSONObject},
	Alibaba: {
		Structured: StructuredJSONObject,
		Rerank:     RerankDashScope,
		compatiblePath: func(path string) string {
			switch {
			case strings.HasSuffix(path, "/compatible-mode/v1"):
				return path
			case strings.HasSuffix(path, "/api/v1"):
				return strings.TrimSuffix(path, "/api/v1") + "/compatible-mode/v1"
			default:
				return path + "/compatible-mode/v1"
			}
		},
	},
	Moonshot:   {Structured: StructuredJSONObject},
	Zhipu:      {Structured: StructuredJSONObject, DisableThinkingFields: map[string]any{"thinking": map[string]any{"type": "disabled"}}},
	Volcengine: {},
	MiniMax:    {},
	XAI:        {Structured: StructuredJSONSchema},
	Mistral:    {Structured: StructuredJSONSchema},
	OpenRouter: {
		Structured: StructuredJSONSchema, Discovery: true,
		DisableThinkingFields: map[string]any{"reasoning": map[string]any{"effort": "none"}},
	},
	TypeSafe: {},
	Ollama: {
		Discovery: true, CredentialsOptional: true,
		compatiblePath: func(path string) string { return strings.TrimSuffix(path, "/v1") + "/v1" },
	},
	OpenAICompatible: {Discovery: true, CredentialsOptional: true},
}

// Of returns the preset of brand and whether the brand is known.
func Of(brand Brand) (Preset, bool) {
	preset, ok := presets[brand]
	preset.Brand = brand
	if preset.Structured == "" {
		preset.Structured = StructuredNone
	}
	if preset.Rerank == "" {
		preset.Rerank = RerankCompatible
	}
	return preset, ok
}

// CompatibleURL normalizes endpoint to the brand's OpenAI-compatible entry
// point: trailing slashes are removed and brands with a separate compatible
// path get it.
func (p Preset) CompatibleURL(endpoint string) (string, error) {
	parsed, err := ParseURL(endpoint)
	if err != nil {
		return "", err
	}
	if p.compatiblePath == nil {
		return strings.TrimSuffix(parsed.String(), "/"), nil
	}
	parsed.Path = p.compatiblePath(strings.TrimSuffix(parsed.Path, "/"))
	parsed.RawPath = ""
	return parsed.String(), nil
}

// ParseURL parses an endpoint that must have a scheme and a host.
func ParseURL(endpoint string) (*url.URL, error) {
	parsed, err := url.Parse(strings.TrimSpace(endpoint))
	if err != nil {
		return nil, fmt.Errorf("vendor: parse endpoint: %w", err)
	}
	if parsed.Scheme == "" || parsed.Host == "" {
		return nil, errors.New("vendor: endpoint must include scheme and host")
	}
	return parsed, nil
}

// AppendPath appends path to endpoint, keeping the endpoint's own path.
func AppendPath(endpoint, path string) (string, error) {
	parsed, err := ParseURL(endpoint)
	if err != nil {
		return "", err
	}
	escaped := strings.TrimLeft(path, "/")
	decoded, err := url.PathUnescape(escaped)
	if err != nil {
		return "", err
	}
	parsed.RawPath = strings.TrimSuffix(parsed.EscapedPath(), "/") + "/" + escaped
	parsed.Path = strings.TrimSuffix(parsed.Path, "/") + "/" + decoded
	return parsed.String(), nil
}
