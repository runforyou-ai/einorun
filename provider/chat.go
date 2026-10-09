// Package provider creates Eino chat model components for the vendors listed
// in package vendor, with einorun's conventions applied: thinking switches,
// structured output, stop reasons and an HTTP transport suited to long model
// calls.
package provider

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"time"

	"github.com/cloudwego/eino-ext/components/model/agenticark"
	"github.com/cloudwego/eino-ext/components/model/agenticclaude"
	"github.com/cloudwego/eino-ext/components/model/agenticdeepseek"
	"github.com/cloudwego/eino-ext/components/model/agenticgemini"
	"github.com/cloudwego/eino-ext/components/model/agenticopenai"
	"github.com/cloudwego/eino-ext/components/model/agenticqwen"
	"github.com/cloudwego/eino-ext/libs/acl/openai"
	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"
	"github.com/volcengine/volcengine-go-sdk/service/arkruntime/model/responses"
	"google.golang.org/genai"

	"github.com/runforyou-ai/einorun/internal/prompt"
	"github.com/runforyou-ai/einorun/llm"
	"github.com/runforyou-ai/einorun/provider/vendor"
)

// ErrStructuredStream reports a Stream call on a model created for structured
// output, which only supports Generate.
var ErrStructuredStream = errors.New("provider: structured output supports Generate only")

// ChatConfig configures a chat model.
type ChatConfig struct {
	// Brand selects the vendor preset: endpoint normalization, thinking
	// switches and the default protocol and structured-output strategy.
	Brand vendor.Brand
	// Protocol overrides the preset's protocol, for example to reach a vendor
	// through an OpenAI-compatible gateway.
	Protocol vendor.Protocol
	// BaseURL is the vendor endpoint as configured; it is normalized with the
	// brand's preset.
	BaseURL string
	APIKey  string
	// Model is the vendor's model identifier.
	Model           string
	MaxOutputTokens int
	// DisableThinking turns thinking off on protocols with a known switch:
	// DeepSeek, Qwen, Ark and compatible vendors whose preset lists thinking
	// fields.
	DisableThinking bool
	// Output, when set, asks for a JSON object matching the schema.
	Output *llm.OutputSchema
	// Structured overrides the preset's structured-output strategy. A
	// strategy the protocol cannot apply is an error.
	Structured vendor.Structured
	// Language selects model-facing text the adapter writes, such as the
	// description of the forced output tool.
	Language llm.Language
	// Transport sends the model's HTTP requests. The default waits at most two
	// minutes for response headers; the caller's context bounds the body.
	// Redirects are never followed.
	Transport http.RoundTripper
}

// defaultTransport is the default model transport.
var defaultTransport = func() *http.Transport {
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.ResponseHeaderTimeout = 2 * time.Minute
	return transport
}()

// Factory returns an llm.ModelFactory that creates models from base with the
// requested options applied: a positive MaxOutputTokens replaces base's limit,
// DisableThinking adds to base's, and Output is taken from the options only
// (base.Output is ignored, so that agents can stream).
func Factory(base ChatConfig) llm.ModelFactory {
	return func(ctx context.Context, options llm.ModelOptions) (model.AgenticModel, error) {
		config := base
		if options.MaxOutputTokens > 0 {
			config.MaxOutputTokens = options.MaxOutputTokens
		}
		config.DisableThinking = config.DisableThinking || options.DisableThinking
		config.Output = options.Output
		return NewChatModel(ctx, config)
	}
}

// NewChatModel creates the agentic chat model for config's protocol. With
// config.Output the model is constrained by the structured-output strategy;
// when the vendor rejects the constraint with HTTP 400, the same Generate call
// is sent again without it and the format relies on the instruction. The
// returned model records normalized stop reasons with llm.SetStopReason, on
// Generate results and on streamed chunks.
func NewChatModel(ctx context.Context, config ChatConfig) (model.AgenticModel, error) {
	preset, known := vendor.Of(config.Brand)
	if !known {
		return nil, fmt.Errorf("provider: unknown brand %q", config.Brand)
	}
	protocol := preset.Protocol
	if config.Protocol != "" {
		protocol = config.Protocol
	}
	// The preset's endpoint rules describe the vendor's own endpoints; an
	// endpoint reached through another protocol, such as a gateway, is used
	// as configured.
	var baseURL string
	var err error
	if protocol == preset.Protocol {
		baseURL, err = preset.CompatibleURL(config.BaseURL)
	} else {
		var parsed *url.URL
		if parsed, err = vendor.ParseURL(config.BaseURL); err == nil {
			baseURL = strings.TrimSuffix(parsed.String(), "/")
		}
	}
	if err != nil {
		return nil, err
	}
	mode := vendor.StructuredNone
	if config.Output != nil {
		mode = preset.Structured
		if config.Structured != "" {
			mode = config.Structured
		}
		if !protocol.Supports(mode) {
			return nil, fmt.Errorf("provider: protocol %s cannot apply structured output %s", protocol, mode)
		}
	}
	var plain model.AgenticModel
	if config.Output != nil {
		unconstrained := config
		unconstrained.Output = nil
		if plain, err = NewChatModel(ctx, unconstrained); err != nil {
			return nil, err
		}
	}
	var transport http.RoundTripper = defaultTransport
	if config.Transport != nil {
		transport = config.Transport
	}
	client := &http.Client{
		Transport:     observingTransport{base: transport},
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	}
	var maxTokens *int
	if config.MaxOutputTokens > 0 {
		maxTokens = &config.MaxOutputTokens
	}
	if protocol == vendor.ProtocolAnthropic && maxTokens == nil {
		return nil, fmt.Errorf("provider: protocol %s requires max output tokens", protocol)
	}
	format := responseFormat(mode, config.Output)
	var m model.AgenticModel
	switch protocol {
	case vendor.ProtocolDeepSeek:
		c := &agenticdeepseek.Config{APIKey: config.APIKey, BaseURL: baseURL, Model: config.Model, MaxTokens: maxTokens, HTTPClient: client}
		if mode == vendor.StructuredJSONObject {
			c.ResponseFormatType = agenticdeepseek.ResponseFormatTypeJSONObject
		}
		m, err = agenticdeepseek.New(ctx, c)
		// The DeepSeek component has no thinking switch; send the field with
		// every request.
		if err == nil && config.DisableThinking {
			m = &requestOptions{AgenticModel: m, options: []model.Option{
				openai.WithExtraFields(map[string]any{"thinking": map[string]any{"type": "disabled"}}),
			}}
		}
	case vendor.ProtocolQwen:
		c := &agenticqwen.Config{APIKey: config.APIKey, BaseURL: baseURL, Model: config.Model, MaxTokens: maxTokens, HTTPClient: client}
		// The component's thinking switch replaces the extra fields, so with a
		// response format the switch travels in the extra fields too, in both
		// places the component itself would put it.
		fields := map[string]any{}
		switch {
		case format != nil:
			fields["response_format"] = format
			if config.DisableThinking {
				fields["enable_thinking"] = false
				fields["chat_template_kwargs"] = map[string]any{"enable_thinking": false}
			}
		case config.DisableThinking:
			c.EnableThinking = new(false)
		}
		m, err = agenticqwen.New(ctx, c)
		if err == nil && len(fields) > 0 {
			m = &requestOptions{AgenticModel: m, options: []model.Option{openai.WithExtraFields(fields)}}
		}
	case vendor.ProtocolArk:
		c := &agenticark.Config{APIKey: config.APIKey, BaseURL: baseURL, Model: config.Model, MaxTokens: maxTokens, HTTPClient: client}
		if config.DisableThinking {
			c.Thinking = &responses.ResponsesThinking{Type: responses.ThinkingType_disabled.Enum()}
		}
		m, err = agenticark.New(ctx, c)
	case vendor.ProtocolAnthropic:
		m, err = agenticclaude.New(ctx, &agenticclaude.Config{
			APIKey: config.APIKey, BaseURL: baseURL, Model: config.Model, MaxTokens: config.MaxOutputTokens, HTTPClient: client,
		})
	case vendor.ProtocolGemini:
		var gc *genai.Client
		gc, err = genai.NewClient(ctx, &genai.ClientConfig{
			APIKey: config.APIKey, Backend: genai.BackendGeminiAPI, HTTPClient: client,
			HTTPOptions: genai.HTTPOptions{BaseURL: baseURL},
		})
		if err != nil {
			break
		}
		c := &agenticgemini.Config{Client: gc, Model: config.Model, MaxTokens: maxTokens}
		if mode == vendor.StructuredGemini {
			c.ResponseJSONSchema = config.Output.Schema
		}
		m, err = agenticgemini.New(ctx, c)
	case vendor.ProtocolOpenAI:
		c := &agenticopenai.ChatConfig{APIKey: config.APIKey, BaseURL: baseURL, Model: config.Model, MaxCompletionTokens: maxTokens, HTTPClient: client}
		if format != nil {
			c.ExtraFields = map[string]any{"response_format": format}
		}
		m, err = agenticopenai.NewChatModel(ctx, c)
	case vendor.ProtocolCompatible:
		// OpenAI-compatible vendors limit output with max_tokens in the
		// request body.
		c := &agenticopenai.ChatConfig{APIKey: config.APIKey, BaseURL: baseURL, Model: config.Model, HTTPClient: client, ExtraFields: map[string]any{}}
		if maxTokens != nil {
			c.ExtraFields["max_tokens"] = *maxTokens
		}
		if format != nil {
			c.ExtraFields["response_format"] = format
		}
		if config.DisableThinking {
			maps.Copy(c.ExtraFields, preset.DisableThinkingFields)
		}
		m, err = agenticopenai.NewChatModel(ctx, c)
	default:
		return nil, fmt.Errorf("provider: unknown protocol %q", protocol)
	}
	if err != nil {
		return nil, fmt.Errorf("provider: create %s model: %w", config.Brand, err)
	}
	if mode == vendor.StructuredForcedTool {
		m = &forcedTool{AgenticModel: m, output: config.Output, description: prompt.For(string(config.Language)).ForcedToolDescription}
	}
	return &observed{AgenticModel: m, plain: plain, brand: config.Brand, model: config.Model}, nil
}

// requestOptions adds fixed options to every call; the caller's options come
// after them and win.
type requestOptions struct {
	model.AgenticModel
	options []model.Option
}

// Generate calls the model with the fixed options.
func (m *requestOptions) Generate(ctx context.Context, input []*schema.AgenticMessage, opts ...model.Option) (*schema.AgenticMessage, error) {
	return m.AgenticModel.Generate(ctx, input, append(slices.Clone(m.options), opts...)...)
}

// Stream streams from the model with the fixed options.
func (m *requestOptions) Stream(ctx context.Context, input []*schema.AgenticMessage, opts ...model.Option) (*schema.StreamReader[*schema.AgenticMessage], error) {
	return m.AgenticModel.Stream(ctx, input, append(slices.Clone(m.options), opts...)...)
}

// responseFormat returns the response_format value of an OpenAI-compatible
// request, or nil.
func responseFormat(mode vendor.Structured, output *llm.OutputSchema) map[string]any {
	switch mode {
	case vendor.StructuredJSONSchema:
		return map[string]any{"type": "json_schema", "json_schema": map[string]any{"name": output.Name, "schema": output.Schema, "strict": true}}
	case vendor.StructuredJSONObject:
		return map[string]any{"type": "json_object"}
	}
	return nil
}

// forcedTool requests structured output by forcing a call to a single tool and
// turns the tool's arguments into text.
type forcedTool struct {
	model.AgenticModel
	output      *llm.OutputSchema
	description string
}

// Generate forces the output tool and returns its arguments as text.
func (m *forcedTool) Generate(ctx context.Context, input []*schema.AgenticMessage, opts ...model.Option) (*schema.AgenticMessage, error) {
	info := &schema.ToolInfo{Name: m.output.Name, Desc: m.description, ParamsOneOf: schema.NewParamsOneOfByJSONSchema(m.output.Schema)}
	choice := &schema.AgenticToolChoice{Type: schema.ToolChoiceForced, Forced: &schema.AgenticForcedToolChoice{Tools: []*schema.AllowedTool{{FunctionName: m.output.Name}}}}
	message, err := m.AgenticModel.Generate(ctx, input, append([]model.Option{model.WithTools([]*schema.ToolInfo{info}), model.WithAgenticToolChoice(choice)}, opts...)...)
	if err != nil {
		return nil, err
	}
	converted := *message
	converted.ContentBlocks = slices.Clone(message.ContentBlocks)
	for i, block := range converted.ContentBlocks {
		if block.Type == schema.ContentBlockTypeFunctionToolCall && block.FunctionToolCall.Name == m.output.Name {
			converted.ContentBlocks[i] = schema.NewContentBlock(&schema.AssistantGenText{Text: block.FunctionToolCall.Arguments})
		}
	}
	return &converted, nil
}

// Stream is not supported for structured output.
func (m *forcedTool) Stream(context.Context, []*schema.AgenticMessage, ...model.Option) (*schema.StreamReader[*schema.AgenticMessage], error) {
	return nil, ErrStructuredStream
}

// stopReason derives a stop reason other than StopCompleted from
// component-specific extensions.
func stopReason(message *schema.AgenticMessage) (llm.StopReason, bool) {
	if message == nil || message.ResponseMeta == nil {
		return "", false
	}
	var reason string
	switch extension := message.ResponseMeta.Extension.(type) {
	case *agenticopenai.ChatResponseMetaExtension:
		reason = extension.FinishReason
	case *agenticdeepseek.ResponseMetaExtension:
		reason = extension.FinishReason
	case *agenticqwen.ResponseMetaExtension:
		reason = extension.FinishReason
	case *agenticark.ResponseMetaExtension:
		if extension.IncompleteDetails == nil {
			return "", false
		}
		reason = extension.IncompleteDetails.Reason
		if extension.IncompleteDetails.ContentFilter != nil {
			reason = "content_filter"
		}
	default:
		return "", false
	}
	normalized := llm.NormalizeStopReason(strings.TrimSpace(reason))
	return normalized, normalized != llm.StopCompleted
}
