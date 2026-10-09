// Package llm holds the model contract shared by the runtime and the provider
// adapters, and single model calls that need no runtime: plain text
// generation and structured output.
package llm

import (
	"context"
	"unicode"

	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"
	"github.com/google/uuid"
)

// Language selects the language of the model-facing text einorun writes.
type Language string

const (
	// Chinese selects Chinese text.
	Chinese Language = "zh"
	// English selects English text.
	English Language = "en"
)

// Modality is an input modality a model accepts besides text.
type Modality string

const (
	// Image is image input.
	Image Modality = "image"
	// Audio is audio input.
	Audio Modality = "audio"
	// Video is video input.
	Video Modality = "video"
)

// ModelOptions are the request options a ModelFactory applies.
type ModelOptions struct {
	// MaxOutputTokens caps a single output; zero means no limit.
	MaxOutputTokens int
	// DisableThinking turns thinking off on models that offer a switch.
	DisableThinking bool
	// Output, when set, asks the model for a JSON object matching the schema,
	// constrained by the provider where it supports structured output.
	Output *OutputSchema
}

// ModelFactory creates a chat model component. The host decides routing,
// metering and billing inside it. The runtime calls it once per agent it
// builds (main agent, summarizer, memory selection, each sub-agent); every
// Generate and Stream call carries a ModelCallID in its context.
type ModelFactory func(ctx context.Context, options ModelOptions) (model.AgenticModel, error)

// DefaultContextWindow is the context window assumed for models that do not
// declare one.
const DefaultContextWindow = 32000

// ContextWindow returns window, or DefaultContextWindow when window is not
// positive.
func ContextWindow(window int) int {
	if window > 0 {
		return window
	}
	return DefaultContextWindow
}

// Usage is the token usage of one or more model calls. Total is always Input
// plus Output.
type Usage struct {
	Input  int `json:"input"`
	Output int `json:"output"`
	Total  int `json:"total"`
}

// Add adds other to u.
func (u *Usage) Add(other Usage) {
	u.Input += other.Input
	u.Output += other.Output
	u.Total += other.Total
}

// UsageOf returns the usage reported in meta, or zero usage when there is none.
func UsageOf(meta *schema.AgenticResponseMeta) Usage {
	if meta == nil || meta.TokenUsage == nil {
		return Usage{}
	}
	in, out := meta.TokenUsage.PromptTokens, meta.TokenUsage.CompletionTokens
	return Usage{Input: in, Output: out, Total: in + out}
}

type modelCallIDKey struct{}

// WithModelCallID returns a context carrying the ID of the model call about to
// be made.
func WithModelCallID(ctx context.Context, id string) context.Context {
	return context.WithValue(ctx, modelCallIDKey{}, id)
}

// NewModelCallID returns a new model call ID, a UUIDv7.
func NewModelCallID() string {
	return uuid.Must(uuid.NewV7()).String()
}

// ModelCallID returns the model call ID carried by ctx, or "" when there is none.
// Model components use it as the ID of their call records so that run records
// and billing records refer to the same call.
func ModelCallID(ctx context.Context) string {
	id, _ := ctx.Value(modelCallIDKey{}).(string)
	return id
}

// EstimateTokens estimates the tokens of text: one per CJK character and one
// per four other characters.
func EstimateTokens(text string) int {
	wide, narrow := 0, 0
	for _, r := range text {
		if unicode.Is(unicode.Han, r) || unicode.Is(unicode.Hiragana, r) ||
			unicode.Is(unicode.Katakana, r) || unicode.Is(unicode.Hangul, r) {
			wide++
			continue
		}
		narrow++
	}
	return wide + narrow/4
}

// Text returns the concatenated generated text of message.
func Text(message *schema.AgenticMessage) string {
	if message == nil {
		return ""
	}
	var text []byte
	for _, block := range message.ContentBlocks {
		if block.Type == schema.ContentBlockTypeAssistantGenText && block.AssistantGenText != nil {
			text = append(text, block.AssistantGenText.Text...)
		}
	}
	return string(text)
}
