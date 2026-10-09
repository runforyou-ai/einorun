package llm

import (
	"strings"

	"github.com/cloudwego/eino/schema"
)

// StopReason is why a model stopped producing output.
type StopReason string

const (
	// StopCompleted is a normal stop, or a provider that gave no reason.
	StopCompleted StopReason = ""
	// StopLength is output cut off at the output limit.
	StopLength StopReason = "length"
	// StopRefusal is a refusal or output blocked by a content policy.
	StopRefusal StopReason = "refusal"
)

// stopReasonKey is the message Extra key under which components record a
// normalized stop reason.
const stopReasonKey = "einorun_stop_reason"

// SetStopReason records reason on message. Model components whose provider
// reports stop reasons in a component-specific extension call it so that
// StopReasonOf works without knowing the component.
func SetStopReason(message *schema.AgenticMessage, reason StopReason) {
	if message.Extra == nil {
		message.Extra = map[string]any{}
	}
	message.Extra[stopReasonKey] = string(reason)
}

// StopReasonOf returns why the model stopped producing message: a reason
// recorded with SetStopReason, otherwise one derived from the OpenAI, Claude
// and Gemini extensions of the message.
func StopReasonOf(message *schema.AgenticMessage) StopReason {
	if message == nil {
		return StopCompleted
	}
	if reason, ok := message.Extra[stopReasonKey].(string); ok {
		return StopReason(reason)
	}
	for _, block := range message.ContentBlocks {
		if block.Type == schema.ContentBlockTypeAssistantGenText && block.AssistantGenText != nil &&
			block.AssistantGenText.OpenAIExtension != nil && block.AssistantGenText.OpenAIExtension.Refusal != nil {
			return StopRefusal
		}
	}
	meta := message.ResponseMeta
	if meta == nil {
		return StopCompleted
	}
	var reason string
	switch {
	case meta.OpenAIExtension != nil && meta.OpenAIExtension.IncompleteDetails != nil:
		reason = meta.OpenAIExtension.IncompleteDetails.Reason
	case meta.ClaudeExtension != nil:
		reason = meta.ClaudeExtension.StopReason
	case meta.GeminiExtension != nil:
		reason = meta.GeminiExtension.FinishReason
	}
	return NormalizeStopReason(reason)
}

// NormalizeStopReason maps a provider's finish reason to a StopReason.
func NormalizeStopReason(reason string) StopReason {
	switch strings.ToLower(reason) {
	case "length", "max_tokens", "max_output_tokens":
		return StopLength
	case "refusal", "content_filter", "safety", "prohibited_content", "blocklist", "spii", "recitation":
		return StopRefusal
	default:
		return StopCompleted
	}
}
