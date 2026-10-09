package provider

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"mime"
	"net/http"
	"strings"

	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"
	schemaopenai "github.com/cloudwego/eino/schema/openai"

	"github.com/runforyou-ai/einorun/llm"
	"github.com/runforyou-ai/einorun/provider/vendor"
)

// unsupportedConstraintMarkers appear in HTTP 400 bodies when a vendor rejects
// a structured-output constraint. Matched in lower case.
var unsupportedConstraintMarkers = []string{
	"response_format", "json_schema", "json_object", "json mode", "response_schema",
	"responsejsonschema", "response_mime_type", "tool_choice", "structured output",
}

type exchangeKey struct{}

// exchange records the last response of a Generate call: status, the body of
// an HTTP 400 and the refusal of a Chat Completions response.
type exchange struct {
	status    int
	errorBody []byte
	refusal   string
}

// unsupportedConstraint reports whether the vendor rejected the request's
// structured-output constraint.
func (e *exchange) unsupportedConstraint() bool {
	if e.status != http.StatusBadRequest {
		return false
	}
	body := strings.ToLower(string(e.errorBody))
	for _, marker := range unsupportedConstraintMarkers {
		if strings.Contains(body, marker) {
			return true
		}
	}
	return false
}

// observingTransport records responses for the exchange carried by the request
// context; bodies are handed back unchanged.
type observingTransport struct {
	base http.RoundTripper
}

// RoundTrip sends the request and records the response.
func (t observingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	response, err := t.base.RoundTrip(req)
	observed, ok := req.Context().Value(exchangeKey{}).(*exchange)
	if err != nil || !ok {
		return response, err
	}
	observed.status = response.StatusCode
	mediaType, _, _ := mime.ParseMediaType(response.Header.Get("Content-Type"))
	if response.StatusCode != http.StatusBadRequest && (response.StatusCode != http.StatusOK || mediaType != "application/json") {
		return response, nil
	}
	body, err := io.ReadAll(response.Body)
	_ = response.Body.Close()
	if err != nil {
		return nil, err
	}
	response.Body = io.NopCloser(bytes.NewReader(body))
	if response.StatusCode == http.StatusBadRequest {
		observed.errorBody = body
		return response, nil
	}
	// The first non-empty message.refusal of a Chat Completions response.
	var completion struct {
		Choices []struct {
			Message struct {
				Refusal string `json:"refusal"`
			} `json:"message"`
		} `json:"choices"`
	}
	if json.Unmarshal(body, &completion) == nil {
		for _, choice := range completion.Choices {
			if choice.Message.Refusal != "" {
				observed.refusal = choice.Message.Refusal
				break
			}
		}
	}
	return response, nil
}

// observed adds refusals the component dropped, normalizes stop reasons, and
// falls back to unconstrained output when a vendor rejects the structured
// constraint.
type observed struct {
	model.AgenticModel
	plain model.AgenticModel // the same model without constraint; nil without structured output
	brand vendor.Brand
	model string
}

// Generate requests a complete output.
func (m *observed) Generate(ctx context.Context, input []*schema.AgenticMessage, opts ...model.Option) (*schema.AgenticMessage, error) {
	ex := &exchange{}
	message, err := m.AgenticModel.Generate(context.WithValue(ctx, exchangeKey{}, ex), input, opts...)
	if err != nil && m.plain != nil && ctx.Err() == nil && ex.unsupportedConstraint() {
		slog.WarnContext(ctx, "provider: model rejected the structured output constraint, retrying without it",
			"brand", string(m.brand), "model", m.model, "error", err)
		message, err = m.plain.Generate(ctx, input, opts...)
		return message, err
	}
	if err != nil {
		return nil, err
	}
	result := *message
	if ex.refusal != "" && llm.StopReasonOf(message) != llm.StopRefusal {
		result.ContentBlocks = append(append([]*schema.ContentBlock{}, message.ContentBlocks...), schema.NewContentBlock(&schema.AssistantGenText{
			OpenAIExtension: &schemaopenai.AssistantGenTextExtension{Refusal: &schemaopenai.OutputRefusal{Reason: ex.refusal}},
		}))
	}
	return withStopReason(&result), nil
}

// withStopReason returns message with the stop reason its component reported
// in a component-specific extension recorded for llm.StopReasonOf. Only
// reasons other than StopCompleted are recorded, so a message carries the
// key at most once even after stream chunks are concatenated.
func withStopReason(message *schema.AgenticMessage) *schema.AgenticMessage {
	reason, ok := stopReason(message)
	if !ok || llm.StopReasonOf(message) != llm.StopCompleted {
		return message
	}
	marked := *message
	marked.Extra = cloneExtra(message.Extra)
	llm.SetStopReason(&marked, reason)
	return &marked
}

// Stream streams the output and records stop reasons on the chunks that
// report them; models created for structured output only support Generate.
func (m *observed) Stream(ctx context.Context, input []*schema.AgenticMessage, opts ...model.Option) (*schema.StreamReader[*schema.AgenticMessage], error) {
	if m.plain != nil {
		return nil, ErrStructuredStream
	}
	stream, err := m.AgenticModel.Stream(ctx, input, opts...)
	if err != nil {
		return nil, err
	}
	return schema.StreamReaderWithConvert(stream, func(chunk *schema.AgenticMessage) (*schema.AgenticMessage, error) {
		return withStopReason(chunk), nil
	}), nil
}

// cloneExtra copies extra so that recording a stop reason does not change the
// component's message.
func cloneExtra(extra map[string]any) map[string]any {
	clone := make(map[string]any, len(extra)+1)
	for key, value := range extra {
		clone[key] = value
	}
	return clone
}
