package einorun

import (
	"context"
	"fmt"
	"log/slog"
	"slices"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/cloudwego/eino/adk"
	"github.com/cloudwego/eino/adk/middlewares/patchtoolcalls"
	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"

	"github.com/runforyou-ai/einorun/internal/prompt"
	"github.com/runforyou-ai/einorun/llm"
)

// emptyRetryLimit is how often one model call is retried when it produced
// only reasoning.
const emptyRetryLimit = 1

// argumentsNormalizer replaces empty tool arguments with an empty JSON object,
// so that requests serialized with omitempty keep the arguments field.
type argumentsNormalizer struct {
	adk.TypedBaseChatModelAgentMiddleware[*schema.AgenticMessage]
}

// BeforeModelRewriteState rewrites empty arguments on copies of the messages.
func (*argumentsNormalizer) BeforeModelRewriteState(ctx context.Context, state *adk.TypedChatModelAgentState[*schema.AgenticMessage], _ *adk.TypedModelContext[*schema.AgenticMessage]) (context.Context, *adk.TypedChatModelAgentState[*schema.AgenticMessage], error) {
	for i, message := range state.Messages {
		if !slices.ContainsFunc(message.ContentBlocks, func(b *schema.ContentBlock) bool {
			return b != nil && b.Type == schema.ContentBlockTypeFunctionToolCall && b.FunctionToolCall.Arguments == ""
		}) {
			continue
		}
		normalized := *message
		normalized.ContentBlocks = slices.Clone(message.ContentBlocks)
		for j, b := range normalized.ContentBlocks {
			if b == nil || b.Type != schema.ContentBlockTypeFunctionToolCall {
				continue
			}
			call := *b.FunctionToolCall
			if call.Arguments == "" {
				call.Arguments = "{}"
			}
			filled := *b
			filled.FunctionToolCall = &call
			normalized.ContentBlocks[j] = &filled
		}
		state.Messages[i] = &normalized
	}
	return ctx, state, nil
}

// newPatchHandler patches calls without a result before every model call:
// with the result results returns for them, or a cancellation note. Orphaned
// and duplicate results are removed.
func newPatchHandler(ctx context.Context, text *prompt.Runtime, results func(callID string) (string, bool)) (adk.TypedChatModelAgentMiddleware[*schema.AgenticMessage], error) {
	handler, err := patchtoolcalls.NewTyped[*schema.AgenticMessage](ctx, &patchtoolcalls.Config{
		RemoveOrphanResults: true, RemoveDuplicateResults: true,
		PatchedToolResultGenerator: func(_ context.Context, name, callID string, _ *schema.ToolArgument) (*patchtoolcalls.PatchedToolResult, error) {
			if results != nil {
				if result, ok := results(callID); ok {
					return &patchtoolcalls.PatchedToolResult{Content: result}, nil
				}
			}
			return &patchtoolcalls.PatchedToolResult{Content: fmt.Sprintf(text.Cancelled, name, callID)}, nil
		},
	})
	if err != nil {
		return nil, fmt.Errorf("einorun: create the tool call patch middleware: %w", err)
	}
	return handler, nil
}

// modelRetry decides whether a model call is retried. A failing call that
// carried media is retried without media, and media stays off for the rest
// of the run; output with only reasoning is retried once. Discarded output
// counts in the usage.
type modelRetry struct {
	runID   string
	enabled *atomic.Bool
	text    *prompt.Runtime

	// discard, when set, takes the usage of discarded outputs in place of
	// the retry's own count.
	discard func(llm.Usage)

	mu      sync.Mutex
	retries int
	usage   llm.Usage
}

// config returns the ADK retry configuration.
func (m *modelRetry) config() *adk.TypedModelRetryConfig[*schema.AgenticMessage] {
	return &adk.TypedModelRetryConfig[*schema.AgenticMessage]{MaxRetries: emptyRetryLimit + 1, ShouldRetry: m.shouldRetry}
}

// shouldRetry decides one retry.
func (m *modelRetry) shouldRetry(ctx context.Context, attempt *adk.TypedRetryContext[*schema.AgenticMessage]) *adk.TypedRetryDecision[*schema.AgenticMessage] {
	m.mu.Lock()
	defer m.mu.Unlock()
	if attempt.RetryAttempt == 1 {
		m.retries = 0
	}
	if ctx.Err() != nil {
		return nil
	}
	var decision *adk.TypedRetryDecision[*schema.AgenticMessage]
	output := attempt.OutputMessage
	switch {
	case attempt.Err != nil:
		if !carriesMedia(attempt.InputMessages) {
			return nil
		}
		slog.WarnContext(ctx, "einorun: a model call carrying media failed, retrying without media for the rest of the run",
			"run_id", m.runID, "error", attempt.Err)
		m.enabled.Store(false)
		decision = &adk.TypedRetryDecision[*schema.AgenticMessage]{
			Retry: true, ModifiedInputMessages: withoutMedia(attempt.InputMessages, m.text), PersistModifiedInputMessages: true,
		}
	case output != nil && (hasToolCalls(output) || strings.TrimSpace(llm.Text(output)) != ""), m.retries >= emptyRetryLimit:
		return nil
	default:
		m.retries++
		slog.WarnContext(ctx, "einorun: the model produced no text, retrying the call", "run_id", m.runID, "attempt", m.retries)
		decision = &adk.TypedRetryDecision[*schema.AgenticMessage]{Retry: true}
	}
	if output != nil {
		if m.discard != nil {
			m.discard(llm.UsageOf(output.ResponseMeta))
		} else {
			m.usage.Add(llm.UsageOf(output.ResponseMeta))
		}
	}
	return decision
}

// discarded returns the usage of discarded outputs.
func (m *modelRetry) discarded() llm.Usage {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.usage
}

// inputCapture records the input of the latest model call, after every
// rewrite, for guards.
type inputCapture struct {
	adk.TypedBaseChatModelAgentMiddleware[*schema.AgenticMessage]
	mu    sync.Mutex
	input []*schema.AgenticMessage
}

// capturingModel records its input before calling the model.
type capturingModel struct {
	model.BaseModel[*schema.AgenticMessage]
	capture *inputCapture
}

// WrapModel wraps the model.
func (c *inputCapture) WrapModel(_ context.Context, m model.BaseModel[*schema.AgenticMessage], _ *adk.TypedModelContext[*schema.AgenticMessage]) (model.BaseModel[*schema.AgenticMessage], error) {
	return &capturingModel{BaseModel: m, capture: c}, nil
}

// Generate records the input.
func (m *capturingModel) Generate(ctx context.Context, input []*schema.AgenticMessage, opts ...model.Option) (*schema.AgenticMessage, error) {
	m.capture.set(input)
	return m.BaseModel.Generate(ctx, input, opts...)
}

// Stream records the input.
func (m *capturingModel) Stream(ctx context.Context, input []*schema.AgenticMessage, opts ...model.Option) (*schema.StreamReader[*schema.AgenticMessage], error) {
	m.capture.set(input)
	return m.BaseModel.Stream(ctx, input, opts...)
}

// set records input.
func (c *inputCapture) set(input []*schema.AgenticMessage) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.input = slices.Clone(input)
}

// latest returns the latest input.
func (c *inputCapture) latest() []*schema.AgenticMessage {
	c.mu.Lock()
	defer c.mu.Unlock()
	return slices.Clone(c.input)
}

// observerMiddleware reports finalized model outputs to output observers.
type observerMiddleware struct {
	adk.TypedBaseChatModelAgentMiddleware[*schema.AgenticMessage]
	scope     AgentScope
	observers []OutputObserver
}

// AfterModelRewriteState reports the output.
func (o *observerMiddleware) AfterModelRewriteState(ctx context.Context, state *adk.TypedChatModelAgentState[*schema.AgenticMessage], _ *adk.TypedModelContext[*schema.AgenticMessage]) (context.Context, *adk.TypedChatModelAgentState[*schema.AgenticMessage], error) {
	output := state.Messages[len(state.Messages)-1]
	for _, observer := range o.observers {
		observer.AfterModelOutput(ctx, o.scope, output)
	}
	return ctx, state, nil
}
