package einorun

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"maps"
	"regexp"
	"slices"
	"strings"
	"sync"

	"github.com/cloudwego/eino/adk"
	"github.com/cloudwego/eino/adk/filesystem"
	fsmiddleware "github.com/cloudwego/eino/adk/middlewares/filesystem"
	"github.com/cloudwego/eino/adk/middlewares/reduction"
	"github.com/cloudwego/eino/adk/middlewares/summarization"
	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"

	"github.com/runforyou-ai/einorun/internal/prompt"
	"github.com/runforyou-ai/einorun/llm"
)

// OffloadReadTool is the tool the model reads offloaded results with. It is
// registered in every run.
const OffloadReadTool = "read_offloaded_tool_result"

const (
	// offloadDir and clearDir hold offloaded and cleared results by call ID.
	offloadDir = "/trunc/"
	clearDir   = "/clear/"
	// bytesPerToken converts token budgets to bytes for CJK text, so byte
	// limits never exceed the token budget.
	bytesPerToken = 3
	// summaryExtraKey marks summary messages.
	summaryExtraKey = "einorun_summary"
)

// OffloadedPath returns the path under which the result of the call with the
// provider call ID is offloaded when it is too large.
func OffloadedPath(callID string) string { return offloadDir + callID }

// summaryAnalysis matches the analysis section of a summary.
var summaryAnalysis = regexp.MustCompile(`(?s)<analysis>.*?</analysis>`)

// ContextPolicy tunes context management. Zero fields take the defaults.
type ContextPolicy struct {
	// ResultPercent is the share of the context window a single tool result
	// may take before it is offloaded with a preview (default 10).
	ResultPercent int
	// MinOffloadBytes is the smallest offload threshold (default 4000).
	MinOffloadBytes int
	// ClearPercent is the share of the window at which older tool calls are
	// cleared (default 75).
	ClearPercent int
	// KeepRounds is how many recent tool rounds clearing keeps (default 2).
	KeepRounds int
	// SummaryPercent is the share of the window a summary may take, capped at
	// the model's output limit (default 10).
	SummaryPercent int
	// InstructionTokens is the estimate of the summary instruction (default
	// 2000).
	InstructionTokens int
	// TokenCounter replaces the default estimate.
	TokenCounter func(ctx context.Context, messages []*schema.AgenticMessage, tools []*schema.ToolInfo) (int64, error)
	// OmitManagementNote leaves the context-management note out of the
	// instruction.
	OmitManagementNote bool
}

// withDefaults fills in zero fields.
func (p ContextPolicy) withDefaults() ContextPolicy {
	defaults := func(v *int, d int) {
		if *v <= 0 {
			*v = d
		}
	}
	defaults(&p.ResultPercent, 10)
	defaults(&p.MinOffloadBytes, 4000)
	defaults(&p.ClearPercent, 75)
	defaults(&p.KeepRounds, 2)
	defaults(&p.SummaryPercent, 10)
	defaults(&p.InstructionTokens, 2000)
	if p.TokenCounter == nil {
		p.TokenCounter = CountTokens
	}
	return p
}

// summaryOutput returns the output limit of summary calls.
func (p ContextPolicy) summaryOutput(window, maxOutput int) int {
	output := window * p.SummaryPercent / 100
	if maxOutput > 0 {
		output = min(output, maxOutput)
	}
	return output
}

// CountTokens estimates the tokens of a model context: one per CJK character
// and a quarter per other character, tool definitions as JSON, and 1280 per
// media item.
func CountTokens(_ context.Context, messages []*schema.AgenticMessage, tools []*schema.ToolInfo) (int64, error) {
	total, media := 0, 0
	for _, m := range messages {
		for _, b := range m.ContentBlocks {
			if b == nil {
				continue
			}
			switch b.Type {
			case schema.ContentBlockTypeReasoning:
				total += llm.EstimateTokens(b.Reasoning.Text)
			case schema.ContentBlockTypeUserInputText:
				total += llm.EstimateTokens(b.UserInputText.Text)
			case schema.ContentBlockTypeUserInputImage, schema.ContentBlockTypeUserInputAudio, schema.ContentBlockTypeUserInputVideo:
				media++
			case schema.ContentBlockTypeAssistantGenText:
				total += llm.EstimateTokens(b.AssistantGenText.Text)
			case schema.ContentBlockTypeFunctionToolCall:
				total += llm.EstimateTokens(b.FunctionToolCall.Name) + llm.EstimateTokens(b.FunctionToolCall.Arguments)
			case schema.ContentBlockTypeFunctionToolResult:
				for _, c := range b.FunctionToolResult.Content {
					if c != nil && c.Type == schema.FunctionToolResultContentBlockTypeText {
						total += llm.EstimateTokens(c.Text.Text)
						continue
					}
					media++
				}
			}
		}
	}
	for _, info := range tools {
		total += llm.EstimateTokens(info.Name) + llm.EstimateTokens(info.Desc)
		parameters, err := info.ToJSONSchema()
		if err != nil {
			return 0, fmt.Errorf("einorun: read tool %q parameters: %w", info.Name, err)
		}
		encoded, err := json.Marshal(parameters)
		if err != nil {
			return 0, fmt.Errorf("einorun: encode tool %q parameters: %w", info.Name, err)
		}
		total += llm.EstimateTokens(string(encoded))
	}
	return int64(total + media*mediaTokens), nil
}

// offloadStore keeps offloaded results in memory and remembers them for the
// checkpoint.
type offloadStore struct {
	*filesystem.InMemoryBackend
	mu    sync.Mutex
	files map[string]string
	// saved persists the checkpoint after a write, before the call whose
	// result was offloaded is saved with its preview.
	saved func(ctx context.Context) error
}

// newOffloadStore returns an empty store; saved, when not nil, persists the
// checkpoint after an offload.
func newOffloadStore(saved func(ctx context.Context) error) *offloadStore {
	return &offloadStore{InMemoryBackend: filesystem.NewInMemoryBackend(), files: map[string]string{}, saved: saved}
}

// Write stores a file and remembers it, atomically for snapshots.
func (s *offloadStore) Write(ctx context.Context, req *filesystem.WriteRequest) error {
	s.mu.Lock()
	if err := s.InMemoryBackend.Write(ctx, req); err != nil {
		s.mu.Unlock()
		return err
	}
	s.files[req.FilePath] = req.Content
	s.mu.Unlock()
	if s.saved != nil && strings.HasPrefix(req.FilePath, offloadDir) {
		if err := s.saved(ctx); err != nil {
			return &abortError{err: err}
		}
	}
	return nil
}

// abortError carries an error that must end the run from inside a tool call,
// such as a journal failure while offloading a result.
type abortError struct{ err error }

// Error returns the underlying error.
func (e *abortError) Error() string { return e.err.Error() }

// Unwrap returns the underlying error.
func (e *abortError) Unwrap() error { return e.err }

// restoreFile writes a saved file back without saving the checkpoint.
func (s *offloadStore) restoreFile(ctx context.Context, path, content string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.InMemoryBackend.Write(ctx, &filesystem.WriteRequest{FilePath: path, Content: content}); err != nil {
		return err
	}
	s.files[path] = content
	return nil
}

// snapshot returns the stored files.
func (s *offloadStore) snapshot() map[string]string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return maps.Clone(s.files)
}

// restore writes saved files back.
func (s *offloadStore) restore(ctx context.Context, files map[string]string) error {
	for path, content := range files {
		if err := s.restoreFile(ctx, path, content); err != nil {
			return fmt.Errorf("einorun: restore offloaded result %s: %w", path, err)
		}
	}
	return nil
}

// reductionHandlers creates the offload read tool and the offload and clear
// middleware. keep lists tools whose results are never cleared, intact tools
// whose results are never offloaded or cleared.
func (e *execution) reductionHandlers(ctx context.Context, store *offloadStore, keep, intact []string) ([]adk.TypedChatModelAgentMiddleware[*schema.AgenticMessage], error) {
	p := e.contextPolicy
	disabled := &fsmiddleware.ToolConfig{Disable: true}
	description := e.text.OffloadRead
	read, err := fsmiddleware.NewTyped[*schema.AgenticMessage](ctx, &fsmiddleware.MiddlewareConfig{
		Backend:             store,
		ReadFileToolConfig:  &fsmiddleware.ToolConfig{Name: OffloadReadTool, Desc: &description},
		LsToolConfig:        disabled,
		WriteFileToolConfig: disabled,
		EditFileToolConfig:  disabled,
		GlobToolConfig:      disabled,
		GrepToolConfig:      disabled,
	})
	if err != nil {
		return nil, fmt.Errorf("einorun: create the offload read middleware: %w", err)
	}
	clearTokens := int64(e.window) * int64(p.ClearPercent) / 100
	offloadBytes := max(e.window*p.ResultPercent/100*bytesPerToken, p.MinOffloadBytes)
	reduce, err := reduction.NewTyped(ctx, &reduction.TypedConfig[*schema.AgenticMessage]{
		Backend:                   store,
		ReadFileToolName:          OffloadReadTool,
		TruncExcludeTools:         append([]string{OffloadReadTool}, intact...),
		ClearExcludeTools:         slices.Concat([]string{OffloadReadTool}, keep, intact),
		MaxLengthForTrunc:         offloadBytes,
		MaxTokensForClear:         clearTokens,
		ClearRetentionSuffixLimit: p.KeepRounds,
		TokenCounter:              p.TokenCounter,
		GenTruncOffloadFilePath: func(ctx context.Context, detail *reduction.ToolDetail) (string, error) {
			path := OffloadedPath(detail.ToolContext.CallID)
			slog.InfoContext(ctx, "einorun: tool result offloaded", "run_id", e.request.RunID, "tool", detail.ToolContext.Name,
				"call_id", detail.ToolContext.CallID, "threshold_bytes", offloadBytes)
			return path, nil
		},
		GenClearOffloadFilePath: func(_ context.Context, detail *reduction.ToolDetail) (string, error) {
			return clearDir + detail.ToolContext.CallID, nil
		},
	})
	if err != nil {
		return nil, fmt.Errorf("einorun: create the reduction middleware: %w", err)
	}
	return []adk.TypedChatModelAgentMiddleware[*schema.AgenticMessage]{read, reduce}, nil
}

// compaction tells the turn history that a summary replaced part of the
// context. With keepFromCallID empty, history before keepFromID becomes the
// summary and kept; otherwise the history becomes the summary and kept, and
// this turn's messages start at the output that made the call
// keepFromCallID.
type compaction struct {
	summary        *schema.AgenticMessage
	keepFromID     string
	keepFromCallID string
	kept           []*schema.AgenticMessage
}

// summarizer summarizes older context when it exceeds the threshold. It keeps
// the newest input of the turn and everything after it; when that does not
// fit, it keeps the newest input, pinned calls and the latest round.
type summarizer struct {
	adk.TypedChatModelAgentMiddleware[*schema.AgenticMessage]
	inner   *summarization.TypedMiddleware[*schema.AgenticMessage]
	trigger int64
	reserve func() int64 // tokens extensions add to model calls outside the state
	output  int
	counter func(ctx context.Context, messages []*schema.AgenticMessage, tools []*schema.ToolInfo) (int64, error)
	pin     func(messages []*schema.AgenticMessage) map[string]bool
	runID   string
	text    *prompt.Runtime

	mu         sync.Mutex
	keepFromID string
}

// newSummarizer creates the summarizer; summary calls use summaryModel.
func (e *execution) newSummarizer(ctx context.Context, summaryModel model.AgenticModel, reserve func() int64) (*summarizer, error) {
	p := e.contextPolicy
	text := e.text
	config := &summarization.TypedConfig[*schema.AgenticMessage]{
		Model: summaryModel,
		Finalize: func(_ context.Context, _ []*schema.AgenticMessage, summary *schema.AgenticMessage) ([]*schema.AgenticMessage, error) {
			body := strings.TrimSpace(summaryAnalysis.ReplaceAllString(llm.Text(summary), ""))
			if body == "" {
				return nil, fmt.Errorf("einorun: the context summary is empty")
			}
			m := schema.UserAgenticMessage(text.SummaryPreamble + "\n\n" + body)
			m.Extra = map[string]any{summaryExtraKey: true}
			return []*schema.AgenticMessage{m}, nil
		},
	}
	if p.OmitManagementNote {
		config.CustomFormatContextManagementInstruction = func(context.Context) string { return "" }
	}
	handler, err := summarization.NewTyped(ctx, config)
	if err != nil {
		return nil, fmt.Errorf("einorun: create the summarization middleware: %w", err)
	}
	output := p.summaryOutput(e.window, e.request.Model.MaxOutputTokens)
	return &summarizer{
		TypedChatModelAgentMiddleware: handler,
		inner:                         handler.(*summarization.TypedMiddleware[*schema.AgenticMessage]),
		trigger:                       int64(e.window - output - p.InstructionTokens),
		reserve:                       reserve,
		output:                        output,
		counter:                       p.TokenCounter,
		pin:                           e.pinned,
		runID:                         e.request.RunID,
		text:                          e.text,
	}, nil
}

// keepFrom records the newest input of the turn.
func (s *summarizer) keepFrom(message *schema.AgenticMessage) {
	adk.EnsureMessageID(message)
	s.mu.Lock()
	defer s.mu.Unlock()
	s.keepFromID = adk.GetMessageID(message)
}

// keptFrom returns the message ID summaries keep from.
func (s *summarizer) keptFrom() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.keepFromID
}

// restoreKeepFrom restores the message ID summaries keep from.
func (s *summarizer) restoreKeepFrom(id string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.keepFromID = id
}

// BeforeModelRewriteState summarizes when the context exceeds the threshold
// and older messages can be summarized, and tells the history.
func (s *summarizer) BeforeModelRewriteState(ctx context.Context, state *adk.TypedChatModelAgentState[*schema.AgenticMessage], _ *adk.TypedModelContext[*schema.AgenticMessage]) (context.Context, *adk.TypedChatModelAgentState[*schema.AgenticMessage], error) {
	trigger := s.trigger
	if s.reserve != nil {
		trigger -= s.reserve()
	}
	tokens, err := s.counter(ctx, state.Messages, state.ToolInfos)
	if err != nil {
		return ctx, nil, err
	}
	if tokens <= trigger {
		return ctx, state, nil
	}
	keepFromID := s.keptFrom()
	start := 0
	for start < len(state.Messages) && state.Messages[start].Role == schema.AgenticRoleTypeSystem {
		start++
	}
	latest, round := -1, -1
	for i := start; i < len(state.Messages); i++ {
		if keepFromID != "" && adk.GetMessageID(state.Messages[i]) == keepFromID {
			latest = i
		}
		if state.Messages[i].Role == schema.AgenticRoleTypeAssistant && hasToolCalls(state.Messages[i]) {
			round = i
		}
	}
	system := state.Messages[:start]
	var event *compaction
	var summarize, kept []*schema.AgenticMessage
	keepAll := latest > start && round <= latest
	var keptPinned, pinnedRest []*schema.AgenticMessage
	if latest > start {
		pinned := s.pin(state.Messages[:latest])
		for i := start; i < latest; i++ {
			message, rest := splitPinned(state.Messages[i], pinned)
			if message != nil {
				keptPinned = append(keptPinned, message)
			}
			if rest != nil {
				pinnedRest = append(pinnedRest, rest)
			}
		}
	}
	if latest > start && round > latest {
		// Keep everything from the newest input, with the pinned calls, when
		// it fits with a summary.
		tail, err := s.counter(ctx, slices.Concat(system, keptPinned, state.Messages[latest:]), state.ToolInfos)
		if err != nil {
			return ctx, nil, err
		}
		keepAll = tail+int64(s.output) <= trigger
	}
	if keepAll {
		summarize = pinnedRest
		kept = append(slices.Clone(keptPinned), state.Messages[latest:]...)
		event = &compaction{keepFromID: keepFromID, kept: keptPinned}
	} else {
		if round <= start {
			return ctx, state, nil
		}
		pinned := s.pin(state.Messages[:round])
		for i := start; i < round; i++ {
			if i == latest {
				kept = append(kept, state.Messages[i])
				continue
			}
			message, rest := splitPinned(state.Messages[i], pinned)
			if message != nil {
				kept = append(kept, message)
			}
			if rest != nil {
				summarize = append(summarize, rest)
			}
		}
		event = &compaction{keepFromCallID: toolCalls(state.Messages[round])[0].CallID, kept: kept}
		kept = append(slices.Clone(kept), state.Messages[round:]...)
	}
	// Nothing to summarize, or only the previous summary.
	if len(summarize) == 0 || (len(summarize) == 1 && summarize[0].Extra[summaryExtraKey] == true) {
		return ctx, state, nil
	}
	// The summary model reads text only.
	prefix := *state
	prefix.Messages = append(slices.Clone(system), withoutMedia(summarize, s.text)...)
	summarized, err := s.inner.Summarize(ctx, &prefix)
	if err != nil {
		return ctx, nil, err
	}
	event.summary = summarized[len(summarized)-1]
	compacted := *state
	compacted.Messages = append(append(slices.Clone(system), event.summary), kept...)
	after, _ := s.counter(ctx, compacted.Messages, compacted.ToolInfos)
	slog.InfoContext(ctx, "einorun: context summarized", "run_id", s.runID, "threshold_tokens", trigger,
		"tokens_before", tokens, "tokens_after", after, "summarized_messages", len(summarize))
	if err := adk.TypedSendEvent(ctx, &adk.TypedAgentEvent[*schema.AgenticMessage]{
		Action: &adk.AgentAction{CustomizedAction: event},
	}); err != nil {
		return ctx, nil, err
	}
	return ctx, &compacted, nil
}

// splitPinned splits message into its pinned tool calls and results and the
// rest; a part without content is nil, and a rest with only reasoning counts
// as empty.
func splitPinned(message *schema.AgenticMessage, pinned map[string]bool) (*schema.AgenticMessage, *schema.AgenticMessage) {
	var keep, rest []*schema.ContentBlock
	substantive := false
	for _, b := range message.ContentBlocks {
		switch {
		case b.Type == schema.ContentBlockTypeFunctionToolCall && pinned[b.FunctionToolCall.CallID],
			b.Type == schema.ContentBlockTypeFunctionToolResult && pinned[b.FunctionToolResult.CallID]:
			keep = append(keep, b)
		default:
			rest = append(rest, b)
			substantive = substantive || b.Type != schema.ContentBlockTypeReasoning
		}
	}
	if len(keep) == 0 {
		return nil, message
	}
	var restMessage *schema.AgenticMessage
	if substantive {
		restMessage = &schema.AgenticMessage{Role: message.Role, ContentBlocks: rest}
	}
	return &schema.AgenticMessage{Role: message.Role, ContentBlocks: keep}, restMessage
}

// pinned returns the calls kept verbatim by summaries: read-backs of
// offloaded results, calls of tools that pin their results, and calls
// extensions pin.
func (e *execution) pinned(messages []*schema.AgenticMessage) map[string]bool {
	ids := map[string]bool{}
	for _, m := range messages {
		for _, c := range toolCalls(m) {
			if entry, ok := e.recorder.entry(c.Name); (ok && entry.spec.PinInSummary) || c.Name == OffloadReadTool {
				ids[c.CallID] = true
			}
		}
	}
	for _, p := range e.extensions.pins {
		maps.Copy(ids, p.Pin(messages))
	}
	return ids
}

// compact applies a compaction to the history and returns this turn's
// messages that remain.
func (h *history) compact(event *compaction, intermediates []*schema.AgenticMessage) []*schema.AgenticMessage {
	if event.keepFromCallID == "" {
		for i, m := range h.messages {
			if adk.GetMessageID(m) == event.keepFromID {
				h.messages = slices.Concat([]*schema.AgenticMessage{event.summary}, event.kept, h.messages[i:])
				break
			}
		}
		return intermediates
	}
	makes := func(m *schema.AgenticMessage) bool {
		return m.Role == schema.AgenticRoleTypeAssistant && slices.ContainsFunc(toolCalls(m), func(c *schema.FunctionToolCall) bool { return c.CallID == event.keepFromCallID })
	}
	for i, m := range intermediates {
		if makes(m) {
			h.messages = append([]*schema.AgenticMessage{event.summary}, event.kept...)
			return intermediates[i:]
		}
	}
	// After recovery the kept round may already be in the history.
	for i, m := range h.messages {
		if makes(m) {
			h.messages = slices.Concat([]*schema.AgenticMessage{event.summary}, event.kept, h.messages[i:])
			return intermediates
		}
	}
	slog.Warn("einorun: summary anchor not found, the history keeps the full context", "call_id", event.keepFromCallID)
	return intermediates
}

// accountedModel assigns a ModelCallID to every call of a model the runtime
// uses internally and adds its usage to the run.
type accountedModel struct {
	model.AgenticModel
	mu    sync.Mutex
	usage llm.Usage
}

// Generate calls the model with a new ModelCallID and adds the usage.
func (m *accountedModel) Generate(ctx context.Context, input []*schema.AgenticMessage, opts ...model.Option) (*schema.AgenticMessage, error) {
	output, err := m.AgenticModel.Generate(llm.WithModelCallID(ctx, llm.NewModelCallID()), input, opts...)
	if output != nil {
		m.mu.Lock()
		m.usage.Add(llm.UsageOf(output.ResponseMeta))
		m.mu.Unlock()
	}
	return output, err
}

// Stream calls the model with a new ModelCallID as a single Generate and
// adds the usage.
func (m *accountedModel) Stream(ctx context.Context, input []*schema.AgenticMessage, opts ...model.Option) (*schema.StreamReader[*schema.AgenticMessage], error) {
	output, err := m.Generate(ctx, input, opts...)
	if err != nil {
		return nil, err
	}
	return schema.StreamReaderFromArray([]*schema.AgenticMessage{output}), nil
}

// used returns the usage of the model.
func (m *accountedModel) used() llm.Usage {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.usage
}
