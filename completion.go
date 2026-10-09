package einorun

import (
	"context"
	"fmt"
	"log/slog"
	"slices"
	"strings"
	"sync"

	"github.com/cloudwego/eino/adk"
	"github.com/cloudwego/eino/schema"

	"github.com/runforyou-ai/einorun/internal/prompt"
)

// completionState keeps the run's completion protocol: the active
// completion, the corrections used and the batch check of the current model
// output.
type completionState struct {
	adk.TypedBaseChatModelAgentMiddleware[*schema.AgenticMessage]
	tools  []string // completion tool names, sorted
	policy *CompletionPolicy
	limit  int
	text   *prompt.Runtime
	save   func(ctx context.Context) error // saves a step with the active completion
	spent  func() bool                     // the current model call is at the end of the iteration budget

	mu         sync.Mutex
	used       int
	active     *Completion
	issue      string // why the current batch is rejected; empty when it is valid
	issueTaken bool   // the batch's correction was counted
}

// isTool reports whether name is a completion tool.
func (c *completionState) isTool(name string) bool {
	_, found := slices.BinarySearch(c.tools, name)
	return found
}

// AfterModelRewriteState checks the batch of the model output before any tool
// runs: at most one completion tool, and never together with other tools.
func (c *completionState) AfterModelRewriteState(ctx context.Context, state *adk.TypedChatModelAgentState[*schema.AgenticMessage], _ *adk.TypedModelContext[*schema.AgenticMessage]) (context.Context, *adk.TypedChatModelAgentState[*schema.AgenticMessage], error) {
	calls := toolCalls(state.Messages[len(state.Messages)-1])
	completions := 0
	for _, call := range calls {
		if c.isTool(call.Name) {
			completions++
		}
	}
	names := strings.Join(c.tools, c.text.ListSeparator)
	issue := ""
	switch {
	case completions > 1:
		issue = fmt.Sprintf(c.text.BatchTooMany, names)
	case completions == 1 && len(calls) > 1:
		issue = fmt.Sprintf(c.text.BatchMixed, names)
	}
	c.mu.Lock()
	c.issue, c.issueTaken = issue, false
	c.mu.Unlock()
	return ctx, state, nil
}

// batchIssue returns why the current batch is rejected, counting the batch
// once against the corrections; empty when the batch is valid.
func (c *completionState) batchIssue(ctx context.Context) string {
	c.mu.Lock()
	issue, taken := c.issue, c.issueTaken
	c.issueTaken = c.issueTaken || issue != ""
	c.mu.Unlock()
	if issue == "" || taken {
		return issue
	}
	return c.reject(ctx, issue)
}

// reject handles an invalid completion output: it uses a correction when one
// is left and the budget is not spent, otherwise it ends the run with the
// fallback completion when the host configured one. It returns the message
// for the model.
func (c *completionState) reject(ctx context.Context, issue string) string {
	spent := c.spent != nil && c.spent()
	if !spent && c.take() {
		slog.WarnContext(ctx, "einorun: invalid completion output, asking the model to correct it", "issue", issue)
		return issue
	}
	reason := ReasonCorrectionsExhausted
	if spent {
		reason = ReasonBudgetExhausted
	}
	if c.fallback(ctx, reason) {
		if err := adk.SetToolReturnDirectly(ctx); err != nil {
			slog.WarnContext(ctx, "einorun: fallback could not return directly", "error", err)
		}
	}
	return issue
}

// take uses a correction and reports whether one was left.
func (c *completionState) take() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.used >= c.limit {
		return false
	}
	c.used++
	return true
}

// left returns the corrections left.
func (c *completionState) left() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return max(c.limit-c.used, 0)
}

// fallback makes the fallback completion active and reports whether the host
// configured one.
func (c *completionState) fallback(ctx context.Context, reason string) bool {
	if c.policy == nil || c.policy.Fallback == nil {
		return false
	}
	c.mu.Lock()
	if c.active == nil || !c.active.Fixed {
		c.active = &Completion{Source: FromFallback, Value: c.policy.Fallback(reason), Fixed: true, Reason: reason}
	}
	c.mu.Unlock()
	slog.WarnContext(ctx, "einorun: output cannot be corrected, ending with the fallback completion", "reason", reason)
	return true
}

// register makes completion active and saves it.
func (c *completionState) register(ctx context.Context, completion Completion) error {
	c.mu.Lock()
	if c.active == nil || !c.active.Fixed {
		c.active = &completion
	}
	c.mu.Unlock()
	if c.save == nil {
		return nil
	}
	return c.save(ctx)
}

// current returns a copy of the active completion.
func (c *completionState) current() *Completion {
	c.mu.Lock()
	defer c.mu.Unlock()
	return cloneCompletion(c.active)
}

// fixed reports whether a fixed completion is active.
func (c *completionState) fixed() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.active != nil && c.active.Fixed
}

// supersede drops a non-fixed active completion, when new input is claimed.
func (c *completionState) supersede() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.active != nil && !c.active.Fixed {
		c.active = nil
	}
}

// restore restores the corrections used and the active completion.
func (c *completionState) restore(used int, active *Completion) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.used, c.active = used, cloneCompletion(active)
}

// snapshot returns the corrections used and the active completion.
func (c *completionState) snapshot() (int, *Completion) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.used, cloneCompletion(c.active)
}

// budgetGuard counts model calls per turn and, when the budget is spent,
// keeps only completion tools and adds a closing note.
type budgetGuard struct {
	adk.TypedBaseChatModelAgentMiddleware[*schema.AgenticMessage]
	max        int
	keep       func(name string) bool // tools kept at the end of the budget
	notice     string
	mu         sync.Mutex
	iterations int
	carry      bool // the next run of the agent keeps the remaining budget
	exhausted  bool // the latest model call came at the end of the budget
}

// BeforeAgent resets the count at the start of a turn; a correction rerun
// keeps it.
func (g *budgetGuard) BeforeAgent(ctx context.Context, runCtx *adk.ChatModelAgentContext[*schema.AgenticMessage]) (context.Context, *adk.ChatModelAgentContext[*schema.AgenticMessage], error) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.carry {
		g.carry = false
		return ctx, runCtx, nil
	}
	g.iterations, g.exhausted = 0, false
	return ctx, runCtx, nil
}

// BeforeModelRewriteState counts the model call and narrows the tools when
// the budget is spent.
func (g *budgetGuard) BeforeModelRewriteState(ctx context.Context, state *adk.TypedChatModelAgentState[*schema.AgenticMessage], _ *adk.TypedModelContext[*schema.AgenticMessage]) (context.Context, *adk.TypedChatModelAgentState[*schema.AgenticMessage], error) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.iterations++
	if g.iterations < g.max {
		return ctx, state, nil
	}
	g.exhausted = true
	state.DeferredToolInfos = nil
	// A nil list would make the framework fall back to all tools.
	kept := make([]*schema.ToolInfo, 0)
	for _, info := range state.ToolInfos {
		if g.keep != nil && g.keep(info.Name) {
			kept = append(kept, info)
		}
	}
	state.ToolInfos = kept
	state.Messages = append(state.Messages, schema.UserAgenticMessage(g.notice))
	return ctx, state, nil
}

// reset starts a new turn's budget, discarding a carry from recovery.
func (g *budgetGuard) reset() {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.iterations, g.carry, g.exhausted = 0, false, false
}

// carryBudget makes the next run of the agent keep the remaining budget.
func (g *budgetGuard) carryBudget() {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.carry = true
}

// spent reports whether the latest model call came at the end of the budget.
func (g *budgetGuard) spent() bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.exhausted
}

// count returns the model calls of the current turn.
func (g *budgetGuard) count() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.iterations
}

// resume restores the count; the next run keeps the remaining budget.
func (g *budgetGuard) resume(iterations int) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.iterations, g.carry = iterations, true
}

// cloneCompletion copies c.
func cloneCompletion(c *Completion) *Completion {
	if c == nil {
		return nil
	}
	clone := *c
	clone.Value = slices.Clone(c.Value)
	return &clone
}
