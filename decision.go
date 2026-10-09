package einorun

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/cloudwego/eino/adk"
	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/compose"
	"github.com/cloudwego/eino/schema"
)

// decidedKey marks the context of a call carried out after a decision.
type decidedKey struct{}

// decisionFrom returns the decision a call is carried out under, or nil.
func decisionFrom(ctx context.Context) *CallDecision {
	decision, _ := ctx.Value(decidedKey{}).(*CallDecision)
	return decision
}

// paused reports whether call waits for, or has, a decision the runtime
// carries out.
func paused(call *ToolCall) bool {
	return call.Status == StatusAwaitingDecision && call.Handover == HandoverNone
}

// confirmable reports whether a call of tool can pause for a decision: the
// tool runs as a plain invokable tool, so that carrying the decision out
// keeps its whole result.
func confirmable(item tool.BaseTool) bool {
	if _, enhanced := item.(tool.EnhancedInvokableTool); enhanced {
		return false
	}
	_, ok := item.(tool.InvokableTool)
	return ok
}

// carryOutDecisions carries out the decisions on the paused calls of the
// last model output, in order: approved calls run through the main agent's
// tool chain, rejected calls give the model the reason. What the model sees
// comes from the resulting records and joins the results patched into the
// restored context; a call handed over again gets its result on a later
// resume.
func (e *execution) carryOutDecisions(ctx context.Context, output *schema.AgenticMessage, records map[string]*ToolCall) (bool, error) {
	carried := false
	for _, c := range toolCalls(output) {
		record, ok := records[c.CallID]
		if !ok || !paused(record) || record.Decision == nil {
			continue
		}
		decision := *record.Decision
		var err error
		if decision.Approved {
			err = e.runDecided(ctx, c, decision)
		} else {
			err = e.reject(ctx, c, decision)
		}
		if err != nil {
			return carried, err
		}
		carried = true
		if done, ok := e.recorder.mainCall(c.CallID); ok {
			if result, ok := modelResult(&done, e.text); ok {
				e.patched[c.CallID] = result
			}
		}
	}
	return carried, nil
}

// reject records a rejected call; the model sees the reason.
func (e *execution) reject(ctx context.Context, c *schema.FunctionToolCall, decision CallDecision) error {
	result := fmt.Sprintf(e.text.CallRejected, decision.Reason)
	input := &compose.ToolInput{Name: c.Name, CallID: c.CallID, Arguments: c.Arguments}
	call, err := e.update(ctx, e.main, input, func(call *ToolCall) {
		now := time.Now()
		call.Status, call.Result, call.CompletedAt = StatusRejected, &result, &now
	})
	if err != nil {
		return err
	}
	e.observe(ctx, CallOutcome{Agent: e.main.scope, Name: c.Name, Call: call, Origin: OriginExecuted})
	return nil
}

// runDecided runs an approved call through the main agent's tool chain, as
// the agent would have, with the decision's arguments when set. A panic ends
// the run like one inside the agent; the call stays running and is settled
// as interrupted on the next resume.
func (e *execution) runDecided(ctx context.Context, c *schema.FunctionToolCall, decision CallDecision) (err error) {
	arguments := c.Arguments
	if decision.Arguments != "" {
		arguments = decision.Arguments
	}
	input := &compose.ToolInput{Name: c.Name, CallID: c.CallID, Arguments: arguments}
	item, ok := e.main.tools[c.Name]
	if !ok || !confirmable(item) {
		// Tools added by middlewares are not reachable outside the agent.
		call, err := e.fail(ctx, e.main, input, errors.New(e.text.CannotConfirm))
		if err != nil {
			return err
		}
		e.observe(ctx, CallOutcome{Agent: e.main.scope, Name: c.Name, Call: call, Origin: OriginExecuted})
		return nil
	}
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("einorun: tool %s panicked: %v", c.Name, r)
		}
	}()
	_, err = chainTool(item, e.main.chain)(context.WithValue(ctx, decidedKey{}, &decision), input)
	return err
}

// chainTool returns a function that runs an invokable tool through
// middlewares, the first outermost, and returns the result text.
func chainTool(item tool.BaseTool, middlewares []compose.ToolMiddleware) func(context.Context, *compose.ToolInput) (string, error) {
	invokable, _ := item.(tool.InvokableTool)
	endpoint := compose.InvokableToolEndpoint(func(ctx context.Context, input *compose.ToolInput) (*compose.ToolOutput, error) {
		result, err := invokable.InvokableRun(ctx, input.Arguments, input.CallOptions...)
		if err != nil {
			return nil, err
		}
		return &compose.ToolOutput{Result: result}, nil
	})
	for i := len(middlewares) - 1; i >= 0; i-- {
		if m := middlewares[i].Invokable; m != nil {
			endpoint = m(endpoint)
		}
	}
	return func(ctx context.Context, input *compose.ToolInput) (string, error) {
		output, err := endpoint(ctx, input)
		if err != nil {
			return "", err
		}
		return output.Result, nil
	}
}

// handlerToolMiddlewares turns the tool wrapping of ADK handlers into tool
// middlewares, the first handler outermost, as the agent applies them.
func handlerToolMiddlewares(handlers []adk.TypedChatModelAgentMiddleware[*schema.AgenticMessage]) []compose.ToolMiddleware {
	middlewares := make([]compose.ToolMiddleware, 0, len(handlers))
	for _, h := range handlers {
		middlewares = append(middlewares, compose.ToolMiddleware{
			Invokable: func(next compose.InvokableToolEndpoint) compose.InvokableToolEndpoint {
				return func(ctx context.Context, input *compose.ToolInput) (*compose.ToolOutput, error) {
					wrapped, err := h.WrapInvokableToolCall(ctx, func(ctx context.Context, arguments string, opts ...tool.Option) (string, error) {
						output, err := next(ctx, &compose.ToolInput{Name: input.Name, CallID: input.CallID, Arguments: arguments, CallOptions: opts})
						if err != nil {
							return "", err
						}
						return output.Result, nil
					}, &adk.ToolContext{Name: input.Name, CallID: input.CallID})
					if err != nil {
						return nil, err
					}
					result, err := wrapped(ctx, input.Arguments, input.CallOptions...)
					if err != nil {
						return nil, err
					}
					return &compose.ToolOutput{Result: result}, nil
				}
			},
		})
	}
	return middlewares
}
