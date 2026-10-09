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

// carryOutDecisions carries out the decisions on the paused calls of the
// last model output, in order: approved calls run through the main agent's
// tool chain, rejected calls give the model the reason. The results join the
// results patched into the restored context.
func (e *execution) carryOutDecisions(ctx context.Context, output *schema.AgenticMessage, records map[string]*ToolCall) error {
	for _, c := range toolCalls(output) {
		record, ok := records[c.CallID]
		if !ok || !paused(record) || record.Decision == nil {
			continue
		}
		decision := *record.Decision
		var result string
		var err error
		if decision.Approved {
			result, err = e.runDecided(ctx, c, decision)
		} else {
			result, err = e.reject(ctx, c, decision)
		}
		if err != nil {
			return err
		}
		e.patched[c.CallID] = result
	}
	return nil
}

// reject records a rejected call and returns what the model sees.
func (e *execution) reject(ctx context.Context, c *schema.FunctionToolCall, decision CallDecision) (string, error) {
	result := fmt.Sprintf(e.text.CallRejected, decision.Reason)
	input := &compose.ToolInput{Name: c.Name, CallID: c.CallID, Arguments: c.Arguments}
	call, err := e.update(ctx, e.main, input, func(call *ToolCall) {
		now := time.Now()
		call.Status, call.Result, call.CompletedAt = StatusRejected, &result, &now
	})
	if err != nil {
		return "", err
	}
	e.observe(ctx, CallOutcome{Agent: e.main.scope, Name: c.Name, Call: call, Origin: OriginExecuted})
	return result, nil
}

// runDecided runs an approved call through the main agent's tool chain, as
// the agent would have, with the decision's arguments when set, and returns
// what the model sees.
func (e *execution) runDecided(ctx context.Context, c *schema.FunctionToolCall, decision CallDecision) (string, error) {
	arguments := c.Arguments
	if decision.Arguments != "" {
		arguments = decision.Arguments
	}
	input := &compose.ToolInput{Name: c.Name, CallID: c.CallID, Arguments: arguments}
	item, ok := e.main.tools[c.Name]
	if !ok {
		// Tools added by middlewares are not reachable outside the agent.
		call, err := e.fail(ctx, e.main, input, errors.New(e.text.CannotConfirm))
		if err != nil {
			return "", err
		}
		e.observe(ctx, CallOutcome{Agent: e.main.scope, Name: c.Name, Call: call, Origin: OriginExecuted})
		return errorResult(errors.New(e.text.CannotConfirm)), nil
	}
	return chainTool(item, e.main.chain)(context.WithValue(ctx, decidedKey{}, &decision), input)
}

// chainTool returns a function that runs a tool through middlewares, the
// first outermost, and returns the result text.
func chainTool(item tool.BaseTool, middlewares []compose.ToolMiddleware) func(context.Context, *compose.ToolInput) (string, error) {
	if enhanced, ok := item.(tool.EnhancedInvokableTool); ok {
		endpoint := compose.EnhancedInvokableToolEndpoint(func(ctx context.Context, input *compose.ToolInput) (*compose.EnhancedInvokableToolOutput, error) {
			result, err := enhanced.InvokableRun(ctx, &schema.ToolArgument{Text: input.Arguments}, input.CallOptions...)
			if err != nil {
				return nil, err
			}
			return &compose.EnhancedInvokableToolOutput{Result: result}, nil
		})
		for i := len(middlewares) - 1; i >= 0; i-- {
			if m := middlewares[i].EnhancedInvokable; m != nil {
				endpoint = m(endpoint)
			}
		}
		return func(ctx context.Context, input *compose.ToolInput) (string, error) {
			output, err := endpoint(ctx, input)
			if err != nil {
				return "", err
			}
			return resultText(output.Result), nil
		}
	}
	invokable, ok := item.(tool.InvokableTool)
	if !ok {
		return func(context.Context, *compose.ToolInput) (string, error) {
			return "", errors.New("einorun: only invokable tools can be carried out after a decision")
		}
	}
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
			EnhancedInvokable: func(next compose.EnhancedInvokableToolEndpoint) compose.EnhancedInvokableToolEndpoint {
				return func(ctx context.Context, input *compose.ToolInput) (*compose.EnhancedInvokableToolOutput, error) {
					wrapped, err := h.WrapEnhancedInvokableToolCall(ctx, func(ctx context.Context, argument *schema.ToolArgument, opts ...tool.Option) (*schema.ToolResult, error) {
						output, err := next(ctx, &compose.ToolInput{Name: input.Name, CallID: input.CallID, Arguments: argument.Text, CallOptions: opts})
						if err != nil {
							return nil, err
						}
						return output.Result, nil
					}, &adk.ToolContext{Name: input.Name, CallID: input.CallID})
					if err != nil {
						return nil, err
					}
					result, err := wrapped(ctx, &schema.ToolArgument{Text: input.Arguments}, input.CallOptions...)
					if err != nil {
						return nil, err
					}
					return &compose.EnhancedInvokableToolOutput{Result: result}, nil
				}
			},
		})
	}
	return middlewares
}
