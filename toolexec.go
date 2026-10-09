package einorun

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"maps"
	"strings"
	"sync"
	"time"

	"github.com/cloudwego/eino/adk"
	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/compose"
	"github.com/cloudwego/eino/schema"
)

// agent is the per-agent state of an agent of the run.
type agent struct {
	scope AgentScope
	// parentCallID is the provider call ID of the delegation call a sub-agent
	// serves; empty for the main agent.
	parentCallID string
	raw          rawResults
	budget       *budgetGuard
	injector     *mediaInjector
	offloaded    *offloadStore
	retry        *modelRetry
	summary      *summarizer
	recorder     *subagentRecorder // a sub-agent's recorder
	// modelCalls maps a sub-agent's provider call IDs to the model call that
	// made them.
	modelCalls sync.Map
}

// modelCallOf returns the model call that made a sub-agent's call.
func (a *agent) modelCallOf(providerCallID string) string {
	id, _ := a.modelCalls.Load(providerCallID)
	s, _ := id.(string)
	return s
}

// rawResults keeps the result each tool returned before context management
// changed it, by provider call ID.
type rawResults struct {
	mu      sync.Mutex
	results map[string]string
}

// put records a raw result.
func (r *rawResults) put(callID, result string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.results == nil {
		r.results = map[string]string{}
	}
	r.results[callID] = result
}

// take returns and forgets a raw result.
func (r *rawResults) take(callID string) string {
	r.mu.Lock()
	defer r.mu.Unlock()
	result := r.results[callID]
	delete(r.results, callID)
	return result
}

// rawCapture is the innermost ADK handler: it records what a tool returned
// before context management sees it.
type rawCapture struct {
	adk.TypedBaseChatModelAgentMiddleware[*schema.AgenticMessage]
	agent *agent
}

// WrapInvokableToolCall records the raw result of plain tools.
func (c *rawCapture) WrapInvokableToolCall(_ context.Context, endpoint adk.InvokableToolCallEndpoint, tCtx *adk.ToolContext) (adk.InvokableToolCallEndpoint, error) {
	return func(ctx context.Context, arguments string, opts ...toolOption) (string, error) {
		result, err := endpoint(ctx, arguments, opts...)
		if err == nil {
			c.agent.raw.put(tCtx.CallID, result)
		}
		return result, err
	}, nil
}

// WrapEnhancedInvokableToolCall records the text of multimodal results.
func (c *rawCapture) WrapEnhancedInvokableToolCall(_ context.Context, endpoint adk.EnhancedInvokableToolCallEndpoint, tCtx *adk.ToolContext) (adk.EnhancedInvokableToolCallEndpoint, error) {
	return func(ctx context.Context, arguments *schema.ToolArgument, opts ...toolOption) (*schema.ToolResult, error) {
		result, err := endpoint(ctx, arguments, opts...)
		if err == nil {
			c.agent.raw.put(tCtx.CallID, resultText(result))
		}
		return result, err
	}, nil
}

// toolOption is an option of a tool call.
type toolOption = tool.Option

// errorResult encodes an error as the result the model sees.
func errorResult(err error) string {
	encoded, _ := json.Marshal(struct {
		Error string `json:"error"`
	}{Error: err.Error()})
	return string(encoded)
}

// toolMiddleware is the runtime's tool middleware for an agent: it records
// calls, submits and hands them over, applies the completion protocol and
// reports outcomes.
func (x *execution) toolMiddleware(a *agent) compose.ToolMiddleware {
	return compose.ToolMiddleware{
		Invokable: func(next compose.InvokableToolEndpoint) compose.InvokableToolEndpoint {
			return func(ctx context.Context, input *compose.ToolInput) (*compose.ToolOutput, error) {
				var output *compose.ToolOutput
				message, err := x.runCall(ctx, a, input, func(ctx context.Context) (string, error) {
					var err error
					if output, err = next(ctx, input); err != nil {
						return "", err
					}
					return output.Result, nil
				})
				if err != nil {
					return nil, err
				}
				if message != nil {
					return &compose.ToolOutput{Result: *message}, nil
				}
				return output, nil
			}
		},
		EnhancedInvokable: func(next compose.EnhancedInvokableToolEndpoint) compose.EnhancedInvokableToolEndpoint {
			return func(ctx context.Context, input *compose.ToolInput) (*compose.EnhancedInvokableToolOutput, error) {
				var output *compose.EnhancedInvokableToolOutput
				message, err := x.runCall(ctx, a, input, func(ctx context.Context) (string, error) {
					var err error
					if output, err = next(ctx, input); err != nil {
						return "", err
					}
					return resultText(output.Result), nil
				})
				if err != nil {
					return nil, err
				}
				if message != nil {
					return &compose.EnhancedInvokableToolOutput{Result: &schema.ToolResult{Parts: []schema.ToolOutputPart{{Type: schema.ToolPartTypeText, Text: *message}}}}, nil
				}
				return output, nil
			}
		},
	}
}

// runCall executes one call through the runtime's protocol. It returns the
// text the model sees in place of the tool's output, or nil when the tool's
// own output stands. Errors abort the run.
func (x *execution) runCall(ctx context.Context, a *agent, input *compose.ToolInput, execute func(context.Context) (string, error)) (*string, error) {
	entry, _ := x.recorder.entry(input.Name)
	text := func(s string) (*string, error) { return &s, nil }
	// A batch the completion protocol rejected does not execute.
	if a.scope.Main {
		if issue := x.completion.batchIssue(ctx); issue != "" {
			call, err := x.fail(ctx, a, input, errors.New(issue))
			if err != nil {
				return nil, err
			}
			x.observe(ctx, CallOutcome{Agent: a.scope, Name: input.Name, Call: call, Origin: OriginExecuted})
			return text(errorResult(errors.New(issue)))
		}
	}
	var policy CallPolicy
	if entry != nil && entry.spec.Policy != nil {
		var err error
		policy, err = entry.spec.Policy(ctx, CallView{Name: input.Name, Arguments: input.Arguments, CallID: input.CallID, Agent: a.scope})
		if err != nil {
			call, saveErr := x.fail(ctx, a, input, err)
			if saveErr != nil {
				return nil, saveErr
			}
			x.observe(ctx, CallOutcome{Agent: a.scope, Name: input.Name, Call: call, Origin: OriginExecuted})
			return text(errorResult(err))
		}
	}
	now := time.Now()
	record, err := x.update(ctx, a, input, func(call *ToolCall) {
		call.Arguments = input.Arguments
		applyPolicy(call, policy)
		if policy.Submit != nil {
			receipt := policy.Submit.Receipt
			call.Status, call.Handover, call.Result, call.Payload = StatusAwaitingDecision, HandoverSubmitted, &receipt, policy.Submit.Payload
			call.StartedAt = &now
			return
		}
		call.Status, call.StartedAt = StatusRunning, &now
	})
	if policy.Submit != nil {
		if rejection, ok := errors.AsType[*CallRejection](err); ok || errors.Is(err, ErrCallRejected) {
			reason := err
			if ok {
				reason = rejection
			}
			message := reason.Error()
			// The host refused the submission; the call fails instead.
			failed, saveErr := x.update(ctx, a, input, func(call *ToolCall) {
				done := time.Now()
				call.Status, call.Handover, call.Payload, call.Result, call.Error, call.CompletedAt = StatusFailed, HandoverNone, nil, nil, &message, &done
			})
			if saveErr != nil {
				return nil, saveErr
			}
			x.observe(ctx, CallOutcome{Agent: a.scope, Name: input.Name, Call: failed, Origin: OriginExecuted})
			return text(errorResult(reason))
		}
		if err != nil {
			return nil, err
		}
		x.observe(ctx, CallOutcome{Agent: a.scope, Name: input.Name, Call: record, Raw: policy.Submit.Receipt, Origin: OriginSubmitted})
		return text(policy.Submit.Receipt)
	}
	if err != nil {
		return nil, err
	}
	media := &callMedia{}
	execCtx := context.WithValue(ctx, callContextKey{}, callMeta{
		CallContext: CallContext{RecordID: record.ID, ModelCallID: record.ModelCallID, ProviderCallID: input.CallID},
		suspendable: a.scope.Main,
		media:       media,
	})
	result, execErr := execute(execCtx)
	raw := a.raw.take(input.CallID)
	if execErr == nil && raw == "" {
		raw = result
	}
	// Cancellation, deadlines and framework interrupts end the run, whether
	// they come from the run's context or from the tool's own.
	_, interrupted := compose.ExtractInterruptInfo(execErr)
	if abort, ok := errors.AsType[*abortError](execErr); ok {
		// A sub-agent keeps the mark, so that the delegation ends the run too.
		if !a.scope.Main {
			return nil, abort
		}
		return nil, abort.err
	}
	if interrupted || (execErr != nil && (ctx.Err() != nil || errors.Is(execErr, context.Canceled) || errors.Is(execErr, context.DeadlineExceeded))) {
		if _, err := x.fail(ctx, a, input, execErr); err != nil {
			return nil, err
		}
		return nil, execErr
	}
	if ctrl, ok := asControl(execErr); ok {
		return x.control(ctx, a, input, entry, execCtx, ctrl)
	}
	if execErr != nil && entry != nil && entry.spec.Completion && a.scope.Main {
		message := x.completion.reject(ctx, execErr.Error())
		call, err := x.fail(ctx, a, input, execErr)
		if err != nil {
			return nil, err
		}
		x.observe(ctx, CallOutcome{Agent: a.scope, Name: input.Name, Call: call, Origin: OriginExecuted})
		return text(errorResult(errors.New(message)))
	}
	var refs []MediaRef
	if execErr == nil {
		refs = media.attached()
	}
	call, err := x.finishWith(ctx, a, input, result, execErr, refs)
	if err != nil {
		return nil, err
	}
	a.injector.add(input.CallID, refs)
	if execErr != nil {
		slog.WarnContext(ctx, "einorun: tool call failed", "run_id", x.request.RunID, "tool", input.Name, "call_id", input.CallID, "error", execErr)
		x.observe(ctx, CallOutcome{Agent: a.scope, Name: input.Name, Call: call, Origin: OriginExecuted})
		return text(errorResult(execErr))
	}
	x.observe(ctx, CallOutcome{Agent: a.scope, Name: input.Name, Call: call, Raw: raw, Origin: OriginExecuted})
	return nil, nil
}

// control applies a control result.
func (x *execution) control(ctx context.Context, a *agent, input *compose.ToolInput, entry *toolEntry, execCtx context.Context, ctrl *control) (*string, error) {
	text := func(s string) (*string, error) { return &s, nil }
	fail := func(reason string) (*string, error) {
		call, err := x.fail(ctx, a, input, errors.New(reason))
		if err != nil {
			return nil, err
		}
		x.observe(ctx, CallOutcome{Agent: a.scope, Name: input.Name, Call: call, Origin: OriginExecuted})
		return text(errorResult(errors.New(reason)))
	}
	switch {
	case ctrl.completion != nil:
		if entry == nil || !entry.spec.Completion || !a.scope.Main {
			return fail(x.text.CannotComplete)
		}
		value := string(ctrl.completion.Value)
		now := time.Now()
		call, err := x.update(ctx, a, input, func(call *ToolCall) {
			call.Status, call.Result, call.CompletedAt = StatusSucceeded, &value, &now
			completion := *ctrl.completion
			call.Completion = &completion
		})
		if err != nil {
			return nil, err
		}
		if err := x.completion.register(ctx, Completion{Source: FromTool, CallID: call.ID, Value: ctrl.completion.Value, Fixed: ctrl.completion.Fixed}); err != nil {
			return nil, err
		}
		if err := adk.SetToolReturnDirectly(ctx); err != nil {
			slog.WarnContext(ctx, "einorun: completion could not return directly", "run_id", x.request.RunID, "error", err)
		}
		x.observe(ctx, CallOutcome{Agent: a.scope, Name: input.Name, Call: call, Raw: value, Origin: OriginExecuted})
		return text(value)
	case ctrl.handover == HandoverAwait:
		if !CanSuspend(execCtx) {
			return fail(x.text.CannotAwait)
		}
		call, err := x.update(ctx, a, input, func(call *ToolCall) {
			call.Handover, call.Status, call.Payload = HandoverAwait, StatusWaiting, ctrl.payload
		})
		if err != nil {
			return nil, err
		}
		x.observe(ctx, CallOutcome{Agent: a.scope, Name: input.Name, Call: call, Origin: OriginExecuted})
		return text(x.text.AwaitingResult)
	default:
		if !a.scope.Main {
			return fail(x.text.CannotDetach)
		}
		receipt := ctrl.receipt
		call, err := x.update(ctx, a, input, func(call *ToolCall) {
			call.Handover, call.Result, call.Payload = HandoverDetached, &receipt, ctrl.payload
		})
		if err != nil {
			return nil, err
		}
		x.observe(ctx, CallOutcome{Agent: a.scope, Name: input.Name, Call: call, Raw: receipt, Origin: OriginExecuted})
		return text(receipt)
	}
}

// fail records a failed call with the error the model sees.
func (x *execution) fail(ctx context.Context, a *agent, input *compose.ToolInput, callErr error) (ToolCall, error) {
	return x.finishWith(ctx, a, input, "", callErr, nil)
}

// finishWith records the outcome of a call with the media attached to its
// result.
func (x *execution) finishWith(ctx context.Context, a *agent, input *compose.ToolInput, result string, callErr error, media []MediaRef) (ToolCall, error) {
	now := time.Now()
	return x.update(ctx, a, input, func(call *ToolCall) {
		call.Media = media
		if call.StartedAt == nil {
			call.StartedAt = &now
		}
		call.CompletedAt = &now
		if callErr != nil {
			message := strings.ReplaceAll(callErr.Error(), "\x00", "")
			call.Status, call.Error, call.Result = StatusFailed, &message, nil
			return
		}
		result = strings.ReplaceAll(result, "\x00", "")
		call.Status, call.Result = StatusSucceeded, &result
	})
}

// update changes the record of a call of agent a and writes it. A
// sub-agent's failures are marked to end the run through the delegation.
func (x *execution) update(ctx context.Context, a *agent, input *compose.ToolInput, change func(*ToolCall)) (ToolCall, error) {
	if a.scope.Main {
		return x.recorder.updateCall(ctx, input.CallID, change)
	}
	parent, ok := x.recorder.mainCall(a.parentCallID)
	if !ok {
		return ToolCall{}, &abortError{err: errors.New("einorun: sub-agent call without its delegation call")}
	}
	if _, exists := x.recorder.childPosition(parent.ID, input.CallID); !exists {
		x.recorder.childStarted(a.parentCallID, a.modelCallOf(input.CallID), input.Name, input.CallID, input.Arguments)
	}
	call, err := x.recorder.updateChild(ctx, parent.ID, input.CallID, change)
	if err != nil {
		return call, &abortError{err: err}
	}
	return call, nil
}

// applyPolicy applies a policy's trait overrides and notes.
func applyPolicy(call *ToolCall, policy CallPolicy) {
	if policy.Replayable != nil {
		call.Replayable = *policy.Replayable
	}
	if policy.SideEffects != nil {
		call.SideEffects = *policy.SideEffects
	}
	if len(policy.Notes) > 0 {
		if call.Notes == nil {
			call.Notes = map[string]string{}
		}
		maps.Copy(call.Notes, policy.Notes)
	}
}

// observe reports an outcome to tool observers.
func (x *execution) observe(ctx context.Context, outcome CallOutcome) {
	for _, observer := range x.extensions.tools {
		observer.AfterTool(ctx, outcome)
	}
}

// resultText turns a multimodal result into text; media are named by type.
func resultText(result *schema.ToolResult) string {
	if result == nil {
		return ""
	}
	parts := make([]string, 0, len(result.Parts))
	for _, part := range result.Parts {
		switch {
		case part.Type == schema.ToolPartTypeText:
			parts = append(parts, part.Text)
		case part.Image != nil:
			parts = append(parts, "[image "+part.Image.MIMEType+"]")
		default:
			parts = append(parts, "["+string(part.Type)+"]")
		}
	}
	return strings.Join(parts, "\n")
}
