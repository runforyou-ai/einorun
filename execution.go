package einorun

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"reflect"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/cloudwego/eino/adk"
	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/compose"
	"github.com/cloudwego/eino/schema"

	"github.com/runforyou-ai/einorun/internal/checkpoint"
	"github.com/runforyou-ai/einorun/internal/prompt"
	"github.com/runforyou-ai/einorun/llm"
)

// defaultMaxIterations is the iteration budget per turn.
const defaultMaxIterations = 20

// checkpointVersion is the version of the checkpoint format; only the
// current version is accepted.
const checkpointVersion = 1

// ErrGuardRejected reports a run whose guard rejected the output when no
// correction was left and no fallback completion is configured.
var ErrGuardRejected = errors.New("einorun: the guard rejected the output")

// state is the checkpoint of a run.
type state struct {
	Version      int                        `json:"version"`
	Messages     []checkpoint.Message       `json:"messages"`
	Seen         []string                   `json:"seen,omitempty"`
	MediaCount   int                        `json:"mediaCount"`
	MediaBytes   int64                      `json:"mediaBytes"`
	MediaEnabled bool                       `json:"mediaEnabled"`
	ClaimedSeq   int64                      `json:"claimedSeq"`
	Turns        int                        `json:"turns"`
	Iterations   int                        `json:"iterations"`
	Corrections  int                        `json:"corrections"`
	Completion   *Completion                `json:"completion,omitempty"`
	Usage        llm.Usage                  `json:"usage"`
	Extensions   map[string]json.RawMessage `json:"extensions,omitempty"`
}

// extensionSet is the run's extension instances by the interfaces they
// implement.
type extensionSet struct {
	all      []Extension
	guard    Guard
	tools    []ToolObserver
	claims   []ClaimObserver
	restores []RestoreObserver
	outputs  []OutputObserver
	stateful map[string]Stateful
	closers  []Closer
}

// execution is one execution attempt of a run.
type execution struct {
	request    Request
	text       *prompt.Runtime
	language   llm.Language
	recorder   *recorder
	completion *completionState
	budget     *budgetGuard
	retry      *modelRetry
	capture    *inputCapture
	inputs     *inputs
	history    history
	extensions extensionSet
	memory     *MemoryJournal
	agent      adk.TypedAgent[*schema.AgenticMessage]
	main       *agent
	releases   []func()
	media      mediaPolicy
	window     int

	saveMu     sync.Mutex
	context    []*schema.AgenticMessage // model context at the latest finalized output, without the system instruction
	turns      int
	lastClaim  Claim
	turnCalls  []CallOutcome
	callsMu    sync.Mutex
	rejected   []*schema.AgenticMessage // rejected text that only enters the correction rerun
	correction string                   // the guard's correction prompt for the rerun
	restored   *state
	patched    map[string]string // results patched into the context on recovery, by provider call ID
	result     Result
	finished   bool
}

// toolObserverFunc adapts a function to ToolObserver.
type toolObserverFunc func(context.Context, CallOutcome)

// AfterTool calls f.
func (f toolObserverFunc) AfterTool(ctx context.Context, outcome CallOutcome) { f(ctx, outcome) }

// assemble prepares an execution: extensions, recorder, tools, completion
// protocol, agent and the restored state.
func (r *Runtime) assemble(ctx context.Context, request Request) (*execution, error) {
	text := prompt.RuntimeFor(string(r.language))
	e := &execution{request: request, text: text, language: r.language, patched: map[string]string{}}
	if request.Journal == nil {
		e.memory = NewMemoryJournal()
		request.Journal = e.memory
		e.request.Journal = e.memory
	}
	streamID := request.StreamID
	if streamID == "" {
		streamID = llm.NewModelCallID()
	}
	if err := e.loadExtensions(ctx); err != nil {
		return e, err
	}
	e.recorder = newRecorder(e.request, streamID)
	e.recorder.pub.start()
	e.window = llm.ContextWindow(request.Model.ContextWindow)
	e.main = &agent{scope: AgentScope{Name: "main", Main: true}}

	tools, completionTools, err := e.registerTools(ctx, e.main.scope)
	if err != nil {
		return e, err
	}
	limit := request.Limits.Corrections
	if limit <= 0 {
		limit = 1
	}
	e.completion = &completionState{tools: completionTools, policy: request.Completion, limit: limit, text: text}
	e.completion.save = func(ctx context.Context) error { return e.save(ctx) }
	maxIterations := request.Limits.MaxIterations
	if maxIterations <= 0 {
		maxIterations = defaultMaxIterations
	}
	notice := text.FinalNotice
	if len(completionTools) > 0 {
		notice = fmt.Sprintf(text.FinalNoticeCompletion, strings.Join(completionTools, text.ListSeparator))
	}
	if request.Completion != nil && request.Completion.FinalNotice != "" {
		notice = request.Completion.FinalNotice
	}
	e.budget = &budgetGuard{max: maxIterations, keep: e.completion.isTool, notice: notice}
	e.completion.spent = e.budget.spent

	enabled := &atomic.Bool{}
	enabled.Store(true)
	e.media = mediaPolicy{read: request.ReadMedia, modalities: map[llm.Modality]bool{}, enabled: enabled}
	for _, m := range request.Model.Inputs {
		e.media.modalities[m] = true
	}
	e.media.maxCount = mediaMaxCount(e.window)
	e.retry = &modelRetry{runID: request.RunID, enabled: enabled, text: text}
	e.capture = &inputCapture{}

	if request.Resume != nil {
		restored, err := decodeState(request.Resume.State)
		if err != nil {
			return e, err
		}
		e.restored = &restored
		// Extension state comes first, before anything reads it.
		if err := e.restoreExtensions(restored); err != nil {
			return e, err
		}
		if e.memory != nil {
			e.memory.Load(*request.Resume)
		}
	}
	e.recorder.onStep = func(ctx context.Context, messages []*schema.AgenticMessage) error {
		e.saveMu.Lock()
		e.context = withoutSystem(messages)
		e.saveMu.Unlock()
		return e.save(ctx)
	}
	e.agent, err = e.buildAgent(ctx, e.main, request.Instruction, tools, maxIterations)
	if err != nil {
		return e, err
	}
	e.inputs = &inputs{feed: request.Feed, hold: e.completion.fixed}
	return e, nil
}

// loadExtensions instantiates the extensions and sorts them by interface.
func (e *execution) loadExtensions(ctx context.Context) error {
	set := extensionSet{stateful: map[string]Stateful{}}
	names := map[string]bool{}
	run := RunScope{RunID: e.request.RunID, Language: e.language}
	for _, prototype := range e.request.Extensions {
		instance := prototype
		if factory, ok := prototype.(Instantiable); ok {
			created, err := factory.Instance(ctx, run)
			if err != nil {
				return fmt.Errorf("einorun: extension %s: %w", prototype.Name(), err)
			}
			instance = created
		}
		name := instance.Name()
		if names[name] {
			return fmt.Errorf("einorun: extension %s is registered twice", name)
		}
		names[name] = true
		set.all = append(set.all, instance)
		if same(prototype, e.request.Guard) || same(instance, e.request.Guard) {
			if guard, ok := instance.(Guard); ok {
				set.guard = guard
			}
		}
		if o, ok := instance.(ToolObserver); ok {
			set.tools = append(set.tools, o)
		}
		if o, ok := instance.(ClaimObserver); ok {
			set.claims = append(set.claims, o)
		}
		if o, ok := instance.(RestoreObserver); ok {
			set.restores = append(set.restores, o)
		}
		if o, ok := instance.(OutputObserver); ok {
			set.outputs = append(set.outputs, o)
		}
		if s, ok := instance.(Stateful); ok {
			set.stateful[name] = s
		}
		if c, ok := instance.(Closer); ok {
			set.closers = append(set.closers, c)
		}
	}
	if set.guard == nil {
		set.guard = e.request.Guard
	}
	// The runtime collects this turn's outcomes of the main agent for guards.
	set.tools = append(set.tools, toolObserverFunc(func(_ context.Context, outcome CallOutcome) {
		if outcome.Agent.Main {
			e.callsMu.Lock()
			e.turnCalls = append(e.turnCalls, outcome)
			e.callsMu.Unlock()
		}
	}))
	e.extensions = set
	return nil
}

// same reports whether two values are the same comparable value.
func same(a, b any) bool {
	if a == nil || b == nil {
		return false
	}
	ta, tb := reflect.TypeOf(a), reflect.TypeOf(b)
	return ta == tb && ta.Comparable() && a == b
}

// registerTools builds the tools of an agent and registers their entries; it
// returns the tools and the sorted names of completion tools.
func (e *execution) registerTools(ctx context.Context, scope AgentScope) ([]tool.BaseTool, []string, error) {
	var tools []tool.BaseTool
	var completions []string
	for _, spec := range e.request.Tools {
		if spec.MainOnly && !scope.Main {
			continue
		}
		item := spec.Tool
		if spec.New != nil {
			created, release, err := spec.New(ctx, scope)
			if err != nil {
				return nil, nil, err
			}
			if release != nil {
				e.releases = append(e.releases, release)
			}
			item = created
		}
		info, err := item.Info(ctx)
		if err != nil {
			return nil, nil, fmt.Errorf("einorun: read tool info: %w", err)
		}
		if scope.Main {
			if _, exists := e.recorder.tools[info.Name]; exists {
				return nil, nil, fmt.Errorf("einorun: tool %s is registered twice", info.Name)
			}
			e.recorder.tools[info.Name] = &toolEntry{spec: spec, name: info.Name}
			if spec.Completion {
				completions = append(completions, info.Name)
			}
		}
		tools = append(tools, item)
	}
	slices.Sort(completions)
	return tools, completions, nil
}

// buildAgent creates an agent with the runtime's middlewares in their fixed
// order and the extensions' after them.
func (e *execution) buildAgent(ctx context.Context, a *agent, instruction string, tools []tool.BaseTool, maxIterations int) (*adk.TypedChatModelAgent[*schema.AgenticMessage], error) {
	chatModel, err := e.request.Model.New(ctx, llm.ModelOptions{MaxOutputTokens: e.request.Model.MaxOutputTokens})
	if err != nil {
		return nil, err
	}
	patch, err := newPatchHandler(ctx, e.text, func(callID string) (string, bool) {
		result, ok := e.patched[callID]
		return result, ok
	})
	if err != nil {
		return nil, err
	}
	handlers := []adk.TypedChatModelAgentMiddleware[*schema.AgenticMessage]{
		e.recorder, &argumentsNormalizer{}, patch, e.budget,
	}
	if len(e.completion.tools) > 0 {
		handlers = append(handlers, e.completion)
	}
	handlers = append(handlers, &observerMiddleware{scope: a.scope, observers: e.extensions.outputs})
	var outer, inner []compose.ToolMiddleware
	for _, ext := range e.extensions.all {
		if m, ok := ext.(ModelMiddleware); ok {
			handlers = append(handlers, m.ModelMiddlewares(a.scope)...)
		}
		if m, ok := ext.(ToolMiddleware); ok {
			for _, staged := range m.ToolMiddlewares(a.scope) {
				if staged.Stage == StageOuter {
					outer = append(outer, staged.Middleware)
				} else {
					inner = append(inner, staged.Middleware)
				}
			}
		}
		if p, ok := ext.(InstructionProvider); ok {
			if extra := strings.TrimSpace(p.Instruction(a.scope)); extra != "" {
				instruction = strings.TrimSpace(instruction + "\n\n" + extra)
			}
		}
	}
	// The input capture wraps the model inside every extension, so guards see
	// what the model received; the raw result capture is the innermost
	// handler around tools.
	if a.scope.Main {
		handlers = append(handlers, e.capture)
	}
	handlers = append(handlers, &rawCapture{agent: a})
	middlewares := slices.Concat(outer, []compose.ToolMiddleware{e.toolMiddleware(a)}, inner)
	built, err := adk.NewTypedChatModelAgent(ctx, &adk.TypedChatModelAgentConfig[*schema.AgenticMessage]{
		Name: a.scope.Name, Instruction: instruction, Model: chatModel,
		ToolsConfig: adk.ToolsConfig{ToolsNodeConfig: compose.ToolsNodeConfig{
			Tools: tools, ToolCallMiddlewares: middlewares,
		}},
		Handlers:         handlers,
		MaxIterations:    maxIterations,
		ModelRetryConfig: e.retry.config(),
	})
	if err != nil {
		return nil, fmt.Errorf("einorun: create agent %s: %w", a.scope.Name, err)
	}
	return built, nil
}

// close releases tools, stops the stream and closes extensions.
func (e *execution) close() {
	for _, release := range e.releases {
		release()
	}
	if e.recorder != nil {
		e.recorder.pub.close()
	}
	for _, c := range e.extensions.closers {
		_ = c.Close()
	}
}

// run restores or starts the turn loop and returns the run's outcome.
func (e *execution) run(ctx context.Context) (Result, error) {
	partial := func() Result {
		return Result{Usage: e.totalUsage(), Blocks: e.recorder.partialBlocks(), Calls: e.recorder.childCalls(), Plan: e.recorder.currentPlan()}
	}
	if e.restored != nil {
		suspended, err := e.restore(ctx)
		if err != nil {
			return partial(), err
		}
		if suspended {
			return Result{Usage: e.totalUsage(), Blocks: e.recorder.blocks(), Calls: e.recorder.childCalls(), Plan: e.recorder.currentPlan(), Suspended: true}, nil
		}
	}
	e.inputs.loop = adk.NewTurnLoop(adk.TurnLoopConfig[trigger, *schema.AgenticMessage]{
		GenInput: e.genInput,
		PrepareAgent: func(context.Context, *adk.TurnLoop[trigger, *schema.AgenticMessage], []trigger) (adk.TypedAgent[*schema.AgenticMessage], error) {
			return e.agent, nil
		},
		OnAgentEvents: e.onAgentEvents,
	})
	if e.restored != nil {
		if err := e.finishRestored(ctx); err != nil {
			return partial(), err
		}
	}
	var err error
	if !e.finished {
		err = e.inputs.run(ctx)
	}
	if errors.Is(err, errSuspended) {
		return Result{Usage: e.totalUsage(), Blocks: e.recorder.blocks(), Calls: e.recorder.childCalls(), Plan: e.recorder.currentPlan(), Suspended: true}, nil
	}
	if err != nil {
		return partial(), err
	}
	if !e.finished {
		return partial(), errors.New("einorun: run stopped without a result")
	}
	e.result.EndSeq = e.inputs.boundary()
	e.result.Usage = e.totalUsage()
	e.result.Blocks = e.recorder.blocks()
	e.result.Calls = e.recorder.childCalls()
	e.result.Plan = e.recorder.currentPlan()
	return e.result, nil
}

// genInput starts a turn: it claims new input and appends it to the
// history, reruns the turn with a correction, or continues a restored turn.
func (e *execution) genInput(ctx context.Context, _ *adk.TurnLoop[trigger, *schema.AgenticMessage], items []trigger) (*adk.GenInputResult[trigger, *schema.AgenticMessage], error) {
	resume := slices.ContainsFunc(items, func(t trigger) bool { return t.resume })
	fresh := slices.ContainsFunc(items, func(t trigger) bool { return !t.resume })
	opts := []adk.AgentRunOption{adk.WithAfterToolCallsHook(e.afterToolCalls)}
	// A restored turn continues alone, unless new input arrived with it and
	// no fixed completion holds it.
	if resume && (!fresh || e.completion.fixed()) {
		return &adk.GenInputResult[trigger, *schema.AgenticMessage]{
			Input:     &adk.TypedAgentInput[*schema.AgenticMessage]{Messages: slices.Clone(e.history.messages), EnableStreaming: true},
			RunOpts:   opts,
			Consumed:  slices.DeleteFunc(slices.Clone(items), func(t trigger) bool { return !t.resume }),
			Remaining: slices.DeleteFunc(slices.Clone(items), func(t trigger) bool { return t.resume }),
		}, nil
	}
	if resume {
		e.dropCompletionTail()
	}
	e.recorder.discardPending()
	var through int64
	for _, t := range items {
		if !t.correction && !t.resume {
			through = max(through, t.seq)
		}
	}
	var messages []*schema.AgenticMessage
	switch {
	case through > 0:
		e.completion.supersede()
		e.budget.reset()
		e.turns++
		if e.request.Limits.MaxTurns > 0 && e.turns > e.request.Limits.MaxTurns {
			return nil, fmt.Errorf("einorun: turn limit %d exceeded", e.request.Limits.MaxTurns)
		}
		e.rejected, e.correction = nil, ""
		e.callsMu.Lock()
		e.turnCalls = nil
		e.callsMu.Unlock()
		claim, err := e.inputs.claim(ctx, through)
		if err != nil {
			return nil, err
		}
		e.lastClaim = claim
		for _, o := range e.extensions.claims {
			o.OnClaim(ctx, claim)
		}
		if !e.media.enabled.Load() {
			e.history.messages = withoutMedia(e.history.messages, e.text)
		}
		messages = e.history.appendInput(ctx, trimHistory(ctx, claim.Messages, e.window), e.media)
	case e.correction != "" || len(e.rejected) > 0:
		messages = slices.Concat(e.history.messages, e.rejected, []*schema.AgenticMessage{schema.UserAgenticMessage(e.correction)})
		e.rejected, e.correction = nil, ""
		e.budget.carryBudget()
	default:
		return nil, errors.New("einorun: turn loop started without input")
	}
	return &adk.GenInputResult[trigger, *schema.AgenticMessage]{
		Input:    &adk.TypedAgentInput[*schema.AgenticMessage]{Messages: messages, EnableStreaming: true},
		RunOpts:  opts,
		Consumed: items,
	}, nil
}

// onAgentEvents collects the turn's messages and decides how the turn ends:
// with a completion, with reviewed text, or superseded by new input.
func (e *execution) onAgentEvents(ctx context.Context, turn *adk.TurnContext[trigger, *schema.AgenticMessage], events *adk.AsyncIterator[*adk.TypedAgentEvent[*schema.AgenticMessage]]) error {
	candidate := ""
	var intermediates []*schema.AgenticMessage
	for {
		event, ok := events.Next()
		if !ok {
			break
		}
		if event.Err != nil {
			if _, canceled := errors.AsType[*adk.CancelError](event.Err); canceled {
				continue
			}
			return event.Err
		}
		if event.Output == nil || event.Output.MessageOutput == nil {
			continue
		}
		message, err := event.Output.MessageOutput.GetMessage()
		if _, retried := errors.AsType[*adk.WillRetryError](err); retried {
			continue
		}
		if err != nil {
			return err
		}
		if message == nil || message.Role == schema.AgenticRoleTypeSystem {
			continue
		}
		intermediates = append(intermediates, message)
		if message.Role == schema.AgenticRoleTypeAssistant && !hasToolCalls(message) {
			if text := strings.TrimSpace(llm.Text(message)); text != "" {
				candidate = text
			}
		}
	}
	e.history.appendOutput(intermediates)
	e.saveMu.Lock()
	e.history.fillResults(e.context)
	e.saveMu.Unlock()
	completion := e.completion.current()
	text := candidate
	if completion != nil {
		text = ""
	}
	if guard := e.extensions.guard; guard != nil && completion == nil && text != "" {
		pending, err := e.inputs.pending(ctx, turn)
		if err != nil {
			return err
		}
		e.callsMu.Lock()
		calls := slices.Clone(e.turnCalls)
		e.callsMu.Unlock()
		verdict, err := guard.Review(ctx, TurnView{
			Claim: e.lastClaim, ModelInput: e.capture.latest(), Calls: calls, Text: text,
			Superseded: pending, BudgetSpent: e.budget.spent(), CorrectionsLeft: e.completion.left(),
		})
		if err != nil {
			return err
		}
		for _, call := range e.recorder.annotate(verdict.Notes) {
			if err := e.request.Journal.SaveToolCall(ctx, call); err != nil {
				return err
			}
		}
		if pending {
			e.dropUndelivered(nil, text, false)
			return nil
		}
		switch verdict.Kind {
		case Correct:
			if !e.budget.spent() && e.completion.take() {
				e.correction = verdict.Prompt
				e.rejected = e.dropUndelivered(nil, text, true)
				return e.inputs.rerun()
			}
			reason := ReasonGuardRejected
			if e.budget.spent() {
				reason = ReasonBudgetExhausted
			}
			if !e.completion.fallback(ctx, reason) {
				return ErrGuardRejected
			}
			if err := e.save(ctx); err != nil {
				return err
			}
			completion, text = e.completion.current(), ""
		case Finish:
			if err := e.completion.register(ctx, Completion{Source: FromGuard, Value: verdict.Value, Fixed: verdict.Fixed}); err != nil {
				return err
			}
			completion, text = e.completion.current(), ""
		}
	}
	if completion != nil && completion.Source == FromFallback {
		if err := e.save(ctx); err != nil {
			return err
		}
	}
	finished, err := e.inputs.finish(ctx, turn, completion, text)
	if err != nil {
		return err
	}
	if finished {
		e.result.Text, e.result.Completion, e.finished = text, completion, true
		return nil
	}
	// A superseded guard completion decided on the candidate text, which was
	// not delivered either.
	if completion != nil && completion.Source == FromGuard {
		e.dropUndelivered(nil, candidate, true)
		return nil
	}
	e.dropUndelivered(completion, text, false)
	return nil
}

// dropUndelivered removes output that was not delivered from the model
// history and returns it: completion calls that never ran, a superseded
// non-fixed completion, and direct text when the request discards it or
// force is set (a rejected candidate).
func (e *execution) dropUndelivered(completion *Completion, text string, force bool) []*schema.AgenticMessage {
	messages := e.history.messages
	if len(messages) == 0 {
		return nil
	}
	last := len(messages) - 1
	if e.onlyCompletionCalls(messages[last]) {
		e.history.messages = messages[:last]
		return messages[last:]
	}
	if completion != nil && !completion.Fixed && completion.Source == FromTool {
		if last >= 1 && e.onlyCompletionCalls(messages[last-1]) && messages[last].Role != schema.AgenticRoleTypeAssistant {
			e.history.messages = messages[:last-1]
			return messages[last-1:]
		}
		return nil
	}
	if text == "" || (!force && !e.request.DiscardUndelivered) {
		return nil
	}
	if messages[last].Role != schema.AgenticRoleTypeAssistant || hasToolCalls(messages[last]) || strings.TrimSpace(llm.Text(messages[last])) != text {
		return nil
	}
	e.history.messages = messages[:last]
	return messages[last:]
}

// onlyCompletionCalls reports whether message is a model output that only
// calls completion tools.
func (e *execution) onlyCompletionCalls(message *schema.AgenticMessage) bool {
	calls := toolCalls(message)
	return message.Role == schema.AgenticRoleTypeAssistant && len(calls) > 0 &&
		!slices.ContainsFunc(calls, func(c *schema.FunctionToolCall) bool { return !e.completion.isTool(c.Name) })
}

// dropCompletionTail removes a superseded completion call at the end of a
// restored context before new input is appended.
func (e *execution) dropCompletionTail() {
	messages := e.history.messages
	for len(messages) > 0 {
		last := messages[len(messages)-1]
		if e.onlyCompletionCalls(last) || (len(resultCallIDs(last)) > 0 && len(messages) >= 2 && e.onlyCompletionCalls(messages[len(messages)-2])) {
			messages = messages[:len(messages)-1]
			continue
		}
		break
	}
	e.history.messages = messages
}

// afterToolCalls suspends the run when a call waits for an external result,
// and otherwise delivers new input, preempting at the safe point.
func (e *execution) afterToolCalls(ctx context.Context) error {
	// A safe point after every batch keeps corrections, fallbacks and
	// extension state changed by tools.
	if err := e.save(ctx); err != nil {
		return err
	}
	if e.recorder.awaiting() {
		return errSuspended
	}
	return e.inputs.poll(ctx, true)
}

// totalUsage returns the usage of the run.
func (e *execution) totalUsage() llm.Usage {
	total := e.recorder.modelUsage()
	total.Add(e.retry.discarded())
	return total
}

// save writes the changes since the last save with the run's checkpoint.
func (e *execution) save(ctx context.Context) error {
	e.saveMu.Lock()
	defer e.saveMu.Unlock()
	encoded, err := e.encodeState()
	if err != nil {
		return err
	}
	return e.recorder.saveStep(ctx, Step{Usage: e.totalUsage(), Plan: e.recorder.currentPlan(), Completion: e.completion.current(), State: encoded})
}

// encodeState encodes the checkpoint; the caller holds saveMu.
func (e *execution) encodeState() ([]byte, error) {
	messages, err := checkpoint.EncodeMessages(e.context)
	if err != nil {
		return nil, err
	}
	used, active := e.completion.snapshot()
	s := state{
		Version: checkpointVersion, Messages: messages, Seen: slices.Sorted(maps.Keys(e.history.seen)),
		MediaCount: e.history.mediaCount, MediaBytes: e.history.mediaBytes, MediaEnabled: e.media.enabled.Load(),
		ClaimedSeq: e.inputs.boundary(), Turns: e.turns, Iterations: e.budget.count(), Corrections: used,
		Completion: active, Usage: e.totalUsage(),
	}
	for name, ext := range e.extensions.stateful {
		data, err := ext.Save()
		if err != nil {
			return nil, fmt.Errorf("einorun: save extension %s: %w", name, err)
		}
		if s.Extensions == nil {
			s.Extensions = map[string]json.RawMessage{}
		}
		s.Extensions[name] = data
	}
	return json.Marshal(s)
}

// decodeState decodes a checkpoint of the current version.
func decodeState(data []byte) (state, error) {
	var s state
	if err := json.Unmarshal(data, &s); err != nil {
		return state{}, fmt.Errorf("einorun: decode checkpoint: %w", err)
	}
	if s.Version != checkpointVersion {
		return state{}, fmt.Errorf("einorun: checkpoint version %d, want %d", s.Version, checkpointVersion)
	}
	return s, nil
}

// restore rebuilds the execution from the checkpoint and the latest records
// in the order the design fixes. It returns true when the run stays
// suspended.
func (e *execution) restore(ctx context.Context) (bool, error) {
	s := e.restored
	resume := e.request.Resume
	messages, err := checkpoint.DecodeMessages(s.Messages)
	if err != nil {
		return false, err
	}
	blocks := cloneBlocks(resume.Blocks)
	children := make([]ToolCall, len(resume.Calls))
	for i, c := range resume.Calls {
		children[i] = c.Clone()
	}
	// A trailing output that only calls completion tools, none of which
	// settled, never took effect; it is dropped and generated again.
	var voided []Block
	if last := len(messages) - 1; last >= 0 && e.onlyCompletionCalls(messages[last]) {
		ids := map[string]bool{}
		for _, c := range toolCalls(messages[last]) {
			ids[c.CallID] = true
		}
		settled := slices.ContainsFunc(blocks, func(b Block) bool {
			return b.Call != nil && ids[b.Call.CallID] && b.Call.Status.Settled()
		})
		if !settled && s.Completion == nil {
			messages = messages[:last]
			blocks = slices.DeleteFunc(blocks, func(b Block) bool {
				if b.Call != nil && ids[b.Call.CallID] {
					voided = append(voided, b)
					return true
				}
				return false
			})
		}
	}
	changed, waiting := settleInterrupted(blocks, children, time.Now(), e.text)
	e.recorder.restore(blocks, children, resume.Plan)
	e.recorder.markChanged(voided, changed)
	e.recorder.mu.Lock()
	e.recorder.usage = s.Usage
	e.recorder.mu.Unlock()
	for _, call := range changed {
		if err := e.request.Journal.SaveToolCall(ctx, call); err != nil {
			return false, err
		}
	}
	for _, o := range e.extensions.restores {
		if err := o.AfterRestore(ctx, RestoreView{Blocks: cloneBlocks(blocks), Calls: children}); err != nil {
			return false, err
		}
	}
	if waiting {
		return true, nil
	}
	// A trailing output with text and no calls was never delivered.
	if last := len(messages) - 1; last >= 0 && messages[last].Role == schema.AgenticRoleTypeAssistant && !hasToolCalls(messages[last]) {
		messages = messages[:last]
	}
	// Results of the last output's calls come from the latest records.
	if last := len(messages) - 1; last >= 0 && messages[last].Role == schema.AgenticRoleTypeAssistant {
		records := map[string]*ToolCall{}
		for _, b := range blocks {
			if b.Call != nil {
				records[b.Call.CallID] = b.Call
			}
		}
		for _, c := range toolCalls(messages[last]) {
			record, ok := records[c.CallID]
			if !ok {
				continue
			}
			if result, ok := modelResult(record); ok {
				e.patched[c.CallID] = result
				e.observe(ctx, CallOutcome{Agent: e.main.scope, Name: c.Name, Call: record.Clone(), Raw: result, Origin: OriginRestored})
			}
		}
	}
	active := s.Completion
	if active == nil {
		active = e.completionAtTail(messages, blocks)
	}
	e.context = messages
	e.history.messages = slices.Clone(messages)
	e.history.seen = map[string]bool{}
	for _, key := range s.Seen {
		e.history.seen[key] = true
	}
	e.history.mediaCount, e.history.mediaBytes = s.MediaCount, s.MediaBytes
	e.turns = s.Turns
	e.media.enabled.Store(s.MediaEnabled)
	e.budget.resume(s.Iterations)
	e.completion.restore(s.Corrections, active)
	e.inputs.resumeFrom(s.ClaimedSeq)
	if s.ClaimedSeq > 0 {
		claim, err := e.inputs.replay(ctx)
		if err != nil {
			return false, err
		}
		e.lastClaim = claim
	}
	return false, nil
}

// restoreExtensions restores the state of stateful extensions. The
// checkpoint and the registered extensions must agree: a stateful extension
// without state, or state without its extension, is an error.
func (e *execution) restoreExtensions(s state) error {
	for name, ext := range e.extensions.stateful {
		data, ok := s.Extensions[name]
		if !ok {
			return fmt.Errorf("einorun: checkpoint has no state for extension %s", name)
		}
		if err := ext.Restore(data); err != nil {
			return fmt.Errorf("einorun: restore extension %s: %w", name, err)
		}
	}
	for name := range s.Extensions {
		if _, ok := e.extensions.stateful[name]; !ok {
			return fmt.Errorf("einorun: checkpoint has state for extension %s, which is not registered", name)
		}
	}
	return nil
}

// completionAtTail returns the completion of a successful completion call
// that ends the context, for a crash between saving the call and saving the
// step.
func (e *execution) completionAtTail(messages []*schema.AgenticMessage, blocks []Block) *Completion {
	if len(messages) == 0 || !e.onlyCompletionCalls(messages[len(messages)-1]) {
		return nil
	}
	ids := map[string]bool{}
	for _, c := range toolCalls(messages[len(messages)-1]) {
		ids[c.CallID] = true
	}
	for _, b := range blocks {
		if call := b.Call; call != nil && ids[call.CallID] && call.Status == StatusSucceeded && call.Completion != nil {
			return &Completion{Source: FromTool, CallID: call.ID, Value: slices.Clone(call.Completion.Value), Fixed: call.Completion.Fixed}
		}
	}
	return nil
}

// finishRestored delivers an active completion of a restored run: a fixed one
// at once, a non-fixed one when no new input is pending; otherwise the
// superseded completion leaves the history and the loop continues.
func (e *execution) finishRestored(ctx context.Context) error {
	active := e.completion.current()
	if active == nil {
		return nil
	}
	if !active.Fixed {
		boundary := e.inputs.boundary()
		latest, err := e.request.Feed.Pending(ctx, boundary)
		if err != nil {
			return err
		}
		if latest > boundary {
			e.completion.supersede()
			e.dropCompletionTail()
			return nil
		}
	}
	e.result.Completion, e.finished = active, true
	return nil
}

// settleInterrupted settles calls that did not finish before the run
// stopped, by their traits, and reports whether a main-agent call still waits
// for an external result. Calls handed over or submitted stay with whoever
// advances them; sub-agent calls never keep the run suspended.
func settleInterrupted(blocks []Block, children []ToolCall, at time.Time, text *prompt.Runtime) (changed []ToolCall, waiting bool) {
	settle := func(call *ToolCall, main bool) {
		if call.Status.Settled() || call.Handover == HandoverDetached || call.Handover == HandoverSubmitted {
			return
		}
		if call.Handover == HandoverAwait {
			waiting = waiting || main
			return
		}
		status, result := interrupted(call.Replayable, call.SideEffects, text)
		call.Status, call.Result, call.CompletedAt = status, &result, &at
		call.Rev++
		changed = append(changed, call.Clone())
	}
	for i := range blocks {
		if blocks[i].Call != nil {
			settle(blocks[i].Call, true)
		}
	}
	for i := range children {
		settle(&children[i], false)
	}
	return changed, waiting
}

// interrupted returns the status and model-visible result of an interrupted
// call.
func interrupted(replayable, sideEffects bool, text *prompt.Runtime) (CallStatus, string) {
	switch {
	case sideEffects && !replayable:
		return StatusNeedsReview, text.NeedsReview
	case replayable:
		return StatusInterrupted, text.InterruptedReplayable
	}
	return StatusInterrupted, text.Interrupted
}

// modelResult returns what the model sees for a call record: a receipt, the
// error or the result.
func modelResult(call *ToolCall) (string, bool) {
	switch {
	case call.Handover == HandoverDetached || call.Handover == HandoverSubmitted:
		if call.Result != nil {
			return *call.Result, true
		}
	case call.Status == StatusFailed && call.Error != nil:
		return errorResult(errors.New(*call.Error)), true
	case call.Status.Settled() && call.Result != nil:
		return *call.Result, true
	}
	return "", false
}
