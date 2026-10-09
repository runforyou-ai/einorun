package einorun

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"

	"github.com/cloudwego/eino/adk"
	"github.com/cloudwego/eino/adk/middlewares/skill"
	"github.com/cloudwego/eino/adk/middlewares/subagent"
	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"

	"github.com/runforyou-ai/einorun/llm"
)

const (
	// DefaultSubagentTool is the default name of the delegation tool.
	DefaultSubagentTool = "agent"
	// DefaultSubagentName is the default name of the sub-agent type.
	DefaultSubagentName = "general"
	// subagentExtensionName is the name of the Subagent extension.
	subagentExtensionName = "einorun.subagent"
)

// SubagentSpec configures delegation to sub-agents.
type SubagentSpec struct {
	// ToolName is the name of the delegation tool; DefaultSubagentTool when
	// empty.
	ToolName string
	// Name is the sub-agent type the model selects; DefaultSubagentName when
	// empty.
	Name string
	// Description tells the main agent what the sub-agent is for; a general
	// description in the run's language when empty.
	Description string
	// Instruction is the sub-agent's instruction.
	Instruction string
	// MaxIterations is the iteration budget of one delegation; the run's
	// budget when zero.
	MaxIterations int
}

// Subagent returns the extension that lets the main agent hand tasks to
// sub-agents. A sub-agent uses the main agent's tools except tools marked
// MainOnly, delegation and the task list, with the same model factory and
// its own middlewares; it shares extension instances, the media budget and
// the multimodal switch with the main agent. Its calls are recorded under
// the delegation call (ToolCall.ParentID) and reported in Result.Calls.
//
// The delegation tool is replayable and has side effects when a tool the
// sub-agent can use has them. Skills that run in a forked context use the
// same sub-agent.
func Subagent(spec SubagentSpec) Extension {
	spec.ToolName = cmp.Or(spec.ToolName, DefaultSubagentTool)
	spec.Name = cmp.Or(spec.Name, DefaultSubagentName)
	return &subagentExtension{spec: spec}
}

// subagentExtension creates the Subagent instance of a run.
type subagentExtension struct{ spec SubagentSpec }

// Name returns the extension name.
func (s *subagentExtension) Name() string { return subagentExtensionName }

// Instance returns the run's instance.
func (s *subagentExtension) Instance(context.Context, RunScope) (Extension, error) {
	return &subagentInstance{spec: s.spec}, nil
}

// subagentInstance is the Subagent extension of one run.
type subagentInstance struct {
	spec       SubagentSpec
	e          *execution
	agent      *subagentType
	middleware adk.TypedChatModelAgentMiddleware[*schema.AgenticMessage]
}

// subagentType is the sub-agent type the delegation middleware runs and the
// agent hub of forked skills.
type subagentType struct{ s *subagentInstance }

// Name returns the extension name.
func (s *subagentInstance) Name() string { return subagentExtensionName }

// bind declares the delegation tool and creates the delegation middleware.
func (s *subagentInstance) bind(ctx context.Context, e *execution) error {
	s.e, s.agent = e, &subagentType{s: s}
	s.spec.Description = cmp.Or(s.spec.Description, e.text.SubagentDescription)
	if err := e.declareTool(s.spec.ToolName, e.delegationSpec(), describeDelegation); err != nil {
		return err
	}
	guide := fmt.Sprintf(e.text.SubagentGuide, s.spec.ToolName)
	middleware, err := subagent.NewTyped(ctx, &subagent.TypedConfig[*schema.AgenticMessage]{
		SubAgents: []adk.TypedAgent[*schema.AgenticMessage]{s.agent},
		ToolName:  s.spec.ToolName,
		ToolDescriptionGenerator: func(context.Context, []adk.TypedAgent[*schema.AgenticMessage]) (string, error) {
			return e.text.SubagentTool, nil
		},
		SystemPrompt: &guide,
		CustomFormatReminder: func(context.Context, *subagent.FormatReminderInput[*schema.AgenticMessage]) (*subagent.FormatReminderOutput, error) {
			return &subagent.FormatReminderOutput{Reminder: e.text.SubagentTypes + "\n- " + s.spec.Name + ": " + s.spec.Description}, nil
		},
	})
	if err != nil {
		return fmt.Errorf("einorun: create the delegation middleware: %w", err)
	}
	s.middleware = middleware
	return nil
}

// ModelMiddlewares adds the delegation tool to the main agent.
func (s *subagentInstance) ModelMiddlewares(scope AgentScope) []adk.TypedChatModelAgentMiddleware[*schema.AgenticMessage] {
	if !scope.Main || s.middleware == nil {
		return nil
	}
	return []adk.TypedChatModelAgentMiddleware[*schema.AgenticMessage]{s.middleware}
}

// delegationSpec returns the traits of a call a sub-agent serves: with side
// effects, and not to be repeated after an interruption, when a tool the
// sub-agent can use has side effects.
func (e *execution) delegationSpec() ToolSpec {
	sideEffects := slices.ContainsFunc(e.toolSpecs(AgentScope{}), func(spec ToolSpec) bool { return spec.SideEffects })
	return ToolSpec{Replayable: !sideEffects, SideEffects: sideEffects}
}

// describeDelegation returns the task description of a delegation call.
func describeDelegation(arguments string) string {
	var parsed struct {
		Description string `json:"description"`
	}
	if json.Unmarshal([]byte(arguments), &parsed) != nil {
		return ""
	}
	return parsed.Description
}

// Name returns the sub-agent type's name.
func (t *subagentType) Name(context.Context) string { return t.s.spec.Name }

// Description returns what the sub-agent is for, as the main agent sees it.
func (t *subagentType) Description(context.Context) string { return t.s.spec.Description }

// Get returns the sub-agent for forked skills.
func (t *subagentType) Get(context.Context, string, *skill.TypedAgentHubOptions[*schema.AgenticMessage]) (adk.TypedAgent[*schema.AgenticMessage], error) {
	return t, nil
}

// Run builds a sub-agent for one delegation, the tool call in ctx, and
// forwards its events; the sub-agent's tools are released when it ends.
func (t *subagentType) Run(ctx context.Context, input *adk.TypedAgentInput[*schema.AgenticMessage], options ...adk.AgentRunOption) *adk.AsyncIterator[*adk.TypedAgentEvent[*schema.AgenticMessage]] {
	iterator, generator := adk.NewAsyncIteratorPair[*adk.TypedAgentEvent[*schema.AgenticMessage]]()
	fail := func(err error) *adk.AsyncIterator[*adk.TypedAgentEvent[*schema.AgenticMessage]] {
		generator.Send(&adk.TypedAgentEvent[*schema.AgenticMessage]{Err: err})
		generator.Close()
		return iterator
	}
	call, ok := CallFrom(ctx)
	if !ok {
		return fail(errors.New("einorun: a sub-agent runs only from a tool call"))
	}
	built, done, err := t.s.e.newSubagent(ctx, t.s.spec, call.ProviderCallID, input)
	if err != nil {
		return fail(err)
	}
	events := built.Run(ctx, input, options...)
	go func() {
		defer generator.Close()
		defer done()
		for {
			event, ok := events.Next()
			if !ok {
				return
			}
			generator.Send(event)
		}
	}()
	return iterator
}

// newSubagent builds a sub-agent serving the main-agent call with the
// provider call ID parentCallID. The returned function releases its tools.
func (e *execution) newSubagent(ctx context.Context, spec SubagentSpec, parentCallID string, input *adk.TypedAgentInput[*schema.AgenticMessage]) (adk.TypedAgent[*schema.AgenticMessage], func(), error) {
	scope := AgentScope{Name: spec.Name}
	maxIterations := cmp.Or(spec.MaxIterations, e.budget.max)
	a := &agent{
		scope: scope, parentCallID: parentCallID,
		budget:    &budgetGuard{max: maxIterations, notice: e.text.FinalNotice},
		injector:  e.newInjector(),
		offloaded: newOffloadStore(nil),
	}
	recorder := &subagentRecorder{e: e, a: a}
	a.recorder = recorder
	a.retry = &modelRetry{runID: e.request.RunID, enabled: e.media.enabled, text: e.text, discard: recorder.discarded}
	tools, releases, err := e.buildTools(ctx, scope)
	done := func() {
		for _, release := range releases {
			release()
		}
	}
	if err != nil {
		done()
		return nil, nil, err
	}
	built, err := e.buildAgent(ctx, a, spec.Instruction, tools, maxIterations)
	if err != nil {
		done()
		return nil, nil, err
	}
	// Summaries keep the delegated task itself.
	if len(input.Messages) > 0 {
		a.summary.keepFrom(input.Messages[len(input.Messages)-1])
	}
	return built, done, nil
}

// subagentRecorder takes the recorder's place in a sub-agent: every model
// call carries a ModelCallID, the calls of an output remember it, and each
// finalized output adds its usage to the run and is saved with the next
// checkpoint at once. The outputs are not part of the run's process.
type subagentRecorder struct {
	adk.TypedBaseChatModelAgentMiddleware[*schema.AgenticMessage]
	e *execution
	a *agent

	mu      sync.Mutex
	last    string // the latest model call; a sub-agent calls its model one at a time
	saveErr error  // a failed save of discarded usage, reported by the next model call
}

// discarded adds the usage of a discarded output to the run and saves it.
// The retry cannot fail, so a save error ends the next model call.
func (r *subagentRecorder) discarded(used llm.Usage) {
	r.e.agentsMu.Lock()
	r.e.subUsage.Add(used)
	r.e.agentsMu.Unlock()
	if err := r.e.save(context.Background()); err != nil {
		r.mu.Lock()
		r.saveErr = err
		r.mu.Unlock()
	}
}

// WrapModel assigns a ModelCallID to every model call.
func (r *subagentRecorder) WrapModel(_ context.Context, m model.BaseModel[*schema.AgenticMessage], _ *adk.TypedModelContext[*schema.AgenticMessage]) (model.BaseModel[*schema.AgenticMessage], error) {
	return &identifiedModel{BaseModel: m, recorder: r}, nil
}

// begin starts a model call and returns its ModelCallID, or the error of a
// failed save.
func (r *subagentRecorder) begin() (string, error) {
	id := llm.NewModelCallID()
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.saveErr != nil {
		return "", &abortError{err: r.saveErr}
	}
	r.last = id
	return id, nil
}

// AfterModelRewriteState links the output's calls to its model call, adds its
// usage to the run and saves it.
func (r *subagentRecorder) AfterModelRewriteState(ctx context.Context, state *adk.TypedChatModelAgentState[*schema.AgenticMessage], _ *adk.TypedModelContext[*schema.AgenticMessage]) (context.Context, *adk.TypedChatModelAgentState[*schema.AgenticMessage], error) {
	n := len(state.Messages)
	if n == 0 {
		return ctx, state, nil
	}
	output := state.Messages[n-1]
	r.mu.Lock()
	id := r.last
	r.mu.Unlock()
	for _, c := range toolCalls(output) {
		r.a.modelCalls.Store(c.CallID, id)
	}
	r.e.agentsMu.Lock()
	r.e.subUsage.Add(llm.UsageOf(output.ResponseMeta))
	r.e.agentsMu.Unlock()
	if err := r.e.save(ctx); err != nil {
		return ctx, state, &abortError{err: err}
	}
	return ctx, state, nil
}

// identifiedModel calls a model with a new ModelCallID in the context.
type identifiedModel struct {
	model.BaseModel[*schema.AgenticMessage]
	recorder *subagentRecorder
}

// Generate calls the model with a new ModelCallID.
func (m *identifiedModel) Generate(ctx context.Context, input []*schema.AgenticMessage, opts ...model.Option) (*schema.AgenticMessage, error) {
	id, err := m.recorder.begin()
	if err != nil {
		return nil, err
	}
	return m.BaseModel.Generate(llm.WithModelCallID(ctx, id), input, opts...)
}

// Stream calls the model with a new ModelCallID.
func (m *identifiedModel) Stream(ctx context.Context, input []*schema.AgenticMessage, opts ...model.Option) (*schema.StreamReader[*schema.AgenticMessage], error) {
	id, err := m.recorder.begin()
	if err != nil {
		return nil, err
	}
	return m.BaseModel.Stream(llm.WithModelCallID(ctx, id), input, opts...)
}

// forkResult returns the result of a skill a sub-agent ran: its last answer.
func (e *execution) forkResult(_ context.Context, output skill.TypedSubAgentOutput[*schema.AgenticMessage]) (string, error) {
	result := e.text.SubagentNoResult
	for i := len(output.Results) - 1; i >= 0; i-- {
		if text := strings.TrimSpace(output.Results[i]); text != "" {
			result = text
			break
		}
	}
	return fmt.Sprintf(e.text.SkillForkResult, output.Skill.Name, result), nil
}
