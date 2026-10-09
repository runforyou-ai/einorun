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
	sideEffects := slices.ContainsFunc(e.toolSpecs(AgentScope{}), func(spec ToolSpec) bool { return spec.SideEffects })
	if err := e.declareTool(s.spec.ToolName, ToolSpec{Replayable: true, SideEffects: sideEffects}, describeDelegation); err != nil {
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
// forwards its events; the sub-agent's tools are released and its usage is
// added to the run when it ends.
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
// provider call ID parentCallID. The returned function releases its tools and
// adds its usage to the run.
func (e *execution) newSubagent(ctx context.Context, spec SubagentSpec, parentCallID string, input *adk.TypedAgentInput[*schema.AgenticMessage]) (adk.TypedAgent[*schema.AgenticMessage], func(), error) {
	scope := AgentScope{Name: spec.Name}
	maxIterations := cmp.Or(spec.MaxIterations, e.budget.max)
	a := &agent{
		scope: scope, parentCallID: parentCallID,
		budget:    &budgetGuard{max: maxIterations, notice: e.text.FinalNotice},
		injector:  e.newInjector(),
		offloaded: newOffloadStore(nil),
		retry:     &modelRetry{runID: e.request.RunID, enabled: e.media.enabled, text: e.text},
	}
	tools, releases, err := e.buildTools(ctx, scope)
	done := func() {
		for _, release := range releases {
			release()
		}
		used := a.usage.total()
		used.Add(a.retry.discarded())
		e.agentsMu.Lock()
		e.subUsage.Add(used)
		e.agentsMu.Unlock()
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

// usageTotal adds up the usage of an agent's model outputs.
type usageTotal struct {
	mu    sync.Mutex
	usage llm.Usage
}

// add adds usage.
func (u *usageTotal) add(usage llm.Usage) {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.usage.Add(usage)
}

// total returns the usage added.
func (u *usageTotal) total() llm.Usage {
	u.mu.Lock()
	defer u.mu.Unlock()
	return u.usage
}

// subagentRecorder takes the recorder's place in a sub-agent: every model
// call carries a ModelCallID and finalized outputs count usage; the outputs
// are not part of the run's process.
type subagentRecorder struct {
	adk.TypedBaseChatModelAgentMiddleware[*schema.AgenticMessage]
	usage *usageTotal
}

// WrapModel assigns a ModelCallID to every model call.
func (r *subagentRecorder) WrapModel(_ context.Context, m model.BaseModel[*schema.AgenticMessage], _ *adk.TypedModelContext[*schema.AgenticMessage]) (model.BaseModel[*schema.AgenticMessage], error) {
	return &identifiedModel{BaseModel: m}, nil
}

// AfterModelRewriteState counts the usage of the finalized output.
func (r *subagentRecorder) AfterModelRewriteState(ctx context.Context, state *adk.TypedChatModelAgentState[*schema.AgenticMessage], _ *adk.TypedModelContext[*schema.AgenticMessage]) (context.Context, *adk.TypedChatModelAgentState[*schema.AgenticMessage], error) {
	if n := len(state.Messages); n > 0 {
		r.usage.add(llm.UsageOf(state.Messages[n-1].ResponseMeta))
	}
	return ctx, state, nil
}

// identifiedModel calls a model with a new ModelCallID in the context.
type identifiedModel struct {
	model.BaseModel[*schema.AgenticMessage]
}

// Generate calls the model with a new ModelCallID.
func (m *identifiedModel) Generate(ctx context.Context, input []*schema.AgenticMessage, opts ...model.Option) (*schema.AgenticMessage, error) {
	return m.BaseModel.Generate(llm.WithModelCallID(ctx, llm.NewModelCallID()), input, opts...)
}

// Stream calls the model with a new ModelCallID.
func (m *identifiedModel) Stream(ctx context.Context, input []*schema.AgenticMessage, opts ...model.Option) (*schema.StreamReader[*schema.AgenticMessage], error) {
	return m.BaseModel.Stream(llm.WithModelCallID(ctx, llm.NewModelCallID()), input, opts...)
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
