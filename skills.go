package einorun

import (
	"cmp"
	"context"
	"errors"
	"fmt"

	"github.com/cloudwego/eino/adk"
	"github.com/cloudwego/eino/adk/middlewares/skill"
	"github.com/cloudwego/eino/schema"

	"github.com/runforyou-ai/einorun/llm"
)

const (
	// DefaultSkillTool is the default name of the skill tool.
	DefaultSkillTool = "skill"
	// skillsExtensionName is the name of the Skills extension.
	skillsExtensionName = "einorun.skills"
)

// SkillsSpec configures skills. Backend is required; the other fields are
// optional and passed to Eino's skill middleware as they are.
type SkillsSpec struct {
	// Backend lists and loads the skills.
	Backend skill.Backend
	// ToolName is the name of the skill tool; DefaultSkillTool when empty.
	ToolName string
	// SystemPrompt renders the instruction added for skills.
	SystemPrompt skill.SystemPromptFunc
	// ToolDescription renders the skill tool's description.
	ToolDescription skill.ToolDescriptionFunc
	// ToolParams adjusts the skill tool's parameters.
	ToolParams func(ctx context.Context, defaults map[string]*schema.ParameterInfo) (map[string]*schema.ParameterInfo, error)
	// FormatReminder renders the reminder that lists the skills.
	FormatReminder func(ctx context.Context, in *skill.FormatReminderInput) (*skill.FormatReminderOutput, error)
	// BuildContent renders a loaded skill, for example with the list of its
	// resources.
	BuildContent func(ctx context.Context, loaded skill.Skill, rawArgs string) (string, error)
	// BuildForkMessages builds the input of a skill run by a sub-agent.
	BuildForkMessages func(ctx context.Context, in skill.TypedSubAgentInput[*schema.AgenticMessage]) ([]*schema.AgenticMessage, error)
	// FormatForkResult renders the result of a skill run by a sub-agent; by
	// default the sub-agent's last answer.
	FormatForkResult func(ctx context.Context, out skill.TypedSubAgentOutput[*schema.AgenticMessage]) (string, error)
}

// Skills returns the extension that gives agents a tool to load skills.
// Skills that declare a forked context run in a sub-agent when the run also
// has the Subagent extension; otherwise, and inside sub-agents, they load in
// the current context. The skill tool's results are never offloaded or
// cleared and are kept verbatim by summaries.
func Skills(spec SkillsSpec) Extension {
	spec.ToolName = cmp.Or(spec.ToolName, DefaultSkillTool)
	return &skillsExtension{spec: spec}
}

// skillsExtension creates the Skills instance of a run.
type skillsExtension struct{ spec SkillsSpec }

// Name returns the extension name.
func (s *skillsExtension) Name() string { return skillsExtensionName }

// Instance returns the run's instance.
func (s *skillsExtension) Instance(context.Context, RunScope) (Extension, error) {
	if s.spec.Backend == nil {
		return nil, errors.New("einorun: Skills needs a backend")
	}
	return &skillsInstance{spec: s.spec}, nil
}

// skillsInstance is the Skills extension of one run.
type skillsInstance struct {
	spec SkillsSpec
	main adk.TypedChatModelAgentMiddleware[*schema.AgenticMessage]
	sub  adk.TypedChatModelAgentMiddleware[*schema.AgenticMessage]
}

// Name returns the extension name.
func (s *skillsInstance) Name() string { return skillsExtensionName }

// bind declares the skill tool and creates the middlewares of the main agent
// and of sub-agents.
func (s *skillsInstance) bind(ctx context.Context, e *execution) error {
	if err := e.declareTool(s.spec.ToolName, ToolSpec{Replayable: true, Retain: RetainIntact, PinInSummary: true}, nil); err != nil {
		return err
	}
	var hub skill.TypedAgentHub[*schema.AgenticMessage]
	if e.subagents != nil {
		hub = e.subagents.agent
	}
	var err error
	if s.main, err = s.middleware(ctx, e, hub); err != nil {
		return err
	}
	s.sub, err = s.middleware(ctx, e, nil)
	return err
}

// middleware creates a skill middleware; without a hub, forked skills load
// in the current context.
func (s *skillsInstance) middleware(ctx context.Context, e *execution, hub skill.TypedAgentHub[*schema.AgenticMessage]) (adk.TypedChatModelAgentMiddleware[*schema.AgenticMessage], error) {
	backend := s.spec.Backend
	if hub == nil {
		backend = inlineSkills{backend}
	}
	name := s.spec.ToolName
	forkResult := s.spec.FormatForkResult
	if forkResult == nil {
		forkResult = e.forkResult
	}
	config := &skill.TypedConfig[*schema.AgenticMessage]{
		Backend: backend, SkillToolName: &name, AgentHub: hub,
		CustomSystemPrompt: s.spec.SystemPrompt, CustomToolDescription: s.spec.ToolDescription,
		CustomToolParams: s.spec.ToolParams, CustomFormatReminder: s.spec.FormatReminder,
		BuildContent: s.spec.BuildContent, BuildForkMessages: s.spec.BuildForkMessages,
		FormatForkResult: forkResult,
	}
	// Eino's default skill text follows the run's language.
	config.UseChinese = e.language == llm.Chinese //nolint:staticcheck // The middleware-level switch keeps the global language untouched.
	middleware, err := skill.NewTyped(ctx, config)
	if err != nil {
		return nil, fmt.Errorf("einorun: create the skill middleware: %w", err)
	}
	return middleware, nil
}

// ModelMiddlewares adds the skill tool to every agent.
func (s *skillsInstance) ModelMiddlewares(scope AgentScope) []adk.TypedChatModelAgentMiddleware[*schema.AgenticMessage] {
	m := s.sub
	if scope.Main {
		m = s.main
	}
	if m == nil {
		return nil
	}
	return []adk.TypedChatModelAgentMiddleware[*schema.AgenticMessage]{m}
}

// inlineSkills loads every skill in the current context.
type inlineSkills struct{ skill.Backend }

// List returns the skills without a forked context.
func (b inlineSkills) List(ctx context.Context) ([]skill.FrontMatter, error) {
	matters, err := b.Backend.List(ctx)
	for i := range matters {
		matters[i].Context = ""
	}
	return matters, err
}

// Get returns the skill without a forked context.
func (b inlineSkills) Get(ctx context.Context, name string) (skill.Skill, error) {
	loaded, err := b.Backend.Get(ctx, name)
	loaded.Context = ""
	return loaded, err
}
