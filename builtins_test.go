package einorun_test

import (
	"context"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/cloudwego/eino/adk/middlewares/skill"
	"github.com/cloudwego/eino/schema"

	"github.com/runforyou-ai/einorun"
	"github.com/runforyou-ai/einorun/inmem"
	"github.com/runforyou-ai/einorun/llm"
	"github.com/runforyou-ai/einorun/stream"
)

func TestSubagentDelegation(t *testing.T) {
	m := &scripted{steps: []step{
		call(invocation{"agent", `{"subagent_type":"general","prompt":"look up a","description":"Look up a"}`}),
		call(invocation{"lookup", `{"x":"a"}`}),
		func(input []*schema.AgenticMessage) *schema.AgenticMessage {
			return say("a is " + lastUser(input))(input)
		},
		func(input []*schema.AgenticMessage) *schema.AgenticMessage {
			return say("done: " + lastUser(input))(input)
		},
	}}
	feed := inmem.NewFeed()
	user(feed, "m1", "find a")
	var deltas []stream.Delta
	var mu sync.Mutex
	result, err := run(t, einorun.Request{RunID: "r1", Model: einorun.Model{New: m.factory}, Feed: feed,
		Tools:      []einorun.ToolSpec{{Tool: echo("lookup"), SideEffects: true}},
		Extensions: []einorun.Extension{einorun.Subagent(einorun.SubagentSpec{Instruction: "help"})},
		Stream:     func(d stream.Delta) { mu.Lock(); deltas = append(deltas, d); mu.Unlock() }, StreamID: "s1"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result.Text, `a is lookup:{"x":"a"}`) {
		t.Fatalf("text %q", result.Text)
	}
	if len(result.Blocks) != 1 || result.Blocks[0].Call == nil {
		t.Fatalf("blocks %+v", result.Blocks)
	}
	delegation := result.Blocks[0].Call
	if delegation.Name != "agent" || delegation.Status != einorun.StatusSucceeded || !delegation.SideEffects {
		t.Fatalf("delegation %+v", delegation)
	}
	if len(result.Calls) != 1 || result.Calls[0].ParentID != delegation.ID || result.Calls[0].Name != "lookup" ||
		result.Calls[0].Status != einorun.StatusSucceeded {
		t.Fatalf("sub-agent calls %+v", result.Calls)
	}
	if result.Usage.Total != 4*12 {
		t.Fatalf("usage %+v", result.Usage)
	}
	for i, id := range m.ids {
		if id == "" {
			t.Fatalf("model call %d without a ModelCallID", i)
		}
	}
	snapshot := stream.Snapshot{Stream: "s1"}
	for _, d := range deltas {
		if _, err := snapshot.Apply(d); err != nil {
			t.Fatal(err)
		}
	}
	if len(snapshot.Blocks) != 1 || snapshot.Blocks[0].Call == nil || snapshot.Blocks[0].Call.Description != "Look up a" {
		t.Fatalf("snapshot %+v", snapshot.Blocks)
	}
}

func TestSubagentCannotAwait(t *testing.T) {
	m := &scripted{steps: []step{
		call(invocation{"agent", `{"subagent_type":"general","prompt":"run it","description":"Run"}`}),
		call(invocation{"dispatch", `{"x":"a"}`}),
		func(input []*schema.AgenticMessage) *schema.AgenticMessage {
			return say("sub: " + lastUser(input))(input)
		},
		say("done"),
	}}
	feed := inmem.NewFeed()
	user(feed, "m1", "go")
	dispatch := &fn{name: "dispatch", run: func(context.Context, string) (string, error) {
		return "", einorun.Await(nil)
	}}
	result, err := run(t, einorun.Request{Model: einorun.Model{New: m.factory}, Feed: feed,
		Tools:      []einorun.ToolSpec{{Tool: dispatch}},
		Extensions: []einorun.Extension{einorun.Subagent(einorun.SubagentSpec{})}})
	if err != nil {
		t.Fatal(err)
	}
	if result.Suspended || len(result.Calls) != 1 || result.Calls[0].Status != einorun.StatusFailed {
		t.Fatalf("result %+v calls %+v", result, result.Calls)
	}
}

func TestPlanningPublishesAndRestoresThePlan(t *testing.T) {
	m := &scripted{steps: []step{
		call(invocation{"TaskCreate", `{"subject":"Read the files","description":"read","activeForm":"Reading"}`}),
		call(invocation{"dispatch", `{"x":"a"}`}),
		call(invocation{"TaskList", `{}`}),
		func(input []*schema.AgenticMessage) *schema.AgenticMessage {
			return say("tasks: " + lastUser(input))(input)
		},
	}}
	feed := inmem.NewFeed()
	user(feed, "m1", "plan it")
	journal := inmem.NewJournal()
	dispatch := &fn{name: "dispatch", run: func(context.Context, string) (string, error) { return "", einorun.Await(nil) }}
	request := einorun.Request{RunID: "r1", Model: einorun.Model{New: m.factory}, Feed: feed, Journal: journal,
		Tools: []einorun.ToolSpec{{Tool: dispatch}}, Extensions: []einorun.Extension{einorun.Planning()}}
	result, err := run(t, request)
	if err != nil {
		t.Fatal(err)
	}
	if !result.Suspended || len(result.Plan) != 1 || result.Plan[0].Subject != "Read the files" || result.Plan[0].Status != stream.PlanPending {
		t.Fatalf("plan %+v", result.Plan)
	}
	if result.Blocks[0].Call.SideEffects || !result.Blocks[0].Call.Replayable {
		t.Fatalf("task call traits %+v", result.Blocks[0].Call)
	}
	waiting := result.Blocks[1].Call
	output := "ok"
	if err := journal.External(context.Background(), einorun.ToolCall{ID: waiting.ID, Status: einorun.StatusSucceeded, Result: &output}); err != nil {
		t.Fatal(err)
	}
	resume := journal.Resume()
	request.Resume = &resume
	result, err = run(t, request)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result.Text, "Read the files") || len(result.Plan) != 1 {
		t.Fatalf("resumed %q plan %+v", result.Text, result.Plan)
	}
}

// skills is a skill backend with fixed skills.
type skills map[string]skill.Skill

func (s skills) List(context.Context) ([]skill.FrontMatter, error) {
	var matters []skill.FrontMatter
	for _, item := range s {
		matters = append(matters, item.FrontMatter)
	}
	return matters, nil
}

func (s skills) Get(_ context.Context, name string) (skill.Skill, error) {
	item, ok := s[name]
	if !ok {
		return skill.Skill{}, errors.New("no such skill")
	}
	return item, nil
}

func TestSkillLoadsInline(t *testing.T) {
	backend := skills{"tidy": {FrontMatter: skill.FrontMatter{Name: "tidy", Description: "Tidy up", Context: skill.ContextModeFork}, Content: "Sort everything."}}
	m := &scripted{steps: []step{
		call(invocation{"skill", `{"skill":"tidy"}`}),
		func(input []*schema.AgenticMessage) *schema.AgenticMessage {
			return say("loaded: " + lastUser(input))(input)
		},
	}}
	feed := inmem.NewFeed()
	user(feed, "m1", "tidy")
	result, err := run(t, einorun.Request{Model: einorun.Model{New: m.factory}, Feed: feed,
		Extensions: []einorun.Extension{einorun.Skills(einorun.SkillsSpec{Backend: backend})}})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result.Text, "Sort everything.") || len(result.Calls) != 0 {
		t.Fatalf("text %q calls %+v", result.Text, result.Calls)
	}
}

func TestForkedSkillRunsInASubagent(t *testing.T) {
	backend := skills{"tidy": {FrontMatter: skill.FrontMatter{Name: "tidy", Description: "Tidy up", Context: skill.ContextModeFork}, Content: "Sort everything."}}
	m := &scripted{steps: []step{
		call(invocation{"skill", `{"skill":"tidy"}`}),
		call(invocation{"lookup", `{"x":"b"}`}),
		say("sorted"),
		func(input []*schema.AgenticMessage) *schema.AgenticMessage {
			return say("main: " + lastUser(input))(input)
		},
	}}
	feed := inmem.NewFeed()
	user(feed, "m1", "tidy")
	result, err := run(t, einorun.Request{Model: einorun.Model{New: m.factory}, Feed: feed,
		Tools: []einorun.ToolSpec{{Tool: echo("lookup")}},
		Extensions: []einorun.Extension{
			einorun.Skills(einorun.SkillsSpec{Backend: backend}),
			einorun.Subagent(einorun.SubagentSpec{}),
		}})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result.Text, "sorted") || len(result.Calls) != 1 || result.Calls[0].ParentID != result.Blocks[0].Call.ID {
		t.Fatalf("text %q calls %+v", result.Text, result.Calls)
	}
	// The skill call is kept intact and pinned: its name is recorded as is.
	if result.Blocks[0].Call.Name != "skill" {
		t.Fatalf("skill call %+v", result.Blocks[0].Call)
	}
}

func TestBuiltinToolNamesClash(t *testing.T) {
	m := &scripted{steps: []step{say("hi")}}
	feed := inmem.NewFeed()
	user(feed, "m1", "hi")
	_, err := run(t, einorun.Request{Model: einorun.Model{New: m.factory}, Feed: feed,
		Tools:      []einorun.ToolSpec{{Tool: echo("agent")}},
		Extensions: []einorun.Extension{einorun.Subagent(einorun.SubagentSpec{})}})
	if err == nil || !strings.Contains(err.Error(), "registered twice") {
		t.Fatalf("err %v", err)
	}
}

// stepBudget fails every step save after the first n.
type stepBudget struct {
	*inmem.Journal
	mu sync.Mutex
	n  int
}

var errStepRefused = errors.New("step refused")

func (j *stepBudget) SaveStep(ctx context.Context, step einorun.Step) error {
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.n == 0 {
		return errStepRefused
	}
	j.n--
	return j.Journal.SaveStep(ctx, step)
}

func TestPlanChangesAreSavedBeforeTheCallSucceeds(t *testing.T) {
	m := &scripted{steps: []step{call(invocation{"TaskCreate", `{"subject":"a","description":"a"}`}), say("ok")}}
	feed := inmem.NewFeed()
	user(feed, "m1", "plan")
	// The step after the model output is saved; the save after the task write fails.
	journal := &stepBudget{Journal: inmem.NewJournal(), n: 1}
	_, err := run(t, einorun.Request{Model: einorun.Model{New: m.factory}, Feed: feed, Journal: journal,
		Extensions: []einorun.Extension{einorun.Planning()}})
	if !errors.Is(err, errStepRefused) {
		t.Fatalf("err %v", err)
	}
	if m.calls() != 1 {
		t.Fatalf("model calls %d", m.calls())
	}
	for _, b := range journal.Resume().Blocks {
		if b.Call != nil && b.Call.Status == einorun.StatusSucceeded {
			t.Fatalf("call recorded as succeeded: %+v", b.Call)
		}
	}
}

func TestSubagentCallsCarryTheirModelCall(t *testing.T) {
	m := &scripted{steps: []step{
		call(invocation{"agent", `{"subagent_type":"general","prompt":"p","description":"d"}`}),
		call(invocation{"lookup", `{"x":"a"}`}),
		say("sub"),
		say("main"),
	}}
	feed := inmem.NewFeed()
	user(feed, "m1", "go")
	var seen string
	lookup := &fn{name: "lookup", run: func(ctx context.Context, _ string) (string, error) {
		c, _ := einorun.CallFrom(ctx)
		seen = c.ModelCallID
		return "ok", nil
	}}
	result, err := run(t, einorun.Request{Model: einorun.Model{New: m.factory}, Feed: feed,
		Tools: []einorun.ToolSpec{{Tool: lookup}}, Extensions: []einorun.Extension{einorun.Subagent(einorun.SubagentSpec{})}})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Calls) != 1 || result.Calls[0].ModelCallID != m.ids[1] || seen != m.ids[1] {
		t.Fatalf("calls %+v ids %v seen %s", result.Calls, m.ids, seen)
	}
	// Without side-effect tools the delegation may be repeated after an interruption.
	if d := result.Blocks[0].Call; !d.Replayable || d.SideEffects {
		t.Fatalf("delegation %+v", d)
	}
}

// cachedSkills returns the same slice on every List.
type cachedSkills struct{ matters []skill.FrontMatter }

func (c *cachedSkills) List(context.Context) ([]skill.FrontMatter, error) { return c.matters, nil }

func (c *cachedSkills) Get(_ context.Context, name string) (skill.Skill, error) {
	return skill.Skill{FrontMatter: c.matters[0], Content: "c"}, nil
}

func TestSkillBackendIsNotModified(t *testing.T) {
	backend := &cachedSkills{matters: []skill.FrontMatter{{Name: "tidy", Description: "d", Context: skill.ContextModeForkWithContext}}}
	m := &scripted{steps: []step{say("hi")}}
	feed := inmem.NewFeed()
	user(feed, "m1", "hi")
	if _, err := run(t, einorun.Request{Model: einorun.Model{New: m.factory}, Feed: feed,
		Extensions: []einorun.Extension{einorun.Skills(einorun.SkillsSpec{Backend: backend})}}); err != nil {
		t.Fatal(err)
	}
	if backend.matters[0].Context != skill.ContextModeForkWithContext {
		t.Fatalf("backend changed: %+v", backend.matters)
	}
}

// frozenJournal ignores every write once frozen, like a process that died.
type frozenJournal struct {
	*inmem.Journal
	frozen atomic.Bool
}

func (j *frozenJournal) SaveStep(ctx context.Context, step einorun.Step) error {
	if j.frozen.Load() {
		return nil
	}
	return j.Journal.SaveStep(ctx, step)
}

func (j *frozenJournal) SaveToolCall(ctx context.Context, call einorun.ToolCall) error {
	if j.frozen.Load() {
		return nil
	}
	return j.Journal.SaveToolCall(ctx, call)
}

func TestInterruptedDelegationWithSideEffectsNeedsReview(t *testing.T) {
	m := &scripted{steps: []step{
		call(invocation{"agent", `{"subagent_type":"general","prompt":"p","description":"d"}`}),
		call(invocation{"pay", `{"x":"1"}`}),
	}}
	feed := inmem.NewFeed()
	user(feed, "m1", "pay")
	journal := &frozenJournal{Journal: inmem.NewJournal()}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	pay := &fn{name: "pay", run: func(ctx context.Context, _ string) (string, error) {
		journal.frozen.Store(true)
		cancel()
		<-ctx.Done()
		return "", ctx.Err()
	}}
	request := einorun.Request{RunID: "r1", Model: einorun.Model{New: m.factory}, Feed: feed, Journal: journal,
		Tools:      []einorun.ToolSpec{{Tool: pay, SideEffects: true}},
		Extensions: []einorun.Extension{einorun.Subagent(einorun.SubagentSpec{})}}
	if _, err := einorun.New(einorun.Config{Language: llm.English}).Run(ctx, request); err == nil {
		t.Fatal("the run did not stop")
	}
	journal.frozen.Store(false)
	m.steps = []step{say("checked")}
	resume := journal.Resume()
	request.Resume = &resume
	result, err := run(t, request)
	if err != nil {
		t.Fatal(err)
	}
	if result.Blocks[0].Call.Status != einorun.StatusNeedsReview || len(result.Calls) != 1 || result.Calls[0].Status != einorun.StatusNeedsReview {
		t.Fatalf("delegation %+v calls %+v", result.Blocks[0].Call, result.Calls)
	}
}

func TestOffloadReadToolNameIsReserved(t *testing.T) {
	m := &scripted{steps: []step{say("hi")}}
	feed := inmem.NewFeed()
	user(feed, "m1", "hi")
	_, err := run(t, einorun.Request{Model: einorun.Model{New: m.factory}, Feed: feed,
		Tools: []einorun.ToolSpec{{Tool: echo(einorun.OffloadReadTool)}}})
	if err == nil || !strings.Contains(err.Error(), "registered twice") {
		t.Fatalf("err %v", err)
	}
}
