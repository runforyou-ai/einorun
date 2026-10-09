package einorun_test

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cloudwego/eino/schema"

	"github.com/runforyou-ai/einorun"
	"github.com/runforyou-ai/einorun/inmem"
	"github.com/runforyou-ai/einorun/llm"
)

// scopeRecorder records the agent scopes extensions and tools see.
type scopeRecorder struct {
	mu      sync.Mutex
	outputs []string
	views   map[string]string // provider call ID -> agent ID
}

func (s *scopeRecorder) Name() string { return "scopes" }

func (s *scopeRecorder) AfterModelOutput(_ context.Context, scope einorun.AgentScope, _ *schema.AgenticMessage) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.outputs = append(s.outputs, scope.ID)
}

// resultsOf returns the tool results in input.
func resultsOf(input []*schema.AgenticMessage) []string {
	var results []string
	for _, m := range input {
		for _, b := range m.ContentBlocks {
			if b.FunctionToolResult != nil {
				results = append(results, b.FunctionToolResult.CallID)
			}
		}
	}
	return results
}

func TestParallelSubagentsHaveTheirOwnScope(t *testing.T) {
	m := &scripted{steps: []step{
		call(invocation{"agent", `{"subagent_type":"general","prompt":"p1","description":"d1"}`},
			invocation{"agent", `{"subagent_type":"general","prompt":"p2","description":"d2"}`}),
	}}
	// Each sub-agent looks up once and then answers, in whatever order they
	// run.
	subStep := func(input []*schema.AgenticMessage) *schema.AgenticMessage {
		if len(resultsOf(input)) > 0 {
			return say("sub")(input)
		}
		return call(invocation{"lookup", `{"x":"a"}`})(input)
	}
	m.steps = append(m.steps, subStep, subStep, subStep, subStep)
	m.steps = append(m.steps, say("done"))
	feed := inmem.NewFeed()
	user(feed, "m1", "go")
	scopes := &scopeRecorder{views: map[string]string{}}
	lookup := &fn{name: "lookup", run: func(context.Context, string) (string, error) { return "ok", nil }}
	result, err := run(t, einorun.Request{Model: einorun.Model{New: m.factory}, Feed: feed,
		Tools: []einorun.ToolSpec{{Tool: lookup, Policy: func(_ context.Context, call einorun.CallView) (einorun.CallPolicy, error) {
			scopes.mu.Lock()
			scopes.views[call.CallID] = call.Agent.ID
			scopes.mu.Unlock()
			return einorun.CallPolicy{}, nil
		}}},
		Extensions: []einorun.Extension{scopes, einorun.Subagent(einorun.SubagentSpec{})}})
	if err != nil {
		t.Fatal(err)
	}
	delegations := map[string]bool{}
	for _, b := range result.Blocks {
		if b.Call != nil {
			delegations[b.Call.ID] = true
		}
	}
	if len(delegations) != 2 || len(result.Calls) != 2 {
		t.Fatalf("blocks %+v calls %+v", result.Blocks, result.Calls)
	}
	for _, c := range result.Calls {
		if scopes.views[c.CallID] != c.ParentID || !delegations[c.ParentID] {
			t.Fatalf("call %s seen by agent %q, parent %s", c.CallID, scopes.views[c.CallID], c.ParentID)
		}
	}
	if result.Calls[0].ParentID == result.Calls[1].ParentID {
		t.Fatal("parallel sub-agents share an ID")
	}
	// The main agent's outputs and each sub-agent's outputs come with their
	// own scope.
	byAgent := map[string]int{}
	for _, id := range scopes.outputs {
		byAgent[id]++
	}
	if byAgent[einorun.MainAgentID] != 2 || byAgent[result.Calls[0].ParentID] != 2 || byAgent[result.Calls[1].ParentID] != 2 {
		t.Fatalf("outputs by agent %v", byAgent)
	}
}

func TestBuiltinToolsTakeNotesAndPolicy(t *testing.T) {
	m := &scripted{steps: []step{
		call(invocation{"agent", `{"subagent_type":"general","prompt":"p","description":"d"}`}),
		func(input []*schema.AgenticMessage) *schema.AgenticMessage {
			return say("main: " + lastUser(input))(input)
		},
	}}
	feed := inmem.NewFeed()
	user(feed, "m1", "go")
	result, err := run(t, einorun.Request{Model: einorun.Model{New: m.factory}, Feed: feed,
		Extensions: []einorun.Extension{einorun.Subagent(einorun.SubagentSpec{})},
		BuiltinTools: map[string]einorun.BuiltinTool{
			"agent": {Notes: map[string]string{"source": "delegation"}, Policy: func(context.Context, einorun.CallView) (einorun.CallPolicy, error) {
				return einorun.CallPolicy{Submit: &einorun.Submission{Receipt: "submitted for approval"}}, nil
			}},
			einorun.OffloadReadTool: {Notes: map[string]string{"source": "builtin"}},
		}})
	if err != nil {
		t.Fatal(err)
	}
	delegation := result.Blocks[0].Call
	if delegation.Notes["source"] != "delegation" || delegation.Handover != einorun.HandoverSubmitted || len(result.Calls) != 0 {
		t.Fatalf("delegation %+v", delegation)
	}
	if !strings.Contains(result.Text, "submitted for approval") {
		t.Fatalf("text %q", result.Text)
	}
}

func TestUnknownBuiltinToolIsRefused(t *testing.T) {
	m := &scripted{steps: []step{say("hi")}}
	feed := inmem.NewFeed()
	user(feed, "m1", "hi")
	_, err := run(t, einorun.Request{Model: einorun.Model{New: m.factory}, Feed: feed,
		Tools:        []einorun.ToolSpec{{Tool: echo("lookup")}},
		BuiltinTools: map[string]einorun.BuiltinTool{"lookup": {}}})
	if err == nil || !strings.Contains(err.Error(), "not a built-in tool") || !strings.Contains(err.Error(), einorun.OffloadReadTool) {
		t.Fatalf("err %v", err)
	}
	// Task list tools exist only with the Planning extension.
	_, err = run(t, einorun.Request{Model: einorun.Model{New: m.factory}, Feed: feed,
		BuiltinTools: map[string]einorun.BuiltinTool{"TaskCreate": {}}})
	if err == nil || !strings.Contains(err.Error(), "not a built-in tool") {
		t.Fatalf("err %v", err)
	}
}

func TestTextOverrides(t *testing.T) {
	m := &scripted{steps: []step{
		call(invocation{"ask", `{"x":"a"}`}, invocation{"lookup", "{}"}),
		func(input []*schema.AgenticMessage) *schema.AgenticMessage {
			if !strings.Contains(lastUser(input), "call ask alone, please") {
				panic("the model did not see the overridden text: " + lastUser(input))
			}
			return call(invocation{"ask", `{"x":"b"}`})(input)
		},
	}}
	feed := inmem.NewFeed()
	user(feed, "m1", "help")
	runtime := einorun.New(einorun.Config{Language: llm.English, Text: einorun.Text{BatchMixed: "call %s alone, please"}})
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	if _, err := runtime.Run(ctx, einorun.Request{Model: einorun.Model{New: m.factory}, Feed: feed,
		Tools: []einorun.ToolSpec{{Tool: completionTool("ask", false), Completion: true}, {Tool: echo("lookup")}}}); err != nil {
		t.Fatal(err)
	}
	if text := runtime.Text(); text.BatchMixed != "call %s alone, please" || text.BatchTooMany != einorun.DefaultText(llm.English).BatchTooMany {
		t.Fatalf("text %+v", text)
	}
}

func TestInterruptedOutcome(t *testing.T) {
	runtime := einorun.New(einorun.Config{Language: llm.Chinese})
	zh := einorun.DefaultText(llm.Chinese)
	for _, c := range []struct {
		replayable, sideEffects bool
		status                  einorun.CallStatus
		text                    string
	}{
		{true, false, einorun.StatusInterrupted, zh.InterruptedReplayable},
		{false, false, einorun.StatusInterrupted, zh.Interrupted},
		{false, true, einorun.StatusNeedsReview, zh.NeedsReview},
		{true, true, einorun.StatusInterrupted, zh.InterruptedReplayable},
	} {
		status, text := runtime.InterruptedOutcome(c.replayable, c.sideEffects)
		if status != c.status || text != c.text {
			t.Fatalf("%+v: %s %q", c, status, text)
		}
	}
	// Overridden text is used too.
	custom := einorun.New(einorun.Config{Language: llm.Chinese, Text: einorun.Text{NeedsReview: "请人工核对"}})
	if status, text := custom.InterruptedOutcome(false, true); status != einorun.StatusNeedsReview || text != "请人工核对" {
		t.Fatalf("overridden %s %q", status, text)
	}
}
