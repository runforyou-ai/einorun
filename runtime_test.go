package einorun_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/schema"

	"github.com/runforyou-ai/einorun"
	"github.com/runforyou-ai/einorun/inmem"
	"github.com/runforyou-ai/einorun/llm"
	"github.com/runforyou-ai/einorun/stream"
)

// step decides one model output from its input.
type step func(input []*schema.AgenticMessage) *schema.AgenticMessage

// scripted is a model that answers calls with its steps in order and records
// the inputs it saw.
type scripted struct {
	mu     sync.Mutex
	steps  []step
	inputs [][]*schema.AgenticMessage
	ids    []string
}

func (s *scripted) factory(context.Context, llm.ModelOptions) (model.AgenticModel, error) {
	return s, nil
}

func (s *scripted) next(ctx context.Context, input []*schema.AgenticMessage) (*schema.AgenticMessage, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.inputs = append(s.inputs, input)
	s.ids = append(s.ids, llm.ModelCallID(ctx))
	if len(s.steps) == 0 {
		return nil, errors.New("scripted model has no step left")
	}
	st := s.steps[0]
	s.steps = s.steps[1:]
	return st(input), nil
}

func (s *scripted) Generate(ctx context.Context, input []*schema.AgenticMessage, _ ...model.Option) (*schema.AgenticMessage, error) {
	return s.next(ctx, input)
}

func (s *scripted) Stream(ctx context.Context, input []*schema.AgenticMessage, _ ...model.Option) (*schema.StreamReader[*schema.AgenticMessage], error) {
	m, err := s.next(ctx, input)
	if err != nil {
		return nil, err
	}
	return schema.StreamReaderFromArray([]*schema.AgenticMessage{m}), nil
}

func (s *scripted) calls() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.inputs)
}

func say(text string) step {
	return func([]*schema.AgenticMessage) *schema.AgenticMessage {
		return &schema.AgenticMessage{Role: schema.AgenticRoleTypeAssistant,
			ContentBlocks: []*schema.ContentBlock{schema.NewContentBlock(&schema.AssistantGenText{Text: text})},
			ResponseMeta:  &schema.AgenticResponseMeta{TokenUsage: &schema.TokenUsage{PromptTokens: 10, CompletionTokens: 2}}}
	}
}

type invocation struct {
	name, args string
}

func call(calls ...invocation) step {
	return func([]*schema.AgenticMessage) *schema.AgenticMessage {
		m := &schema.AgenticMessage{Role: schema.AgenticRoleTypeAssistant,
			ResponseMeta: &schema.AgenticResponseMeta{TokenUsage: &schema.TokenUsage{PromptTokens: 10, CompletionTokens: 2}}}
		for i, c := range calls {
			m.ContentBlocks = append(m.ContentBlocks, schema.NewContentBlock(&schema.FunctionToolCall{
				CallID: fmt.Sprintf("%s-%d-%d", c.name, time.Now().UnixNano(), i), Name: c.name, Arguments: c.args}))
		}
		return m
	}
}

// fn is a tool backed by a function.
type fn struct {
	name string
	run  func(ctx context.Context, args string) (string, error)
}

func (f *fn) Info(context.Context) (*schema.ToolInfo, error) {
	return &schema.ToolInfo{Name: f.name, Desc: f.name, ParamsOneOf: schema.NewParamsOneOfByParams(map[string]*schema.ParameterInfo{
		"x": {Type: schema.String},
	})}, nil
}

func (f *fn) InvokableRun(ctx context.Context, args string, _ ...tool.Option) (string, error) {
	return f.run(ctx, args)
}

func echo(name string) *fn {
	return &fn{name: name, run: func(_ context.Context, args string) (string, error) { return name + ":" + args, nil }}
}

// lastText returns the text of the last user message the model saw.
func lastUser(input []*schema.AgenticMessage) string {
	for i := len(input) - 1; i >= 0; i-- {
		if input[i].Role == schema.AgenticRoleTypeUser {
			var parts []string
			for _, b := range input[i].ContentBlocks {
				switch {
				case b.UserInputText != nil:
					parts = append(parts, b.UserInputText.Text)
				case b.FunctionToolResult != nil:
					for _, c := range b.FunctionToolResult.Content {
						if c.Text != nil {
							parts = append(parts, c.Text.Text)
						}
					}
				}
			}
			return strings.Join(parts, "\n")
		}
	}
	return ""
}

// allResults returns the text of every tool result the model saw.
func allResults(input []*schema.AgenticMessage) string {
	var parts []string
	for _, m := range input {
		for _, b := range m.ContentBlocks {
			if b.FunctionToolResult != nil {
				for _, c := range b.FunctionToolResult.Content {
					if c.Text != nil {
						parts = append(parts, c.Text.Text)
					}
				}
			}
		}
	}
	return strings.Join(parts, "\n")
}

func user(feed *inmem.Feed, id, content string) int64 {
	return feed.Append(einorun.Message{ID: id, Revision: "1", Role: einorun.RoleUser, Content: content})
}

func run(t *testing.T, request einorun.Request) (einorun.Result, error) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	return einorun.New(einorun.Config{Language: llm.English}).Run(ctx, request)
}

func TestTextReply(t *testing.T) {
	m := &scripted{steps: []step{say("hello there")}}
	feed := inmem.NewFeed()
	seq := user(feed, "m1", "hi")
	var deltas []stream.Delta
	var mu sync.Mutex
	result, err := run(t, einorun.Request{RunID: "r1", Instruction: "be kind", Model: einorun.Model{New: m.factory}, Feed: feed,
		Stream: func(d stream.Delta) { mu.Lock(); deltas = append(deltas, d); mu.Unlock() }, StreamID: "s1"})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "hello there" || result.EndSeq != seq || result.Usage.Total != 12 {
		t.Fatalf("result %+v", result)
	}
	snapshot := stream.Snapshot{Stream: "s1"}
	for _, d := range deltas {
		if _, err := snapshot.Apply(d); err != nil {
			t.Fatal(err)
		}
	}
	if snapshot.Candidate != "hello there" {
		t.Fatalf("snapshot %+v", snapshot)
	}
	if m.ids[0] == "" {
		t.Fatal("model call without a ModelCallID")
	}
}

func TestToolCallIsRecorded(t *testing.T) {
	m := &scripted{steps: []step{call(invocation{"lookup", `{"x":"a"}`}), func(input []*schema.AgenticMessage) *schema.AgenticMessage {
		return say("found " + lastUser(input))(input)
	}}}
	feed := inmem.NewFeed()
	user(feed, "m1", "find a")
	journal := inmem.NewJournal()
	result, err := run(t, einorun.Request{RunID: "r1", Model: einorun.Model{New: m.factory}, Feed: feed, Journal: journal,
		Tools: []einorun.ToolSpec{{Tool: echo("lookup"), Replayable: true, Notes: map[string]string{"source": "test"}}}})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != `found lookup:{"x":"a"}` {
		t.Fatalf("text %q", result.Text)
	}
	if len(result.Blocks) != 1 || result.Blocks[0].Call == nil {
		t.Fatalf("blocks %+v", result.Blocks)
	}
	call := result.Blocks[0].Call
	if call.Status != einorun.StatusSucceeded || *call.Result != `lookup:{"x":"a"}` || !call.Replayable || call.Notes["source"] != "test" {
		t.Fatalf("call %+v", call)
	}
	stored, ok := journal.Call(call.ID)
	if !ok || stored.Status != einorun.StatusSucceeded || stored.Rev != call.Rev {
		t.Fatalf("stored %+v", stored)
	}
	resume := journal.Resume()
	if len(resume.Blocks) != 1 || len(resume.State) == 0 {
		t.Fatalf("resume %+v", resume)
	}
}

func TestToolErrorGoesToTheModel(t *testing.T) {
	failing := &fn{name: "broken", run: func(context.Context, string) (string, error) { return "", errors.New("disk full") }}
	m := &scripted{steps: []step{call(invocation{"broken", "{}"}), func(input []*schema.AgenticMessage) *schema.AgenticMessage {
		return say("saw " + lastUser(input))(input)
	}}}
	feed := inmem.NewFeed()
	user(feed, "m1", "go")
	result, err := run(t, einorun.Request{Model: einorun.Model{New: m.factory}, Feed: feed, Tools: []einorun.ToolSpec{{Tool: failing}}})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result.Text, "disk full") || result.Blocks[0].Call.Status != einorun.StatusFailed {
		t.Fatalf("result %+v", result)
	}
}

// completionTool returns its arguments as the completion value.
func completionTool(name string, fixed bool) *fn {
	return &fn{name: name, run: func(_ context.Context, args string) (string, error) {
		if strings.Contains(args, "bad") {
			return "", errors.New("message is required")
		}
		return "", einorun.Complete(json.RawMessage(args), fixed)
	}}
}

func TestCompletionEndsTheRun(t *testing.T) {
	m := &scripted{steps: []step{call(invocation{"ask", `{"x":"which one?"}`})}}
	feed := inmem.NewFeed()
	seq := user(feed, "m1", "help")
	journal := inmem.NewJournal()
	result, err := run(t, einorun.Request{Model: einorun.Model{New: m.factory}, Feed: feed, Journal: journal,
		Tools: []einorun.ToolSpec{{Tool: completionTool("ask", false), Completion: true}}})
	if err != nil {
		t.Fatal(err)
	}
	if result.Completion == nil || string(result.Completion.Value) != `{"x":"which one?"}` || result.Completion.Fixed || result.EndSeq != seq {
		t.Fatalf("result %+v", result)
	}
	if resume := journal.Resume(); resume.Completion == nil || resume.Completion.CallID != result.Blocks[0].Call.ID {
		t.Fatalf("active completion not saved: %+v", resume.Completion)
	}
	if m.calls() != 1 {
		t.Fatalf("model calls %d", m.calls())
	}
}

func TestCompletionCorrectionAndFallback(t *testing.T) {
	// The first invalid completion uses the only correction, the second ends
	// the run with the fallback.
	m := &scripted{steps: []step{call(invocation{"ask", `{"x":"bad"}`}), call(invocation{"ask", `{"x":"bad again"}`})}}
	feed := inmem.NewFeed()
	user(feed, "m1", "help")
	var reasons []string
	result, err := run(t, einorun.Request{Model: einorun.Model{New: m.factory}, Feed: feed,
		Tools: []einorun.ToolSpec{{Tool: completionTool("ask", false), Completion: true}},
		Completion: &einorun.CompletionPolicy{Fallback: func(reason string) json.RawMessage {
			reasons = append(reasons, reason)
			return json.RawMessage(`{"handoff":true}`)
		}}})
	if err != nil {
		t.Fatal(err)
	}
	if result.Completion == nil || result.Completion.Source != einorun.FromFallback || !result.Completion.Fixed ||
		len(reasons) != 1 || reasons[0] != einorun.ReasonCorrectionsExhausted {
		t.Fatalf("result %+v reasons %v", result.Completion, reasons)
	}
}

func TestCompletionBatchViolation(t *testing.T) {
	m := &scripted{steps: []step{
		call(invocation{"ask", `{"x":"a"}`}, invocation{"lookup", "{}"}),
		func(input []*schema.AgenticMessage) *schema.AgenticMessage {
			if !strings.Contains(lastUser(input), "on its own") {
				panic("the model did not see the batch violation: " + lastUser(input))
			}
			return call(invocation{"ask", `{"x":"b"}`})(input)
		},
	}}
	feed := inmem.NewFeed()
	user(feed, "m1", "help")
	lookups := 0
	lookup := &fn{name: "lookup", run: func(context.Context, string) (string, error) { lookups++; return "r", nil }}
	result, err := run(t, einorun.Request{Model: einorun.Model{New: m.factory}, Feed: feed,
		Tools: []einorun.ToolSpec{{Tool: completionTool("ask", false), Completion: true}, {Tool: lookup}}})
	if err != nil {
		t.Fatal(err)
	}
	if lookups != 0 || result.Completion == nil || string(result.Completion.Value) != `{"x":"b"}` {
		t.Fatalf("lookups %d result %+v", lookups, result.Completion)
	}
	for _, b := range result.Blocks[:2] {
		if b.Call.Status != einorun.StatusFailed {
			t.Fatalf("violating call %+v", b.Call)
		}
	}
}

func TestSubmission(t *testing.T) {
	m := &scripted{steps: []step{call(invocation{"pay", `{"x":"10"}`}), func(input []*schema.AgenticMessage) *schema.AgenticMessage {
		return say("told: " + lastUser(input))(input)
	}}}
	feed := inmem.NewFeed()
	user(feed, "m1", "pay")
	executed := false
	pay := &fn{name: "pay", run: func(context.Context, string) (string, error) { executed = true; return "paid", nil }}
	journal := inmem.NewJournal()
	result, err := run(t, einorun.Request{Model: einorun.Model{New: m.factory}, Feed: feed, Journal: journal,
		Tools: []einorun.ToolSpec{{Tool: pay, SideEffects: true, Policy: func(_ context.Context, c einorun.CallView) (einorun.CallPolicy, error) {
			return einorun.CallPolicy{Submit: &einorun.Submission{Receipt: "submitted for approval", Payload: json.RawMessage(`{"amount":10}`)}}, nil
		}}}})
	if err != nil {
		t.Fatal(err)
	}
	call := result.Blocks[0].Call
	if executed || result.Text != "told: submitted for approval" || call.Status != einorun.StatusAwaitingDecision ||
		call.Handover != einorun.HandoverSubmitted || string(call.Payload) != `{"amount":10}` {
		t.Fatalf("executed %v result %q call %+v", executed, result.Text, call)
	}
}

// rejecting refuses submissions.
type rejecting struct{ *inmem.Journal }

func (j rejecting) SaveToolCall(ctx context.Context, c einorun.ToolCall) error {
	if c.Handover == einorun.HandoverSubmitted {
		return einorun.RejectCall("nobody can approve this")
	}
	return j.Journal.SaveToolCall(ctx, c)
}

func TestRejectedSubmission(t *testing.T) {
	m := &scripted{steps: []step{call(invocation{"pay", "{}"}), func(input []*schema.AgenticMessage) *schema.AgenticMessage {
		return say(lastUser(input))(input)
	}}}
	feed := inmem.NewFeed()
	user(feed, "m1", "pay")
	result, err := run(t, einorun.Request{Model: einorun.Model{New: m.factory}, Feed: feed, Journal: rejecting{inmem.NewJournal()},
		Tools: []einorun.ToolSpec{{Tool: echo("pay"), Policy: func(context.Context, einorun.CallView) (einorun.CallPolicy, error) {
			return einorun.CallPolicy{Submit: &einorun.Submission{Receipt: "r"}}, nil
		}}}})
	if err != nil {
		t.Fatal(err)
	}
	call := result.Blocks[0].Call
	if !strings.Contains(result.Text, "nobody can approve this") || call.Status != einorun.StatusFailed || call.Handover != einorun.HandoverNone {
		t.Fatalf("text %q call %+v", result.Text, call)
	}
}

func TestDetached(t *testing.T) {
	m := &scripted{steps: []step{call(invocation{"delegate", "{}"}), func(input []*schema.AgenticMessage) *schema.AgenticMessage {
		return say(lastUser(input))(input)
	}}}
	feed := inmem.NewFeed()
	user(feed, "m1", "go")
	delegate := &fn{name: "delegate", run: func(context.Context, string) (string, error) {
		return "", einorun.Detached("handed to the local agent", json.RawMessage(`{"session":1}`))
	}}
	result, err := run(t, einorun.Request{Model: einorun.Model{New: m.factory}, Feed: feed, Tools: []einorun.ToolSpec{{Tool: delegate}}})
	if err != nil {
		t.Fatal(err)
	}
	call := result.Blocks[0].Call
	if result.Text != "handed to the local agent" || call.Handover != einorun.HandoverDetached || call.Status != einorun.StatusRunning {
		t.Fatalf("text %q call %+v", result.Text, call)
	}
}

func TestAwaitSuspendsAndResumes(t *testing.T) {
	m := &scripted{steps: []step{call(invocation{"dispatch", `{"x":"ls"}`}), func(input []*schema.AgenticMessage) *schema.AgenticMessage {
		return say("result: " + lastUser(input))(input)
	}}}
	feed := inmem.NewFeed()
	seq := user(feed, "m1", "list files")
	journal := inmem.NewJournal()
	dispatch := &fn{name: "dispatch", run: func(ctx context.Context, _ string) (string, error) {
		if !einorun.CanSuspend(ctx) {
			return "", errors.New("cannot suspend")
		}
		return "", einorun.Await(json.RawMessage(`{"computer":"c1"}`))
	}}
	request := einorun.Request{RunID: "r1", Model: einorun.Model{New: m.factory}, Feed: feed, Journal: journal,
		Tools: []einorun.ToolSpec{{Tool: dispatch, SideEffects: true}}}
	result, err := run(t, request)
	if err != nil {
		t.Fatal(err)
	}
	if !result.Suspended || len(result.Blocks) != 1 {
		t.Fatalf("result %+v", result)
	}
	call := result.Blocks[0].Call
	if call.Handover != einorun.HandoverAwait || call.Status != einorun.StatusWaiting {
		t.Fatalf("call %+v", call)
	}
	// Resuming before the result arrives keeps the run suspended.
	resume := journal.Resume()
	request.Resume = &resume
	if again, err := run(t, request); err != nil || !again.Suspended {
		t.Fatalf("early resume %+v %v", again, err)
	}
	// The external executor reports its result.
	output := "a.txt b.txt"
	if err := journal.External(context.Background(), einorun.ToolCall{ID: call.ID, Status: einorun.StatusSucceeded, Result: &output}); err != nil {
		t.Fatal(err)
	}
	resume = journal.Resume()
	request.Resume = &resume
	result, err = run(t, request)
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "result: a.txt b.txt" || result.EndSeq != seq {
		t.Fatalf("resumed result %+v", result)
	}
}

func TestInterruptedCallsAreSettledOnResume(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	m := &scripted{steps: []step{call(invocation{"write", "{}"}, invocation{"read", "{}"})}}
	feed := inmem.NewFeed()
	user(feed, "m1", "go")
	journal := inmem.NewJournal()
	block := &fn{name: "write", run: func(ctx context.Context, _ string) (string, error) {
		cancel()
		<-ctx.Done()
		return "", ctx.Err()
	}}
	request := einorun.Request{RunID: "r1", Model: einorun.Model{New: m.factory}, Feed: feed, Journal: journal,
		Tools: []einorun.ToolSpec{{Tool: block, SideEffects: true}, {Tool: echo("read"), Replayable: true}}}
	if _, err := einorun.New(einorun.Config{}).Run(ctx, request); err == nil {
		t.Fatal("canceled run returned no error")
	}
	// Simulate a crash in which the write call was still running.
	resume := journal.Resume()
	for i := range resume.Blocks {
		if c := resume.Blocks[i].Call; c != nil && c.Name == "write" {
			c.Status, c.Error, c.CompletedAt = einorun.StatusRunning, nil, nil
		}
	}
	m.steps = []step{func(input []*schema.AgenticMessage) *schema.AgenticMessage {
		return say("after: " + allResults(input))(input)
	}}
	request.Resume = &resume
	result, err := run(t, request)
	if err != nil {
		t.Fatal(err)
	}
	statuses := map[string]einorun.CallStatus{}
	for _, b := range result.Blocks {
		statuses[b.Call.Name] = b.Call.Status
	}
	if statuses["write"] != einorun.StatusNeedsReview {
		t.Fatalf("statuses %v", statuses)
	}
	if !strings.Contains(result.Text, "human review") {
		t.Fatalf("text %q", result.Text)
	}
}

func TestNewInputPreemptsAndDiscards(t *testing.T) {
	feed := inmem.NewFeed()
	user(feed, "m1", "first")
	started := make(chan struct{})
	release := make(chan struct{})
	slow := &fn{name: "slow", run: func(context.Context, string) (string, error) {
		close(started)
		<-release
		return "done", nil
	}}
	m := &scripted{steps: []step{call(invocation{"slow", "{}"}), func(input []*schema.AgenticMessage) *schema.AgenticMessage {
		return say("answer to: " + lastUser(input))(input)
	}}}
	go func() {
		<-started
		user(feed, "m2", "second")
		time.Sleep(50 * time.Millisecond)
		close(release)
	}()
	result, err := run(t, einorun.Request{Model: einorun.Model{New: m.factory}, Feed: feed, Tools: []einorun.ToolSpec{{Tool: slow}}})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "answer to: second" || result.EndSeq != 2 {
		t.Fatalf("result %q %d", result.Text, result.EndSeq)
	}
}

// guardOnce corrects the first candidate and notes the calls it saw.
type guardOnce struct {
	mu       sync.Mutex
	reviewed []string
}

func (g *guardOnce) Review(_ context.Context, turn einorun.TurnView) (einorun.Verdict, error) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.reviewed = append(g.reviewed, turn.Text)
	notes := map[string]map[string]string{}
	for _, c := range turn.Calls {
		notes[c.Call.ID] = map[string]string{"evidence": "true"}
	}
	if len(g.reviewed) == 1 {
		return einorun.Verdict{Kind: einorun.Correct, Prompt: "cite your source", Notes: notes}, nil
	}
	return einorun.Verdict{Kind: einorun.Accept}, nil
}

func TestGuardCorrects(t *testing.T) {
	m := &scripted{steps: []step{
		call(invocation{"lookup", "{}"}),
		say("unsupported"),
		func(input []*schema.AgenticMessage) *schema.AgenticMessage {
			if lastUser(input) != "cite your source" {
				panic("missing correction prompt: " + lastUser(input))
			}
			return say("supported")(input)
		},
	}}
	feed := inmem.NewFeed()
	user(feed, "m1", "q")
	g := &guardOnce{}
	journal := inmem.NewJournal()
	result, err := run(t, einorun.Request{Model: einorun.Model{New: m.factory}, Feed: feed, Journal: journal, Guard: g,
		Tools: []einorun.ToolSpec{{Tool: echo("lookup")}}})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "supported" || len(g.reviewed) != 2 {
		t.Fatalf("result %q reviewed %v", result.Text, g.reviewed)
	}
	stored, _ := journal.Call(result.Blocks[0].Call.ID)
	if stored.Notes["evidence"] != "true" {
		t.Fatalf("notes %v", stored.Notes)
	}
}

func TestGuardWithoutFallbackFails(t *testing.T) {
	m := &scripted{steps: []step{say("a"), say("b")}}
	feed := inmem.NewFeed()
	user(feed, "m1", "q")
	always := guardFunc(func(einorun.TurnView) einorun.Verdict { return einorun.Verdict{Kind: einorun.Correct, Prompt: "again"} })
	_, err := run(t, einorun.Request{Model: einorun.Model{New: m.factory}, Feed: feed, Guard: always})
	if !errors.Is(err, einorun.ErrGuardRejected) {
		t.Fatalf("err %v", err)
	}
}

type guardFunc func(einorun.TurnView) einorun.Verdict

func (f guardFunc) Review(_ context.Context, turn einorun.TurnView) (einorun.Verdict, error) {
	return f(turn), nil
}

func TestFixedCompletionIsDeliveredOnResume(t *testing.T) {
	m := &scripted{steps: []step{call(invocation{"handoff", `{"x":"human"}`})}}
	feed := inmem.NewFeed()
	user(feed, "m1", "help")
	journal := inmem.NewJournal()
	request := einorun.Request{Model: einorun.Model{New: m.factory}, Feed: feed, Journal: journal,
		Tools: []einorun.ToolSpec{{Tool: completionTool("handoff", true), Completion: true}}}
	if _, err := run(t, request); err != nil {
		t.Fatal(err)
	}
	// New input arrives; a restart delivers the fixed completion without
	// calling the model.
	user(feed, "m2", "hello?")
	resume := journal.Resume()
	request.Resume = &resume
	result, err := run(t, request)
	if err != nil {
		t.Fatal(err)
	}
	if result.Completion == nil || !result.Completion.Fixed || m.calls() != 1 {
		t.Fatalf("result %+v calls %d", result.Completion, m.calls())
	}
}

func TestEmptyOutputIsRetried(t *testing.T) {
	thinking := func([]*schema.AgenticMessage) *schema.AgenticMessage {
		return &schema.AgenticMessage{Role: schema.AgenticRoleTypeAssistant,
			ContentBlocks: []*schema.ContentBlock{schema.NewContentBlock(&schema.Reasoning{Text: "hmm"})}}
	}
	m := &scripted{steps: []step{thinking, say("ok")}}
	feed := inmem.NewFeed()
	user(feed, "m1", "q")
	result, err := run(t, einorun.Request{Model: einorun.Model{New: m.factory}, Feed: feed})
	if err != nil || result.Text != "ok" {
		t.Fatalf("result %+v %v", result, err)
	}
}

// counter is a stateful extension.
type counter struct{ value int }

func (c *counter) Name() string { return "counter" }
func (c *counter) OnClaim(context.Context, einorun.Claim) {
	c.value++
}
func (c *counter) Save() (json.RawMessage, error) { return json.Marshal(c.value) }
func (c *counter) Restore(data json.RawMessage) error {
	return json.Unmarshal(data, &c.value)
}

func TestStatefulExtension(t *testing.T) {
	m := &scripted{steps: []step{call(invocation{"dispatch", "{}"})}}
	feed := inmem.NewFeed()
	user(feed, "m1", "q")
	journal := inmem.NewJournal()
	dispatch := &fn{name: "dispatch", run: func(context.Context, string) (string, error) { return "", einorun.Await(nil) }}
	c := &counter{}
	request := einorun.Request{Model: einorun.Model{New: m.factory}, Feed: feed, Journal: journal,
		Tools: []einorun.ToolSpec{{Tool: dispatch}}, Extensions: []einorun.Extension{c}}
	if result, err := run(t, request); err != nil || !result.Suspended {
		t.Fatalf("%+v %v", result, err)
	}
	restored := &counter{}
	resume := journal.Resume()
	request.Resume, request.Extensions = &resume, []einorun.Extension{restored}
	if _, err := run(t, request); err != nil {
		t.Fatal(err)
	}
	if restored.value != 1 {
		t.Fatalf("restored %d", restored.value)
	}
	// A checkpoint without the state of a stateful extension is refused.
	request.Extensions = []einorun.Extension{&counter{}, &stateful{named{"other"}}}
	if _, err := run(t, request); err == nil || !strings.Contains(err.Error(), "no state for extension other") {
		t.Fatalf("missing state: %v", err)
	}
}

// stateful is a stateful extension with nothing to keep.
type stateful struct{ named }

func (s *stateful) Save() (json.RawMessage, error) { return json.RawMessage(`null`), nil }
func (s *stateful) Restore(json.RawMessage) error  { return nil }

// named is an extension without state.
type named struct{ name string }

func (n *named) Name() string { return n.name }

func TestDuplicateExtensionNames(t *testing.T) {
	m := &scripted{steps: []step{say("x")}}
	feed := inmem.NewFeed()
	user(feed, "m1", "q")
	_, err := run(t, einorun.Request{Model: einorun.Model{New: m.factory}, Feed: feed, Extensions: []einorun.Extension{&named{"a"}, &named{"a"}}})
	if err == nil {
		t.Fatal("duplicate names accepted")
	}
}

// partialFeed claims at most up to cap at first.
type partialFeed struct {
	*inmem.Feed
	mu   sync.Mutex
	caps []int64
}

func (f *partialFeed) Claim(ctx context.Context, through int64) (einorun.Claim, error) {
	f.mu.Lock()
	if len(f.caps) > 0 {
		through = min(through, f.caps[0])
		f.caps = f.caps[1:]
	}
	f.mu.Unlock()
	return f.Feed.Claim(ctx, through)
}

func TestPartialClaimContinues(t *testing.T) {
	feed := &partialFeed{Feed: inmem.NewFeed(), caps: []int64{1}}
	user(feed.Feed, "m1", "one")
	user(feed.Feed, "m2", "two")
	m := &scripted{steps: []step{
		func(input []*schema.AgenticMessage) *schema.AgenticMessage {
			return say("re " + lastUser(input))(input)
		},
		func(input []*schema.AgenticMessage) *schema.AgenticMessage {
			return say("re " + lastUser(input))(input)
		},
	}}
	result, err := run(t, einorun.Request{Model: einorun.Model{New: m.factory}, Feed: feed})
	if err != nil {
		t.Fatal(err)
	}
	if result.EndSeq != 2 || result.Text != "re two" {
		t.Fatalf("result %q %d", result.Text, result.EndSeq)
	}
}

func TestDefaultJournalResumesTwice(t *testing.T) {
	m := &scripted{steps: []step{call(invocation{"dispatch", "{}"})}}
	feed := inmem.NewFeed()
	user(feed, "m1", "q")
	dispatch := &fn{name: "dispatch", run: func(context.Context, string) (string, error) { return "", einorun.Await(nil) }}
	request := einorun.Request{Model: einorun.Model{New: m.factory}, Feed: feed, Tools: []einorun.ToolSpec{{Tool: dispatch}}}
	result, err := run(t, request)
	if err != nil || !result.Suspended || result.Resume == nil {
		t.Fatalf("%+v %v", result, err)
	}
	for range 2 {
		request.Resume = result.Resume
		if result, err = run(t, request); err != nil || !result.Suspended || result.Resume == nil || len(result.Resume.State) == 0 {
			t.Fatalf("resume %+v %v", result, err)
		}
	}
}

func TestToolDeadlineEndsTheRun(t *testing.T) {
	m := &scripted{steps: []step{call(invocation{"slow", "{}"})}}
	feed := inmem.NewFeed()
	user(feed, "m1", "q")
	slow := &fn{name: "slow", run: func(context.Context, string) (string, error) {
		return "", fmt.Errorf("fetch: %w", context.DeadlineExceeded)
	}}
	_, err := run(t, einorun.Request{Model: einorun.Model{New: m.factory}, Feed: feed, Tools: []einorun.ToolSpec{{Tool: slow}}})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err %v", err)
	}
}

func TestUnknownExtensionStateIsRefused(t *testing.T) {
	m := &scripted{steps: []step{call(invocation{"dispatch", "{}"})}}
	feed := inmem.NewFeed()
	user(feed, "m1", "q")
	journal := inmem.NewJournal()
	dispatch := &fn{name: "dispatch", run: func(context.Context, string) (string, error) { return "", einorun.Await(nil) }}
	request := einorun.Request{Model: einorun.Model{New: m.factory}, Feed: feed, Journal: journal,
		Tools: []einorun.ToolSpec{{Tool: dispatch}}, Extensions: []einorun.Extension{&counter{}}}
	if _, err := run(t, request); err != nil {
		t.Fatal(err)
	}
	resume := journal.Resume()
	request.Resume, request.Extensions = &resume, nil
	if _, err := run(t, request); err == nil || !strings.Contains(err.Error(), "not registered") {
		t.Fatalf("err %v", err)
	}
}

func TestSupersededCompletionLeavesTheHistory(t *testing.T) {
	feed := inmem.NewFeed()
	user(feed, "m1", "first")
	m := &scripted{}
	m.steps = []step{
		func(input []*schema.AgenticMessage) *schema.AgenticMessage {
			// New input arrives while the completion call is made.
			user(feed, "m2", "second")
			return call(invocation{"ask", `{"x":"which?"}`})(input)
		},
		func(input []*schema.AgenticMessage) *schema.AgenticMessage {
			for _, msg := range input {
				for _, b := range msg.ContentBlocks {
					if b.FunctionToolCall != nil && b.FunctionToolCall.Name == "ask" {
						panic("the superseded completion is still in the history")
					}
				}
			}
			return call(invocation{"ask", `{"x":"answer to second"}`})(input)
		},
	}
	result, err := run(t, einorun.Request{Model: einorun.Model{New: m.factory}, Feed: feed,
		Tools: []einorun.ToolSpec{{Tool: completionTool("ask", false), Completion: true}}})
	if err != nil {
		t.Fatal(err)
	}
	if result.Completion == nil || string(result.Completion.Value) != `{"x":"answer to second"}` || result.EndSeq != 2 {
		t.Fatalf("result %+v", result.Completion)
	}
}

func TestResumedTurnWithNewInputGetsAFullBudget(t *testing.T) {
	m := &scripted{steps: []step{call(invocation{"dispatch", "{}"})}}
	feed := inmem.NewFeed()
	user(feed, "m1", "q")
	journal := inmem.NewJournal()
	dispatch := &fn{name: "dispatch", run: func(context.Context, string) (string, error) { return "", einorun.Await(nil) }}
	request := einorun.Request{Model: einorun.Model{New: m.factory}, Feed: feed, Journal: journal,
		Tools: []einorun.ToolSpec{{Tool: dispatch}, {Tool: echo("lookup")}}, Limits: einorun.Limits{MaxIterations: 2}}
	result, err := run(t, request)
	if err != nil || !result.Suspended {
		t.Fatalf("%+v %v", result, err)
	}
	output := "done"
	if err := journal.External(context.Background(), einorun.ToolCall{ID: result.Blocks[0].Call.ID, Status: einorun.StatusSucceeded, Result: &output}); err != nil {
		t.Fatal(err)
	}
	user(feed, "m2", "more")
	// One model call of the restored turn was spent; the new input must still
	// be able to call a tool before the budget ends.
	m.steps = []step{
		func(input []*schema.AgenticMessage) *schema.AgenticMessage {
			if len(input) > 0 && strings.Contains(lastUser(input), "limit") {
				panic("the new turn started at the end of the budget")
			}
			return call(invocation{"lookup", "{}"})(input)
		},
		say("ok"),
	}
	resume := journal.Resume()
	request.Resume = &resume
	result, err = run(t, request)
	if err != nil || result.Text != "ok" {
		t.Fatalf("%+v %v", result, err)
	}
}

// instructed contributes an instruction from its state.
type instructed struct{ counter }

func (i *instructed) Instruction(einorun.AgentScope) string {
	return fmt.Sprintf("claims so far: %d", i.value)
}

func TestInstructionSeesRestoredState(t *testing.T) {
	m := &scripted{steps: []step{call(invocation{"dispatch", "{}"})}}
	feed := inmem.NewFeed()
	user(feed, "m1", "q")
	journal := inmem.NewJournal()
	dispatch := &fn{name: "dispatch", run: func(context.Context, string) (string, error) { return "", einorun.Await(nil) }}
	request := einorun.Request{Model: einorun.Model{New: m.factory}, Feed: feed, Journal: journal, Instruction: "base",
		Tools: []einorun.ToolSpec{{Tool: dispatch}}, Extensions: []einorun.Extension{&instructed{}}}
	if _, err := run(t, request); err != nil {
		t.Fatal(err)
	}
	output := "x"
	resume := journal.Resume()
	if err := journal.External(context.Background(), einorun.ToolCall{ID: resume.Blocks[0].Call.ID, Status: einorun.StatusSucceeded, Result: &output}); err != nil {
		t.Fatal(err)
	}
	resume = journal.Resume()
	var system string
	m.steps = []step{func(input []*schema.AgenticMessage) *schema.AgenticMessage {
		for _, msg := range input {
			if msg.Role == schema.AgenticRoleTypeSystem {
				system = llmText(msg)
			}
		}
		return say("ok")(input)
	}}
	request.Resume, request.Extensions = &resume, []einorun.Extension{&instructed{}}
	if _, err := run(t, request); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(system, "claims so far: 1") {
		t.Fatalf("system %q", system)
	}
}

// llmText returns the text blocks of any message.
func llmText(m *schema.AgenticMessage) string {
	var parts []string
	for _, b := range m.ContentBlocks {
		if b.UserInputText != nil {
			parts = append(parts, b.UserInputText.Text)
		}
		if b.AssistantGenText != nil {
			parts = append(parts, b.AssistantGenText.Text)
		}
	}
	return strings.Join(parts, "")
}

func TestCorrectionSurvivesACrashAfterTheBatch(t *testing.T) {
	// An invalid completion uses the only correction; the run then suspends on
	// an awaited call in the next batch. After resuming, another invalid
	// completion must end with the fallback, not get a second correction.
	m := &scripted{steps: []step{call(invocation{"ask", `{"x":"bad"}`}), call(invocation{"dispatch", "{}"})}}
	feed := inmem.NewFeed()
	user(feed, "m1", "q")
	journal := inmem.NewJournal()
	dispatch := &fn{name: "dispatch", run: func(context.Context, string) (string, error) { return "", einorun.Await(nil) }}
	request := einorun.Request{Model: einorun.Model{New: m.factory}, Feed: feed, Journal: journal,
		Tools:      []einorun.ToolSpec{{Tool: completionTool("ask", false), Completion: true}, {Tool: dispatch}},
		Completion: &einorun.CompletionPolicy{Fallback: func(reason string) json.RawMessage { return json.RawMessage(`"` + reason + `"`) }}}
	if result, err := run(t, request); err != nil || !result.Suspended {
		t.Fatalf("%+v %v", result, err)
	}
	resume := journal.Resume()
	var dispatched string
	for _, b := range resume.Blocks {
		if b.Call != nil && b.Call.Name == "dispatch" {
			dispatched = b.Call.ID
		}
	}
	output := "x"
	if err := journal.External(context.Background(), einorun.ToolCall{ID: dispatched, Status: einorun.StatusSucceeded, Result: &output}); err != nil {
		t.Fatal(err)
	}
	resume = journal.Resume()
	request.Resume = &resume
	m.steps = []step{call(invocation{"ask", `{"x":"bad again"}`})}
	result, err := run(t, request)
	if err != nil || result.Completion == nil || result.Completion.Source != einorun.FromFallback {
		t.Fatalf("%+v %v", result.Completion, err)
	}
}
