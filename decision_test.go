package einorun_test

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/cloudwego/eino/schema"

	"github.com/runforyou-ai/einorun"
	"github.com/runforyou-ai/einorun/inmem"
	"github.com/runforyou-ai/einorun/llm"
)

// confirmed returns a policy that pauses every call for a decision.
func confirmed(context.Context, einorun.CallView) (einorun.CallPolicy, error) {
	return einorun.CallPolicy{Confirm: &einorun.Confirmation{Payload: json.RawMessage(`{"why":"pay"}`)}}, nil
}

// decide writes a decision on the call with the provider call prefix.
func decide(t *testing.T, journal *inmem.Journal, prefix string, decision einorun.CallDecision) {
	t.Helper()
	for _, b := range journal.Resume().Blocks {
		if b.Call != nil && strings.HasPrefix(b.Call.CallID, prefix) {
			if err := journal.External(context.Background(), einorun.ToolCall{ID: b.Call.ID, Decision: &decision}); err != nil {
				t.Fatal(err)
			}
			return
		}
	}
	t.Fatalf("no call %s", prefix)
}

// resume runs request again from the journal.
func resume(t *testing.T, request einorun.Request, journal *inmem.Journal) einorun.Result {
	t.Helper()
	saved := journal.Resume()
	request.Resume = &saved
	result, err := run(t, request)
	if err != nil {
		t.Fatal(err)
	}
	return result
}

func TestConfirmPausesAndRunsOnApproval(t *testing.T) {
	m := &scripted{steps: []step{call(invocation{"pay", `{"x":"1"}`}), func(input []*schema.AgenticMessage) *schema.AgenticMessage {
		return say("done: " + lastUser(input))(input)
	}}}
	feed := inmem.NewFeed()
	user(feed, "m1", "pay")
	journal := inmem.NewJournal()
	var runs atomic.Int32
	pay := &fn{name: "pay", run: func(_ context.Context, args string) (string, error) {
		runs.Add(1)
		return "paid " + args, nil
	}}
	request := einorun.Request{RunID: "r1", Model: einorun.Model{New: m.factory}, Feed: feed, Journal: journal,
		Tools: []einorun.ToolSpec{{Tool: pay, SideEffects: true, Policy: confirmed}}}
	result, err := run(t, request)
	if err != nil {
		t.Fatal(err)
	}
	paused := result.Blocks[0].Call
	if !result.Suspended || paused.Status != einorun.StatusAwaitingDecision || paused.Handover != einorun.HandoverNone ||
		string(paused.Payload) != `{"why":"pay"}` || runs.Load() != 0 {
		t.Fatalf("result %+v call %+v", result, paused)
	}
	// Without a decision the run stays suspended.
	if again := resume(t, request, journal); !again.Suspended || runs.Load() != 0 {
		t.Fatalf("resumed without a decision: %+v", again)
	}
	decide(t, journal, "pay", einorun.CallDecision{Approved: true})
	result = resume(t, request, journal)
	if result.Text != `done: paid {"x":"1"}` || runs.Load() != 1 {
		t.Fatalf("text %q runs %d", result.Text, runs.Load())
	}
	done := result.Blocks[0].Call
	if done.Status != einorun.StatusSucceeded || done.Decision == nil || !done.Decision.Approved {
		t.Fatalf("call %+v", done)
	}
}

func TestConfirmRunsWithChangedArguments(t *testing.T) {
	m := &scripted{steps: []step{call(invocation{"pay", `{"x":"1"}`}), func(input []*schema.AgenticMessage) *schema.AgenticMessage {
		return say(lastUser(input))(input)
	}}}
	feed := inmem.NewFeed()
	user(feed, "m1", "pay")
	journal := inmem.NewJournal()
	var got string
	pay := &fn{name: "pay", run: func(_ context.Context, args string) (string, error) {
		got = args
		return "paid", nil
	}}
	request := einorun.Request{Model: einorun.Model{New: m.factory}, Feed: feed, Journal: journal,
		Tools: []einorun.ToolSpec{{Tool: pay, Policy: confirmed}}}
	if _, err := run(t, request); err != nil {
		t.Fatal(err)
	}
	decide(t, journal, "pay", einorun.CallDecision{Approved: true, Arguments: `{"x":"2"}`})
	result := resume(t, request, journal)
	if got != `{"x":"2"}` || !strings.Contains(result.Text, `{"x":"2"}`) || !strings.HasSuffix(result.Text, "paid") {
		t.Fatalf("args %s text %q", got, result.Text)
	}
	if call := result.Blocks[0].Call; call.Arguments != `{"x":"1"}` || call.Decision.Arguments != `{"x":"2"}` {
		t.Fatalf("call %+v", call)
	}
}

func TestConfirmRejectionGivesTheReason(t *testing.T) {
	m := &scripted{steps: []step{call(invocation{"pay", `{}`}), func(input []*schema.AgenticMessage) *schema.AgenticMessage {
		return say(lastUser(input))(input)
	}}}
	feed := inmem.NewFeed()
	user(feed, "m1", "pay")
	journal := inmem.NewJournal()
	pay := &fn{name: "pay", run: func(context.Context, string) (string, error) {
		t.Error("rejected call ran")
		return "", nil
	}}
	request := einorun.Request{Model: einorun.Model{New: m.factory}, Feed: feed, Journal: journal,
		Tools: []einorun.ToolSpec{{Tool: pay, Policy: confirmed}}}
	if _, err := run(t, request); err != nil {
		t.Fatal(err)
	}
	decide(t, journal, "pay", einorun.CallDecision{Reason: "too much"})
	result := resume(t, request, journal)
	if !strings.Contains(result.Text, "too much") || result.Blocks[0].Call.Status != einorun.StatusRejected {
		t.Fatalf("text %q call %+v", result.Text, result.Blocks[0].Call)
	}
}

func TestConfirmWaitsForEveryDecision(t *testing.T) {
	m := &scripted{steps: []step{call(invocation{"pay", `{"x":"a"}`}, invocation{"pay", `{"x":"b"}`}), func(input []*schema.AgenticMessage) *schema.AgenticMessage {
		return say(allResults(input))(input)
	}}}
	feed := inmem.NewFeed()
	user(feed, "m1", "pay")
	journal := inmem.NewJournal()
	request := einorun.Request{Model: einorun.Model{New: m.factory}, Feed: feed, Journal: journal,
		Tools: []einorun.ToolSpec{{Tool: echo("pay"), Policy: confirmed}}}
	if _, err := run(t, request); err != nil {
		t.Fatal(err)
	}
	blocks := journal.Resume().Blocks
	first := blocks[0].Call
	if err := journal.External(context.Background(), einorun.ToolCall{ID: first.ID, Decision: &einorun.CallDecision{Approved: true}}); err != nil {
		t.Fatal(err)
	}
	if again := resume(t, request, journal); !again.Suspended {
		t.Fatal("resumed with a decision missing")
	}
	second := blocks[1].Call
	if err := journal.External(context.Background(), einorun.ToolCall{ID: second.ID, Decision: &einorun.CallDecision{Reason: "no"}}); err != nil {
		t.Fatal(err)
	}
	result := resume(t, request, journal)
	if !strings.Contains(result.Text, `pay:{"x":"a"}`) || !strings.Contains(result.Text, "no") {
		t.Fatalf("text %q", result.Text)
	}
}

func TestSubagentCallsCannotPause(t *testing.T) {
	m := &scripted{steps: []step{
		call(invocation{"agent", `{"subagent_type":"general","prompt":"p","description":"d"}`}),
		call(invocation{"pay", `{}`}),
		func(input []*schema.AgenticMessage) *schema.AgenticMessage {
			return say("sub: " + lastUser(input))(input)
		},
		say("done"),
	}}
	feed := inmem.NewFeed()
	user(feed, "m1", "go")
	result, err := run(t, einorun.Request{Model: einorun.Model{New: m.factory}, Feed: feed,
		Tools:      []einorun.ToolSpec{{Tool: echo("pay"), Policy: confirmed}},
		Extensions: []einorun.Extension{einorun.Subagent(einorun.SubagentSpec{})}})
	if err != nil {
		t.Fatal(err)
	}
	if result.Suspended || len(result.Calls) != 1 || result.Calls[0].Status != einorun.StatusFailed {
		t.Fatalf("result %+v calls %+v", result, result.Calls)
	}
}

// refusingPause refuses every call paused for a decision.
type refusingPause struct{ *inmem.Journal }

func (j refusingPause) SaveToolCall(ctx context.Context, call einorun.ToolCall) error {
	if call.Status == einorun.StatusAwaitingDecision && call.Handover == einorun.HandoverNone {
		return einorun.RejectCall("nobody can confirm")
	}
	return j.Journal.SaveToolCall(ctx, call)
}

func TestHostCanRefuseAPause(t *testing.T) {
	m := &scripted{steps: []step{call(invocation{"pay", `{}`}), func(input []*schema.AgenticMessage) *schema.AgenticMessage {
		return say(lastUser(input))(input)
	}}}
	feed := inmem.NewFeed()
	user(feed, "m1", "pay")
	result, err := run(t, einorun.Request{Model: einorun.Model{New: m.factory}, Feed: feed, Journal: refusingPause{inmem.NewJournal()},
		Tools: []einorun.ToolSpec{{Tool: echo("pay"), Policy: confirmed}}})
	if err != nil {
		t.Fatal(err)
	}
	if result.Suspended || !strings.Contains(result.Text, "nobody can confirm") || result.Blocks[0].Call.Status != einorun.StatusFailed {
		t.Fatalf("result %+v", result)
	}
}

func TestApprovedCallInterruptedWhileRunningIsSettled(t *testing.T) {
	m := &scripted{steps: []step{call(invocation{"pay", `{}`})}}
	feed := inmem.NewFeed()
	user(feed, "m1", "pay")
	journal := inmem.NewJournal()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	pay := &fn{name: "pay", run: func(ctx context.Context, _ string) (string, error) {
		cancel()
		<-ctx.Done()
		return "", ctx.Err()
	}}
	request := einorun.Request{RunID: "r1", Model: einorun.Model{New: m.factory}, Feed: feed, Journal: journal,
		Tools: []einorun.ToolSpec{{Tool: pay, SideEffects: true, Policy: confirmed}}}
	if _, err := run(t, request); err != nil {
		t.Fatal(err)
	}
	decide(t, journal, "pay", einorun.CallDecision{Approved: true})
	saved := journal.Resume()
	request.Resume = &saved
	if _, err := einorun.New(einorun.Config{Language: llm.English}).Run(ctx, request); !errors.Is(err, context.Canceled) {
		t.Fatalf("err %v", err)
	}
	// The call failed when the run was cancelled; it does not run again.
	m.steps = []step{say("checked")}
	pay.run = func(context.Context, string) (string, error) {
		t.Error("cancelled call ran again")
		return "", nil
	}
	result := resume(t, request, journal)
	if result.Text != "checked" || result.Blocks[0].Call.Status != einorun.StatusFailed {
		t.Fatalf("result %+v call %+v", result, result.Blocks[0].Call)
	}
}

func TestApprovedLargeResultIsOffloaded(t *testing.T) {
	m := &scripted{steps: []step{call(invocation{"dump", `{}`}), func(input []*schema.AgenticMessage) *schema.AgenticMessage {
		return say(lastUser(input))(input)
	}}}
	feed := inmem.NewFeed()
	user(feed, "m1", "dump")
	journal := inmem.NewJournal()
	dump := &fn{name: "dump", run: func(context.Context, string) (string, error) { return strings.Repeat("x", 50000), nil }}
	request := einorun.Request{Model: einorun.Model{New: m.factory, ContextWindow: 4000}, Feed: feed, Journal: journal,
		Tools: []einorun.ToolSpec{{Tool: dump, Policy: confirmed}}}
	if _, err := run(t, request); err != nil {
		t.Fatal(err)
	}
	decide(t, journal, "dump", einorun.CallDecision{Approved: true})
	result := resume(t, request, journal)
	if !strings.Contains(result.Text, einorun.OffloadedPath("")) || len(result.Text) >= 50000 {
		t.Fatalf("result not offloaded: %d bytes", len(result.Text))
	}
}
