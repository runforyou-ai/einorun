// Package journaltest checks implementations of einorun.Journal and
// einorun.Feed against their contracts. Hosts run these suites against their
// persistent implementations in their own tests.
package journaltest

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"sync"
	"testing"
	"time"

	"github.com/runforyou-ai/einorun"
)

// JournalHarness gives the suite access to one empty run in the
// implementation under test.
type JournalHarness struct {
	Journal einorun.Journal
	// Load returns what the host would hand back to resume the run: main
	// calls attached to their blocks, sub-agent calls in Resume.Calls.
	Load func(ctx context.Context) (einorun.Resume, error)
	// External applies a write the host makes outside the runtime, with
	// einorun.OverlayExternal semantics: Status, Result, Error, Media,
	// CompletedAt, Handover and Payload when set. The suite only uses it on
	// calls that exist.
	External func(ctx context.Context, update einorun.ToolCall) error
	// Usage, when set, returns the usage of the last saved step.
	Usage func(ctx context.Context) (einorun.Usage, error)
}

// RunJournal runs the Journal contract suite. newHarness returns a harness for
// a fresh run each time it is called.
func RunJournal(t *testing.T, newHarness func(t *testing.T) JournalHarness) {
	for _, c := range []struct {
		name string
		run  func(*testing.T, JournalHarness)
	}{
		{"newer revisions win", testNewerRevisionsWin},
		{"repeated writes are stable", testRepeatedWrite},
		{"stale step snapshots lose", testStaleStep},
		{"settled outcomes are final", testSettledFinal},
		{"handover by the runtime", testRuntimeHandover},
		{"handover by the host first", testHostHandoverFirst},
		{"external result before handover", testExternalResultFirst},
		{"stale writes do not fill a handover", testStaleHandoverFill},
		{"non-handover writes do not fill a receipt", testNonHandoverFill},
		{"notes follow revisions", testNotes},
		{"steps", testSteps},
		{"blocks link to merged calls", testBlockLinks},
		{"concurrent writes to one call", testConcurrentOneCall},
		{"invalid calls are refused", testInvalid},
	} {
		t.Run(c.name, func(t *testing.T) { c.run(t, newHarness(t)) })
	}
}

// FeedHarness gives the suite access to one empty conversation in the
// implementation under test.
type FeedHarness struct {
	Feed einorun.Feed
	// Append adds a message and returns its sequence number.
	Append func(ctx context.Context, message einorun.Message) (int64, error)
	// Consume marks input up to seq as consumed.
	Consume func(ctx context.Context, seq int64) error
}

// RunFeed runs the Feed contract suite.
func RunFeed(t *testing.T, newHarness func(t *testing.T) FeedHarness) {
	for _, c := range []struct {
		name string
		run  func(*testing.T, FeedHarness)
	}{
		{"pending and claim", testPendingClaim},
		{"messages round-trip", testMessageRoundTrip},
		{"replay", testReplay},
		{"consumed input", testConsumed},
		{"watch", testWatch},
	} {
		t.Run(c.name, func(t *testing.T) { c.run(t, newHarness(t)) })
	}
}

var at = time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

// child returns a sub-agent call snapshot; sub-agent calls come back in
// Resume.Calls without needing a block.
func child(id string, rev uint64, status einorun.CallStatus) einorun.ToolCall {
	return einorun.ToolCall{ID: id, ParentID: "parent", CallID: "c-" + id, Name: "tool", Arguments: "{}", Rev: rev, Status: status, StartedAt: &at}
}

func text(s string) *string { return &s }

// load returns the resume of h.
func load(t *testing.T, h JournalHarness) einorun.Resume {
	t.Helper()
	resume, err := h.Load(context.Background())
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	return resume
}

// find returns the record of the call "a", which every journal test uses.
func find(t *testing.T, h JournalHarness) einorun.ToolCall {
	const id = "a"
	t.Helper()
	resume := load(t, h)
	for _, block := range resume.Blocks {
		if block.Call != nil && block.Call.ID == id {
			return *block.Call
		}
	}
	for _, c := range resume.Calls {
		if c.ID == id {
			return c
		}
	}
	t.Fatalf("call %s not found", id)
	return einorun.ToolCall{}
}

func save(t *testing.T, h JournalHarness, c einorun.ToolCall) {
	t.Helper()
	if err := h.Journal.SaveToolCall(context.Background(), c); err != nil {
		t.Fatalf("save %s rev %d: %v", c.ID, c.Rev, err)
	}
}

func external(t *testing.T, h JournalHarness, update einorun.ToolCall) {
	t.Helper()
	if err := h.External(context.Background(), update); err != nil {
		t.Fatalf("external %s: %v", update.ID, err)
	}
}

func str(p *string) string {
	if p == nil {
		return "<nil>"
	}
	return *p
}

func testNewerRevisionsWin(t *testing.T, h JournalHarness) {
	running := child("a", 2, einorun.StatusRunning)
	running.Arguments = `{"x":2}`
	queued := child("a", 1, einorun.StatusQueued)
	save(t, h, running)
	save(t, h, queued) // arrives late
	got := find(t, h)
	if got.Status != einorun.StatusRunning || got.Rev != 2 || got.Arguments != `{"x":2}` {
		t.Fatalf("got %s rev %d %s, want running rev 2", got.Status, got.Rev, got.Arguments)
	}
}

func testRepeatedWrite(t *testing.T, h JournalHarness) {
	done := child("a", 3, einorun.StatusSucceeded)
	done.Result, done.CompletedAt, done.Notes = text("ok"), &at, map[string]string{"k": "v"}
	save(t, h, done)
	save(t, h, done)
	got := find(t, h)
	if got.Status != einorun.StatusSucceeded || str(got.Result) != "ok" || got.Rev != 3 || got.Notes["k"] != "v" {
		t.Fatalf("repeated write changed the record: %+v", got)
	}
}

func testStaleStep(t *testing.T, h JournalHarness) {
	ctx := context.Background()
	save(t, h, child("a", 1, einorun.StatusQueued))
	done := child("a", 3, einorun.StatusSucceeded)
	done.Result, done.CompletedAt = text("ok"), &at
	save(t, h, done)
	stale := child("a", 2, einorun.StatusRunning)
	if err := h.Journal.SaveStep(ctx, einorun.Step{Changes: einorun.Changes{Calls: []einorun.ToolCall{stale}}, State: []byte("s")}); err != nil {
		t.Fatal(err)
	}
	if got := find(t, h); got.Status != einorun.StatusSucceeded || str(got.Result) != "ok" {
		t.Fatalf("stale snapshot won: %+v", got)
	}
}

func testSettledFinal(t *testing.T, h JournalHarness) {
	failed := child("a", 1, einorun.StatusFailed)
	failed.Error, failed.CompletedAt = text("boom"), &at
	save(t, h, failed)
	later := at.Add(time.Hour)
	retry := child("a", 2, einorun.StatusSucceeded)
	retry.Result, retry.CompletedAt, retry.Payload = text("ok"), &later, json.RawMessage(`{}`)
	retry.Handover = einorun.HandoverAwait
	save(t, h, retry)
	got := find(t, h)
	if got.Status != einorun.StatusFailed || str(got.Error) != "boom" || got.Result != nil || !got.CompletedAt.Equal(at) ||
		len(got.Payload) != 0 || got.Handover != einorun.HandoverNone {
		t.Fatalf("settled outcome changed: %+v", got)
	}
}

func testRuntimeHandover(t *testing.T, h JournalHarness) {
	save(t, h, child("a", 1, einorun.StatusRunning))
	detached := child("a", 2, einorun.StatusRunning)
	detached.Handover, detached.Result, detached.Payload = einorun.HandoverDetached, text("receipt"), json.RawMessage(`{"session":"s1"}`)
	save(t, h, detached)
	late := child("a", 3, einorun.StatusInterrupted)
	late.Result = text("interrupted")
	save(t, h, late)
	got := find(t, h)
	if got.Handover != einorun.HandoverDetached || str(got.Result) != "receipt" || got.Status != einorun.StatusRunning {
		t.Fatalf("handover not kept: %+v", got)
	}
	if string(got.Payload) != `{"session":"s1"}` {
		t.Fatalf("payload %s", got.Payload)
	}
}

func testHostHandoverFirst(t *testing.T, h JournalHarness) {
	save(t, h, child("a", 1, einorun.StatusRunning))
	external(t, h, einorun.ToolCall{ID: "a", Status: einorun.StatusQueued, Handover: einorun.HandoverDetached})
	detached := child("a", 2, einorun.StatusRunning)
	detached.Handover, detached.Result, detached.Payload = einorun.HandoverDetached, text("receipt"), json.RawMessage(`{"op":1}`)
	save(t, h, detached)
	got := find(t, h)
	if got.Status != einorun.StatusQueued || got.Handover != einorun.HandoverDetached {
		t.Fatalf("external status overwritten: %+v", got)
	}
	if str(got.Result) != "receipt" || string(got.Payload) != `{"op":1}` {
		t.Fatalf("receipt or payload not filled in: %+v", got)
	}
	again := child("a", 3, einorun.StatusRunning)
	again.Handover, again.Result, again.Payload = einorun.HandoverDetached, text("other"), json.RawMessage(`{"op":2}`)
	save(t, h, again)
	got = find(t, h)
	if str(got.Result) != "receipt" || string(got.Payload) != `{"op":1}` {
		t.Fatalf("receipt or payload replaced: %+v", got)
	}
}

func testExternalResultFirst(t *testing.T, h JournalHarness) {
	save(t, h, child("a", 1, einorun.StatusRunning))
	external(t, h, einorun.ToolCall{ID: "a", Status: einorun.StatusSucceeded, Result: text("external"), CompletedAt: &at, Handover: einorun.HandoverAwait})
	await := child("a", 2, einorun.StatusWaiting)
	await.Handover, await.Payload = einorun.HandoverAwait, json.RawMessage(`{"op":1}`)
	save(t, h, await)
	got := find(t, h)
	if got.Status != einorun.StatusSucceeded || str(got.Result) != "external" {
		t.Fatalf("external result overwritten: %+v", got)
	}
}

func testStaleHandoverFill(t *testing.T, h JournalHarness) {
	save(t, h, child("a", 5, einorun.StatusRunning))
	external(t, h, einorun.ToolCall{ID: "a", Status: einorun.StatusQueued, Handover: einorun.HandoverAwait})
	stale := child("a", 2, einorun.StatusWaiting)
	stale.Handover, stale.Result, stale.Payload = einorun.HandoverAwait, text("receipt"), json.RawMessage(`{"op":1}`)
	save(t, h, stale)
	got := find(t, h)
	if got.Result != nil || len(got.Payload) != 0 || got.Status != einorun.StatusQueued {
		t.Fatalf("stale write filled the handover: %+v", got)
	}
}

func testNonHandoverFill(t *testing.T, h JournalHarness) {
	save(t, h, child("a", 1, einorun.StatusRunning))
	external(t, h, einorun.ToolCall{ID: "a", Status: einorun.StatusQueued, Handover: einorun.HandoverAwait})
	interrupted := child("a", 2, einorun.StatusInterrupted)
	interrupted.Result = text("interrupted")
	save(t, h, interrupted)
	got := find(t, h)
	if got.Result != nil || got.Status != einorun.StatusQueued {
		t.Fatalf("a settlement became the receipt: %+v", got)
	}
}

func testNotes(t *testing.T, h JournalHarness) {
	first := child("a", 1, einorun.StatusRunning)
	first.Notes = map[string]string{"k": "old"}
	second := child("a", 2, einorun.StatusSucceeded)
	second.Result, second.Notes = text("r"), map[string]string{"k": "new", "evidence": "true"}
	third := child("a", 3, einorun.StatusSucceeded)
	third.Result, third.Notes = text("r"), map[string]string{"k": "new", "evidence": "true", "extra": "x"}
	// Out of order: the newest snapshot wins whatever arrives later.
	save(t, h, third)
	save(t, h, first)
	save(t, h, second)
	want := map[string]string{"k": "new", "evidence": "true", "extra": "x"}
	if got := find(t, h); !maps.Equal(got.Notes, want) {
		t.Fatalf("notes %v, want %v", got.Notes, want)
	}
	// Notes still change after the outcome is final.
	fourth := third.Clone()
	fourth.Rev, fourth.Notes["late"] = 4, "y"
	save(t, h, fourth)
	if got := find(t, h); got.Notes["late"] != "y" {
		t.Fatalf("late note lost: %v", got.Notes)
	}
}

func testSteps(t *testing.T, h JournalHarness) {
	ctx := context.Background()
	main := einorun.ToolCall{ID: "a", CallID: "c-a", Name: "tool", Arguments: "{}", Rev: 1, Status: einorun.StatusRunning, StartedAt: &at}
	step := einorun.Step{
		Changes: einorun.Changes{Blocks: []einorun.Block{
			{ID: "b1", Position: 1, Kind: einorun.KindContent, Text: "hello"},
			{ID: "b2", Position: 2, Kind: einorun.KindToolCall, Call: &main},
		}, Calls: []einorun.ToolCall{main, child("s", 1, einorun.StatusRunning)}},
		Usage:      einorun.Usage{Input: 1, Output: 2, Total: 3},
		Plan:       []einorun.PlanTask{{ID: "1", Subject: "s", Status: "pending"}},
		Completion: &einorun.Completion{Source: einorun.FromGuard, Value: json.RawMessage(`{"a":1}`), Fixed: true},
		State:      []byte("state-1"),
	}
	if err := h.Journal.SaveStep(ctx, step); err != nil {
		t.Fatal(err)
	}
	resume := load(t, h)
	if string(resume.State) != "state-1" || len(resume.Plan) != 1 || resume.Plan[0].Subject != "s" {
		t.Fatalf("state %q plan %+v", resume.State, resume.Plan)
	}
	if resume.Completion == nil || resume.Completion.Source != einorun.FromGuard || !resume.Completion.Fixed || string(resume.Completion.Value) != `{"a":1}` {
		t.Fatalf("completion %+v", resume.Completion)
	}
	if len(resume.Blocks) != 2 || resume.Blocks[0].Text != "hello" || resume.Blocks[1].Call == nil || resume.Blocks[1].Call.ID != "a" {
		t.Fatalf("blocks %+v", resume.Blocks)
	}
	if len(resume.Calls) != 1 || resume.Calls[0].ID != "s" {
		t.Fatalf("sub-agent calls %+v", resume.Calls)
	}
	if h.Usage != nil {
		usage, err := h.Usage(ctx)
		if err != nil || usage != step.Usage {
			t.Fatalf("usage %+v %v", usage, err)
		}
	}
	second := einorun.Step{Changes: einorun.Changes{RemovedBlocks: []string{"b2"}, RemovedCalls: []string{"a"}}, State: []byte("state-2")}
	if err := h.Journal.SaveStep(ctx, second); err != nil {
		t.Fatal(err)
	}
	resume = load(t, h)
	if string(resume.State) != "state-2" || resume.Completion != nil || len(resume.Plan) != 0 {
		t.Fatalf("second step state %q completion %+v plan %+v", resume.State, resume.Completion, resume.Plan)
	}
	if len(resume.Blocks) != 1 || resume.Blocks[0].ID != "b1" {
		t.Fatalf("blocks after removal %+v", resume.Blocks)
	}
	for _, c := range resume.Calls {
		if c.ID == "a" {
			t.Fatal("removed call is still there")
		}
	}
}

func testBlockLinks(t *testing.T, h JournalHarness) {
	ctx := context.Background()
	queued := einorun.ToolCall{ID: "a", CallID: "c-a", Name: "tool", Arguments: "{}", Rev: 1, Status: einorun.StatusQueued}
	block := einorun.Block{ID: "b", Position: 1, Kind: einorun.KindToolCall, Call: &queued}
	if err := h.Journal.SaveStep(ctx, einorun.Step{Changes: einorun.Changes{Blocks: []einorun.Block{block}, Calls: []einorun.ToolCall{queued}}}); err != nil {
		t.Fatal(err)
	}
	done := queued.Clone()
	done.Rev, done.Status, done.Result = 2, einorun.StatusSucceeded, text("ok")
	save(t, h, done)
	// A later step carries the block again with a stale embedded call.
	if err := h.Journal.SaveStep(ctx, einorun.Step{Changes: einorun.Changes{Blocks: []einorun.Block{block}}}); err != nil {
		t.Fatal(err)
	}
	resume := load(t, h)
	if len(resume.Blocks) != 1 || resume.Blocks[0].Call == nil || resume.Blocks[0].Call.Status != einorun.StatusSucceeded {
		t.Fatalf("block call %+v", resume.Blocks)
	}
	for _, c := range resume.Calls {
		if c.ID == "a" {
			t.Fatal("main call listed in Resume.Calls")
		}
	}
}

func testConcurrentOneCall(t *testing.T, h JournalHarness) {
	ctx := context.Background()
	save(t, h, child("a", 1, einorun.StatusQueued))
	var wg sync.WaitGroup
	for rev := uint64(2); rev <= 21; rev++ {
		wg.Add(2)
		go func() {
			defer wg.Done()
			c := child("a", rev, einorun.StatusRunning)
			c.Arguments = fmt.Sprintf(`{"rev":%d}`, rev)
			if err := h.Journal.SaveToolCall(ctx, c); err != nil {
				t.Error(err)
			}
		}()
		go func() {
			defer wg.Done()
			stale := child("a", rev-1, einorun.StatusQueued)
			if err := h.Journal.SaveStep(ctx, einorun.Step{Changes: einorun.Changes{Calls: []einorun.ToolCall{stale}}}); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	got := find(t, h)
	if got.Rev != 21 || got.Arguments != `{"rev":21}` || got.Status != einorun.StatusRunning {
		t.Fatalf("lost update: rev %d %s %s", got.Rev, got.Arguments, got.Status)
	}
}

func testInvalid(t *testing.T, h JournalHarness) {
	if err := h.Journal.SaveToolCall(context.Background(), einorun.ToolCall{ID: "a", Rev: 1}); err == nil {
		t.Fatal("call without a status accepted")
	}
	if err := h.Journal.SaveToolCall(context.Background(), einorun.ToolCall{Rev: 1, Status: einorun.StatusQueued}); err == nil {
		t.Fatal("call without an ID accepted")
	}
}

func appendUser(t *testing.T, h FeedHarness, id string) int64 {
	t.Helper()
	seq, err := h.Append(context.Background(), einorun.Message{ID: id, Revision: "1", Role: einorun.RoleUser, Content: id})
	if err != nil {
		t.Fatal(err)
	}
	return seq
}

func testPendingClaim(t *testing.T, h FeedHarness) {
	ctx := context.Background()
	if latest, err := h.Feed.Pending(ctx, 0); err != nil || latest != 0 {
		t.Fatalf("empty pending %d %v", latest, err)
	}
	first := appendUser(t, h, "m1")
	second := appendUser(t, h, "m2")
	latest, err := h.Feed.Pending(ctx, 0)
	if err != nil || latest != second {
		t.Fatalf("pending %d %v, want %d", latest, err, second)
	}
	if latest, _ := h.Feed.Pending(ctx, second); latest != 0 {
		t.Fatalf("pending after latest = %d", latest)
	}
	claim, err := h.Feed.Claim(ctx, first)
	if err != nil {
		t.Fatal(err)
	}
	if claim.EndSeq <= 0 || claim.EndSeq > first || len(claim.Messages) == 0 {
		t.Fatalf("claim %+v", claim)
	}
	claim, err = h.Feed.Claim(ctx, second)
	if err != nil || claim.EndSeq != second || len(claim.Messages) < 2 || claim.Messages[len(claim.Messages)-1].ID != "m2" {
		t.Fatalf("claim snapshot %+v %v", claim, err)
	}
}

func testMessageRoundTrip(t *testing.T, h FeedHarness) {
	ctx := context.Background()
	media := &einorun.MediaRef{Key: "k", MIME: "image/png", SHA256: "h", Size: 3}
	in := einorun.Message{ID: "m", Revision: "r2", Role: einorun.RoleAssistant, Content: "c", Media: media, Meta: map[string]string{"fact": "true"}}
	seq, err := h.Append(ctx, in)
	if err != nil {
		t.Fatal(err)
	}
	claim, err := h.Feed.Claim(ctx, seq)
	if err != nil {
		t.Fatal(err)
	}
	out := claim.Messages[len(claim.Messages)-1]
	if out.ID != in.ID || out.Revision != in.Revision || out.Role != in.Role || out.Content != in.Content ||
		out.Media == nil || *out.Media != *media || !maps.Equal(out.Meta, in.Meta) {
		t.Fatalf("message %+v, want %+v", out, in)
	}
	// Changing the returned snapshot does not change the conversation.
	out.Meta["fact"], out.Media.Key = "false", "other"
	again, err := h.Feed.Claim(ctx, seq)
	if err != nil {
		t.Fatal(err)
	}
	if last := again.Messages[len(again.Messages)-1]; last.Meta["fact"] != "true" || last.Media.Key != "k" {
		t.Fatalf("snapshot shares memory with the feed: %+v", last)
	}
}

func testReplay(t *testing.T, h FeedHarness) {
	ctx := context.Background()
	seq := appendUser(t, h, "m1")
	first, err := h.Feed.Claim(ctx, seq)
	if err != nil {
		t.Fatal(err)
	}
	appendUser(t, h, "m2")
	again, err := h.Feed.Claim(ctx, first.EndSeq)
	if err != nil || again.EndSeq != first.EndSeq || len(again.Messages) != len(first.Messages) {
		t.Fatalf("replay %+v %v, want %+v", again, err, first)
	}
}

func testConsumed(t *testing.T, h FeedHarness) {
	ctx := context.Background()
	first := appendUser(t, h, "m1")
	if err := h.Consume(ctx, first); err != nil {
		t.Fatal(err)
	}
	if latest, err := h.Feed.Pending(ctx, 0); err != nil || latest != 0 {
		t.Fatalf("pending after consumption %d %v", latest, err)
	}
	if _, err := h.Feed.Claim(ctx, first); err == nil {
		t.Fatal("claimed consumed input")
	}
	second := appendUser(t, h, "m2")
	if latest, err := h.Feed.Pending(ctx, 0); err != nil || latest != second {
		t.Fatalf("pending after new input %d %v", latest, err)
	}
	claim, err := h.Feed.Claim(ctx, second)
	if err != nil || claim.EndSeq != second {
		t.Fatalf("claim after partial consumption %+v %v", claim, err)
	}
}

func testWatch(t *testing.T, h FeedHarness) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	signals, stop, err := h.Feed.Watch(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer stop()
	appendUser(t, h, "m1")
	select {
	case <-signals:
	case <-ctx.Done():
		t.Fatal("no signal after append")
	}
}
