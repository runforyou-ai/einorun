// Package journaltest checks implementations of einorun.Journal and
// einorun.Feed against their contracts. Hosts run these suites against their
// persistent implementations in their own tests.
package journaltest

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/runforyou-ai/einorun"
)

// JournalHarness gives the suite access to one empty run in the
// implementation under test.
type JournalHarness struct {
	Journal einorun.Journal
	// Load returns what the host would hand back to resume the run.
	Load func(ctx context.Context) (einorun.Resume, error)
	// External applies a write the host makes outside the runtime, such as
	// dispatching a call to an external executor or settling it. The host's
	// own representation decides what it persists; the suite only writes
	// Status, Result, Error, Handover and Payload this way.
	External func(ctx context.Context, call einorun.ToolCall) error
}

// RunJournal runs the Journal contract suite. newHarness returns a harness for
// a fresh run each time it is called.
func RunJournal(t *testing.T, newHarness func(t *testing.T) JournalHarness) {
	t.Run("newer revisions win", func(t *testing.T) { testNewerRevisionsWin(t, newHarness(t)) })
	t.Run("stale step snapshots lose", func(t *testing.T) { testStaleStep(t, newHarness(t)) })
	t.Run("settled outcomes are frozen", func(t *testing.T) { testSettledFrozen(t, newHarness(t)) })
	t.Run("handover by the runtime", func(t *testing.T) { testRuntimeHandover(t, newHarness(t)) })
	t.Run("handover by the host first", func(t *testing.T) { testHostHandoverFirst(t, newHarness(t)) })
	t.Run("external result before handover", func(t *testing.T) { testExternalResultFirst(t, newHarness(t)) })
	t.Run("notes merge", func(t *testing.T) { testNotes(t, newHarness(t)) })
	t.Run("steps", func(t *testing.T) { testSteps(t, newHarness(t)) })
	t.Run("concurrent writes", func(t *testing.T) { testConcurrent(t, newHarness(t)) })
}

// FeedHarness gives the suite access to one empty conversation in the
// implementation under test.
type FeedHarness struct {
	Feed einorun.Feed
	// Append adds a user message and returns its sequence number.
	Append func(ctx context.Context, message einorun.Message) (int64, error)
	// Consume marks input up to seq as consumed.
	Consume func(ctx context.Context, seq int64) error
}

// RunFeed runs the Feed contract suite.
func RunFeed(t *testing.T, newHarness func(t *testing.T) FeedHarness) {
	t.Run("pending and claim", func(t *testing.T) { testPendingClaim(t, newHarness(t)) })
	t.Run("replay", func(t *testing.T) { testReplay(t, newHarness(t)) })
	t.Run("consumed input", func(t *testing.T) { testConsumed(t, newHarness(t)) })
	t.Run("watch", func(t *testing.T) { testWatch(t, newHarness(t)) })
}

var at = time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

func call(id string, rev uint64, status einorun.CallStatus) einorun.ToolCall {
	return einorun.ToolCall{ID: id, CallID: "c-" + id, Name: "tool", Arguments: "{}", Rev: rev, Status: status, StartedAt: &at}
}

func text(s string) *string { return &s }

// find returns the record of id from a resume.
func find(t *testing.T, h JournalHarness, id string) einorun.ToolCall {
	t.Helper()
	resume, err := h.Load(context.Background())
	if err != nil {
		t.Fatalf("load: %v", err)
	}
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

func testNewerRevisionsWin(t *testing.T, h JournalHarness) {
	running := call("a", 2, einorun.StatusRunning)
	queued := call("a", 1, einorun.StatusQueued)
	save(t, h, running)
	save(t, h, queued) // arrives late
	if got := find(t, h, "a"); got.Status != einorun.StatusRunning || got.Rev != 2 {
		t.Fatalf("got %s rev %d, want running rev 2", got.Status, got.Rev)
	}
}

func testStaleStep(t *testing.T, h JournalHarness) {
	ctx := context.Background()
	save(t, h, call("a", 1, einorun.StatusQueued))
	done := call("a", 3, einorun.StatusSucceeded)
	done.Result, done.CompletedAt = text("ok"), &at
	save(t, h, done)
	stale := call("a", 2, einorun.StatusRunning)
	block := einorun.Block{ID: "b1", Position: 1, Kind: einorun.KindToolCall, Call: &stale}
	if err := h.Journal.SaveStep(ctx, einorun.Step{Changes: einorun.Changes{Blocks: []einorun.Block{block}, Calls: []einorun.ToolCall{stale}}, State: []byte("s")}); err != nil {
		t.Fatal(err)
	}
	if got := find(t, h, "a"); got.Status != einorun.StatusSucceeded || got.Result == nil || *got.Result != "ok" {
		t.Fatalf("stale snapshot won: %+v", got)
	}
}

func testSettledFrozen(t *testing.T, h JournalHarness) {
	failed := call("a", 1, einorun.StatusFailed)
	failed.Error, failed.CompletedAt = text("boom"), &at
	save(t, h, failed)
	retry := call("a", 2, einorun.StatusRunning)
	save(t, h, retry)
	if got := find(t, h, "a"); got.Status != einorun.StatusFailed {
		t.Fatalf("settled status changed to %s", got.Status)
	}
}

func testRuntimeHandover(t *testing.T, h JournalHarness) {
	save(t, h, call("a", 1, einorun.StatusRunning))
	detached := call("a", 2, einorun.StatusRunning)
	detached.Handover, detached.Result, detached.Payload = einorun.HandoverDetached, text("receipt"), json.RawMessage(`{"session":"s1"}`)
	save(t, h, detached)
	late := call("a", 3, einorun.StatusInterrupted)
	late.Result = text("interrupted")
	save(t, h, late)
	got := find(t, h, "a")
	if got.Handover != einorun.HandoverDetached || got.Result == nil || *got.Result != "receipt" || got.Status != einorun.StatusRunning {
		t.Fatalf("handover not kept: %+v", got)
	}
	if string(got.Payload) != `{"session":"s1"}` {
		t.Fatalf("payload %s", got.Payload)
	}
}

func testHostHandoverFirst(t *testing.T, h JournalHarness) {
	ctx := context.Background()
	save(t, h, call("a", 1, einorun.StatusRunning))
	dispatched := call("a", 0, einorun.StatusQueued)
	dispatched.Handover = einorun.HandoverAwait
	if err := h.External(ctx, dispatched); err != nil {
		t.Fatal(err)
	}
	await := call("a", 2, einorun.StatusWaiting)
	await.Handover, await.Payload = einorun.HandoverAwait, json.RawMessage(`{"op":1}`)
	save(t, h, await)
	got := find(t, h, "a")
	if got.Status != einorun.StatusQueued || got.Handover != einorun.HandoverAwait {
		t.Fatalf("external status overwritten: %+v", got)
	}
}

func testExternalResultFirst(t *testing.T, h JournalHarness) {
	ctx := context.Background()
	save(t, h, call("a", 1, einorun.StatusRunning))
	settled := call("a", 0, einorun.StatusSucceeded)
	settled.Handover, settled.Result = einorun.HandoverAwait, text("external")
	if err := h.External(ctx, settled); err != nil {
		t.Fatal(err)
	}
	await := call("a", 2, einorun.StatusWaiting)
	await.Handover = einorun.HandoverAwait
	save(t, h, await)
	got := find(t, h, "a")
	if got.Status != einorun.StatusSucceeded || got.Result == nil || *got.Result != "external" {
		t.Fatalf("external result overwritten: %+v", got)
	}
}

func testNotes(t *testing.T, h JournalHarness) {
	first := call("a", 2, einorun.StatusSucceeded)
	first.Result, first.Notes = text("r"), map[string]string{"evidence": "true", "k": "new"}
	save(t, h, first)
	older := call("a", 1, einorun.StatusRunning)
	older.Notes = map[string]string{"k": "old", "extra": "x"}
	save(t, h, older)
	later := call("a", 3, einorun.StatusSucceeded)
	later.Notes = map[string]string{"k": "latest"}
	save(t, h, later)
	got := find(t, h, "a")
	want := map[string]string{"evidence": "true", "k": "latest", "extra": "x"}
	for key, value := range want {
		if got.Notes[key] != value {
			t.Fatalf("notes %v, want %v", got.Notes, want)
		}
	}
}

func testSteps(t *testing.T, h JournalHarness) {
	ctx := context.Background()
	c := call("a", 1, einorun.StatusRunning)
	step := einorun.Step{
		Changes: einorun.Changes{Blocks: []einorun.Block{
			{ID: "b1", Position: 1, Kind: einorun.KindContent, Text: "hello"},
			{ID: "b2", Position: 2, Kind: einorun.KindToolCall, Call: &c},
		}, Calls: []einorun.ToolCall{c}},
		Usage:      einorun.Usage{Input: 1, Output: 2, Total: 3},
		Plan:       []einorun.PlanTask{{ID: "1", Subject: "s", Status: "pending"}},
		Completion: &einorun.Completion{Source: einorun.FromGuard, Value: json.RawMessage(`{"a":1}`), Fixed: true},
		State:      []byte("state-1"),
	}
	if err := h.Journal.SaveStep(ctx, step); err != nil {
		t.Fatal(err)
	}
	second := einorun.Step{Changes: einorun.Changes{RemovedBlocks: []string{"b2"}, RemovedCalls: []string{"a"}}, State: []byte("state-2")}
	if err := h.Journal.SaveStep(ctx, second); err != nil {
		t.Fatal(err)
	}
	resume, err := h.Load(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if string(resume.State) != "state-2" || resume.Completion != nil {
		t.Fatalf("resume state %q completion %+v", resume.State, resume.Completion)
	}
	if len(resume.Blocks) != 1 || resume.Blocks[0].Text != "hello" {
		t.Fatalf("blocks %+v", resume.Blocks)
	}
}

func testConcurrent(t *testing.T, h JournalHarness) {
	ctx := context.Background()
	var wg sync.WaitGroup
	for i := range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			id := fmt.Sprintf("c%d", i)
			for rev := uint64(1); rev <= 5; rev++ {
				status := einorun.StatusRunning
				if rev == 5 {
					status = einorun.StatusSucceeded
				}
				if err := h.Journal.SaveToolCall(ctx, call(id, rev, status)); err != nil {
					t.Error(err)
				}
			}
		}()
	}
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := range 5 {
			if err := h.Journal.SaveStep(ctx, einorun.Step{State: []byte(fmt.Sprint(i))}); err != nil {
				t.Error(err)
			}
		}
	}()
	wg.Wait()
	for i := range 8 {
		if got := find(t, h, fmt.Sprintf("c%d", i)); got.Status != einorun.StatusSucceeded {
			t.Fatalf("c%d ended %s", i, got.Status)
		}
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
	if err != nil || claim.EndSeq != second || claim.Messages[len(claim.Messages)-1].ID != "m2" {
		t.Fatalf("claim snapshot %+v %v", claim, err)
	}
	if len(claim.Messages) < 2 {
		t.Fatalf("claim is not a snapshot: %+v", claim.Messages)
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
	seq := appendUser(t, h, "m1")
	if err := h.Consume(ctx, seq); err != nil {
		t.Fatal(err)
	}
	if _, err := h.Feed.Claim(ctx, seq); err == nil {
		t.Fatal("claimed consumed input")
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
