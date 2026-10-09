// Package journaltest checks implementations of einorun.Journal and
// einorun.Feed against their contracts. Hosts run these suites against their
// persistent implementations in their own tests.
//
// Every record ID and ModelCallID the suites write is a fresh UUIDv7, so the
// suites run against schemas with UUID columns and share one database across
// harnesses. Payloads are compared as JSON values.
package journaltest

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/runforyou-ai/einorun"
	"github.com/runforyou-ai/einorun/llm"
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
	// Call returns the stored record of a call, main or sub-agent, and false
	// when there is none.
	Call func(ctx context.Context, id string) (einorun.ToolCall, bool, error)
	// Usage, when set, returns the usage of the last saved step.
	Usage func(ctx context.Context) (einorun.Usage, error)
}

// RunJournal runs the Journal contract suite. newHarness returns a harness for
// a fresh run each time it is called.
func RunJournal(t *testing.T, newHarness func(t *testing.T) JournalHarness) {
	for _, c := range []struct {
		name string
		run  func(*testing.T, JournalHarness, ids)
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
		t.Run(c.name, func(t *testing.T) { c.run(t, newHarness(t), ids{}) })
	}
}

// FeedHarness gives the suite access to one empty conversation in the
// implementation under test.
type FeedHarness struct {
	Feed einorun.Feed
	// Append adds a message like message to the conversation and returns its
	// sequence number and the message as Claim hands it back. Hosts that
	// derive messages from their own records may assign their own ID and
	// revision and keep only the fields they store; the suite compares claims
	// with the returned message. The suite appends user and assistant
	// messages, with and without media and meta.
	Append func(ctx context.Context, message einorun.Message) (int64, einorun.Message, error)
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

// ids gives each name used by one test a fresh UUIDv7.
type ids map[string]string

// of returns the ID of name.
func (n ids) of(name string) string {
	if id, ok := n[name]; ok {
		return id
	}
	n[name] = llm.NewModelCallID()
	return n[name]
}

// child returns a sub-agent call snapshot; sub-agent calls come back in
// Resume.Calls without needing a block.
func child(n ids, name string, rev uint64, status einorun.CallStatus) einorun.ToolCall {
	return einorun.ToolCall{ID: n.of(name), ParentID: n.of("parent"), ModelCallID: n.of("model"), CallID: "call_" + name + "_" + n.of("call")[24:],
		Name: "tool", Arguments: "{}", Rev: rev, Status: status, StartedAt: &at, Replayable: true}
}

// mainCall returns a main-agent call snapshot.
func mainCall(n ids, name string, rev uint64, status einorun.CallStatus) einorun.ToolCall {
	call := child(n, name, rev, status)
	call.ParentID, call.StartedAt = "", nil
	return call
}

// sameJSON reports whether a and b hold the same JSON value; an empty payload
// only equals another empty payload.
func sameJSON(a, b []byte) bool {
	if len(a) == 0 || len(b) == 0 {
		return len(a) == len(b)
	}
	decode := func(data []byte) (any, error) {
		decoder := json.NewDecoder(bytes.NewReader(data))
		decoder.UseNumber()
		var v any
		return v, decoder.Decode(&v)
	}
	x, errX := decode(a)
	y, errY := decode(b)
	if errX != nil || errY != nil {
		return string(a) == string(b)
	}
	return reflect.DeepEqual(x, y)
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

// find returns the record of the call "a", which most journal tests use. It
// checks that Load hands back the same record, as resuming would see it.
func find(t *testing.T, h JournalHarness, n ids) einorun.ToolCall {
	t.Helper()
	call, ok := lookup(t, h, n.of("a"))
	if !ok {
		t.Fatal("call a not found")
	}
	resume := load(t, h)
	var loaded *einorun.ToolCall
	for _, block := range resume.Blocks {
		if block.Call != nil && block.Call.ID == call.ID {
			loaded = block.Call
		}
	}
	for i := range resume.Calls {
		if resume.Calls[i].ID == call.ID {
			loaded = &resume.Calls[i]
		}
	}
	if loaded == nil {
		t.Fatal("call a is missing from Load")
	}
	if diff := differ(call, *loaded); diff != "" {
		t.Fatalf("Load hands back a different record than Call: %s", diff)
	}
	return call
}

// differ names the first field in which a and b differ, or returns "".
func differ(a, b einorun.ToolCall) string {
	sameTime := func(x, y *time.Time) bool { return (x == nil) == (y == nil) && (x == nil || x.Equal(*y)) }
	switch {
	case a.Rev != b.Rev:
		return "Rev"
	case a.Status != b.Status:
		return "Status"
	case str(a.Result) != str(b.Result):
		return "Result"
	case str(a.Error) != str(b.Error):
		return "Error"
	case a.Handover != b.Handover:
		return "Handover"
	case !sameJSON(a.Payload, b.Payload):
		return "Payload"
	case a.Name != b.Name || a.Arguments != b.Arguments:
		return "Name or Arguments"
	case a.ParentID != b.ParentID || a.ModelCallID != b.ModelCallID || a.CallID != b.CallID:
		return "ParentID, ModelCallID or CallID"
	case a.Replayable != b.Replayable || a.SideEffects != b.SideEffects:
		return "Replayable or SideEffects"
	case !maps.Equal(a.Notes, b.Notes):
		return "Notes"
	case !sameTime(a.StartedAt, b.StartedAt) || !sameTime(a.CompletedAt, b.CompletedAt):
		return "times"
	}
	return ""
}

// lookup returns the stored record of id.
func lookup(t *testing.T, h JournalHarness, id string) (einorun.ToolCall, bool) {
	t.Helper()
	call, ok, err := h.Call(context.Background(), id)
	if err != nil {
		t.Fatalf("call %s: %v", id, err)
	}
	return call, ok
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

func testNewerRevisionsWin(t *testing.T, h JournalHarness, n ids) {
	running := child(n, "a", 2, einorun.StatusRunning)
	running.Arguments = `{"x":2}`
	queued := child(n, "a", 1, einorun.StatusQueued)
	save(t, h, running)
	save(t, h, queued) // arrives late
	got := find(t, h, n)
	if got.Status != einorun.StatusRunning || got.Rev != 2 || got.Arguments != `{"x":2}` {
		t.Fatalf("got %s rev %d %s, want running rev 2", got.Status, got.Rev, got.Arguments)
	}
}

func testRepeatedWrite(t *testing.T, h JournalHarness, n ids) {
	done := child(n, "a", 3, einorun.StatusSucceeded)
	done.Result, done.CompletedAt, done.Notes = text("ok"), &at, map[string]string{"k": "v"}
	save(t, h, done)
	save(t, h, done)
	got := find(t, h, n)
	if got.Status != einorun.StatusSucceeded || str(got.Result) != "ok" || got.Rev != 3 || got.Notes["k"] != "v" {
		t.Fatalf("repeated write changed the record: %+v", got)
	}
}

func testStaleStep(t *testing.T, h JournalHarness, n ids) {
	ctx := context.Background()
	save(t, h, child(n, "a", 1, einorun.StatusQueued))
	done := child(n, "a", 3, einorun.StatusSucceeded)
	done.Result, done.CompletedAt = text("ok"), &at
	save(t, h, done)
	stale := child(n, "a", 2, einorun.StatusRunning)
	if err := h.Journal.SaveStep(ctx, einorun.Step{Changes: einorun.Changes{Calls: []einorun.ToolCall{stale}}, State: []byte("s")}); err != nil {
		t.Fatal(err)
	}
	if got := find(t, h, n); got.Status != einorun.StatusSucceeded || str(got.Result) != "ok" {
		t.Fatalf("stale snapshot won: %+v", got)
	}
}

func testSettledFinal(t *testing.T, h JournalHarness, n ids) {
	failed := child(n, "a", 1, einorun.StatusFailed)
	failed.Error, failed.CompletedAt = text("boom"), &at
	save(t, h, failed)
	later := at.Add(time.Hour)
	retry := child(n, "a", 2, einorun.StatusSucceeded)
	retry.Result, retry.CompletedAt, retry.Payload = text("ok"), &later, json.RawMessage(`{}`)
	retry.Handover = einorun.HandoverAwait
	save(t, h, retry)
	got := find(t, h, n)
	if got.Status != einorun.StatusFailed || str(got.Error) != "boom" || got.Result != nil || !got.CompletedAt.Equal(at) ||
		len(got.Payload) != 0 || got.Handover != einorun.HandoverNone {
		t.Fatalf("settled outcome changed: %+v", got)
	}
}

func testRuntimeHandover(t *testing.T, h JournalHarness, n ids) {
	save(t, h, child(n, "a", 1, einorun.StatusRunning))
	detached := child(n, "a", 2, einorun.StatusRunning)
	detached.Handover, detached.Result, detached.Payload = einorun.HandoverDetached, text("receipt"), json.RawMessage(`{"session":"s1"}`)
	save(t, h, detached)
	late := child(n, "a", 3, einorun.StatusInterrupted)
	late.Result = text("interrupted")
	save(t, h, late)
	got := find(t, h, n)
	if got.Handover != einorun.HandoverDetached || str(got.Result) != "receipt" || got.Status != einorun.StatusRunning {
		t.Fatalf("handover not kept: %+v", got)
	}
	if !sameJSON(got.Payload, []byte(`{"session":"s1"}`)) {
		t.Fatalf("payload %s", got.Payload)
	}
}

func testHostHandoverFirst(t *testing.T, h JournalHarness, n ids) {
	save(t, h, child(n, "a", 1, einorun.StatusRunning))
	external(t, h, einorun.ToolCall{ID: n.of("a"), Status: einorun.StatusQueued, Handover: einorun.HandoverDetached})
	if got := find(t, h, n); got.Name != "tool" || got.Arguments != "{}" {
		t.Fatalf("external write lost fields: %+v", got)
	}
	detached := child(n, "a", 2, einorun.StatusRunning)
	detached.Handover, detached.Result, detached.Payload = einorun.HandoverDetached, text("receipt"), json.RawMessage(`{"op":1}`)
	save(t, h, detached)
	got := find(t, h, n)
	if got.Status != einorun.StatusQueued || got.Handover != einorun.HandoverDetached {
		t.Fatalf("external status overwritten: %+v", got)
	}
	if str(got.Result) != "receipt" || !sameJSON(got.Payload, []byte(`{"op":1}`)) {
		t.Fatalf("receipt or payload not filled in: %+v", got)
	}
	again := child(n, "a", 3, einorun.StatusRunning)
	again.Handover, again.Result, again.Payload = einorun.HandoverDetached, text("other"), json.RawMessage(`{"op":2}`)
	save(t, h, again)
	got = find(t, h, n)
	if str(got.Result) != "receipt" || !sameJSON(got.Payload, []byte(`{"op":1}`)) {
		t.Fatalf("receipt or payload replaced: %+v", got)
	}
}

func testExternalResultFirst(t *testing.T, h JournalHarness, n ids) {
	save(t, h, child(n, "a", 1, einorun.StatusRunning))
	external(t, h, einorun.ToolCall{ID: n.of("a"), Status: einorun.StatusSucceeded, Result: text("external"), CompletedAt: &at, Handover: einorun.HandoverAwait})
	await := child(n, "a", 2, einorun.StatusWaiting)
	await.Handover, await.Payload = einorun.HandoverAwait, json.RawMessage(`{"op":1}`)
	save(t, h, await)
	got := find(t, h, n)
	if got.Status != einorun.StatusSucceeded || str(got.Result) != "external" {
		t.Fatalf("external result overwritten: %+v", got)
	}
	if !sameJSON(got.Payload, []byte(`{"op":1}`)) {
		t.Fatalf("payload not filled in after an external result: %s", got.Payload)
	}
}

func testStaleHandoverFill(t *testing.T, h JournalHarness, n ids) {
	save(t, h, child(n, "a", 5, einorun.StatusRunning))
	external(t, h, einorun.ToolCall{ID: n.of("a"), Status: einorun.StatusQueued, Handover: einorun.HandoverAwait})
	stale := child(n, "a", 2, einorun.StatusWaiting)
	stale.Handover, stale.Result, stale.Payload = einorun.HandoverAwait, text("receipt"), json.RawMessage(`{"op":1}`)
	save(t, h, stale)
	got := find(t, h, n)
	if got.Result != nil || len(got.Payload) != 0 || got.Status != einorun.StatusQueued {
		t.Fatalf("stale write filled the handover: %+v", got)
	}
}

func testNonHandoverFill(t *testing.T, h JournalHarness, n ids) {
	save(t, h, child(n, "a", 1, einorun.StatusRunning))
	external(t, h, einorun.ToolCall{ID: n.of("a"), Status: einorun.StatusQueued, Handover: einorun.HandoverAwait})
	interrupted := child(n, "a", 2, einorun.StatusInterrupted)
	interrupted.Result = text("interrupted")
	save(t, h, interrupted)
	got := find(t, h, n)
	if got.Result != nil || got.Status != einorun.StatusQueued {
		t.Fatalf("a settlement became the receipt: %+v", got)
	}
}

func testNotes(t *testing.T, h JournalHarness, n ids) {
	first := child(n, "a", 1, einorun.StatusRunning)
	first.Notes = map[string]string{"k": "old"}
	second := child(n, "a", 2, einorun.StatusSucceeded)
	second.Result, second.Notes = text("r"), map[string]string{"k": "new", "evidence": "true"}
	third := child(n, "a", 3, einorun.StatusSucceeded)
	third.Result, third.Notes = text("r"), map[string]string{"k": "new", "evidence": "true", "extra": "x"}
	// Out of order: the newest snapshot wins whatever arrives later.
	save(t, h, third)
	save(t, h, first)
	save(t, h, second)
	want := map[string]string{"k": "new", "evidence": "true", "extra": "x"}
	if got := find(t, h, n); !maps.Equal(got.Notes, want) {
		t.Fatalf("notes %v, want %v", got.Notes, want)
	}
	// Notes still change after the outcome is final.
	fourth := third.Clone()
	fourth.Rev, fourth.Notes["late"] = 4, "y"
	save(t, h, fourth)
	if got := find(t, h, n); got.Notes["late"] != "y" {
		t.Fatalf("late note lost: %v", got.Notes)
	}
}

func testSteps(t *testing.T, h JournalHarness, n ids) {
	ctx := context.Background()
	main := mainCall(n, "a", 1, einorun.StatusRunning)
	main.StartedAt = &at
	step := einorun.Step{
		Changes: einorun.Changes{Blocks: []einorun.Block{
			{ID: n.of("b1"), Position: 1, ModelCallID: n.of("model"), Kind: einorun.KindContent, Text: "hello"},
			{ID: n.of("b2"), Position: 2, ModelCallID: n.of("model"), Kind: einorun.KindToolCall, Call: &main},
		}, Calls: []einorun.ToolCall{main, child(n, "s", 1, einorun.StatusRunning)}},
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
	if resume.Completion == nil || resume.Completion.Source != einorun.FromGuard || !resume.Completion.Fixed || !sameJSON(resume.Completion.Value, []byte(`{"a":1}`)) {
		t.Fatalf("completion %+v", resume.Completion)
	}
	if len(resume.Blocks) != 2 || resume.Blocks[0].Text != "hello" || resume.Blocks[1].Call == nil || resume.Blocks[1].Call.ID != n.of("a") ||
		resume.Blocks[0].ModelCallID != n.of("model") || resume.Blocks[1].Position != 2 {
		t.Fatalf("blocks %+v", resume.Blocks)
	}
	if diff := differ(*resume.Blocks[1].Call, main); diff != "" {
		t.Fatalf("main call read back differs in %s: %+v", diff, *resume.Blocks[1].Call)
	}
	if len(resume.Calls) != 1 || resume.Calls[0].ID != n.of("s") {
		t.Fatalf("sub-agent calls %+v", resume.Calls)
	}
	if diff := differ(resume.Calls[0], step.Changes.Calls[1]); diff != "" {
		t.Fatalf("sub-agent call read back differs in %s: %+v", diff, resume.Calls[0])
	}
	if h.Usage != nil {
		usage, err := h.Usage(ctx)
		if err != nil || usage != step.Usage {
			t.Fatalf("usage %+v %v", usage, err)
		}
	}
	second := einorun.Step{Changes: einorun.Changes{RemovedBlocks: []string{n.of("b2")}, RemovedCalls: []string{n.of("a")}}, State: []byte("state-2")}
	if err := h.Journal.SaveStep(ctx, second); err != nil {
		t.Fatal(err)
	}
	resume = load(t, h)
	if string(resume.State) != "state-2" || resume.Completion != nil || len(resume.Plan) != 0 {
		t.Fatalf("second step state %q completion %+v plan %+v", resume.State, resume.Completion, resume.Plan)
	}
	if len(resume.Blocks) != 1 || resume.Blocks[0].ID != n.of("b1") {
		t.Fatalf("blocks after removal %+v", resume.Blocks)
	}
	if _, ok := lookup(t, h, n.of("a")); ok {
		t.Fatal("removed call is still there")
	}
	if _, ok := lookup(t, h, n.of("s")); !ok {
		t.Fatal("a call that was not removed is gone")
	}
}

func testBlockLinks(t *testing.T, h JournalHarness, n ids) {
	ctx := context.Background()
	queued := mainCall(n, "a", 1, einorun.StatusQueued)
	block := einorun.Block{ID: n.of("b"), Position: 1, ModelCallID: n.of("model"), Kind: einorun.KindToolCall, Call: &queued}
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
		if c.ID == n.of("a") {
			t.Fatal("main call listed in Resume.Calls")
		}
	}
}

func testConcurrentOneCall(t *testing.T, h JournalHarness, n ids) {
	ctx := context.Background()
	save(t, h, child(n, "a", 1, einorun.StatusQueued))
	var wg sync.WaitGroup
	for rev := uint64(2); rev <= 21; rev++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			c := child(n, "a", rev, einorun.StatusRunning)
			c.Arguments = fmt.Sprintf(`{"rev":%d}`, rev)
			if err := h.Journal.SaveToolCall(ctx, c); err != nil {
				t.Error(err)
			}
		}()
	}
	// Steps are saved one at a time, concurrently with the calls.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for rev := uint64(1); rev <= 20; rev++ {
			stale := child(n, "a", rev, einorun.StatusQueued)
			if err := h.Journal.SaveStep(ctx, einorun.Step{Changes: einorun.Changes{Calls: []einorun.ToolCall{stale}}}); err != nil {
				t.Error(err)
			}
		}
	}()
	wg.Wait()
	got := find(t, h, n)
	if got.Rev != 21 || got.Arguments != `{"rev":21}` || got.Status != einorun.StatusRunning {
		t.Fatalf("lost update: rev %d %s %s", got.Rev, got.Arguments, got.Status)
	}
}

func testInvalid(t *testing.T, h JournalHarness, n ids) {
	if err := h.Journal.SaveToolCall(context.Background(), einorun.ToolCall{ID: n.of("a"), Rev: 1}); err == nil {
		t.Fatal("call without a status accepted")
	}
	if err := h.Journal.SaveToolCall(context.Background(), einorun.ToolCall{Rev: 1, Status: einorun.StatusQueued}); err == nil {
		t.Fatal("call without an ID accepted")
	}
}

// appendMessage appends message and returns its sequence number and the
// message as the feed hands it back.
func appendMessage(t *testing.T, h FeedHarness, message einorun.Message) (int64, einorun.Message) {
	t.Helper()
	seq, stored, err := h.Append(context.Background(), message)
	if err != nil {
		t.Fatal(err)
	}
	// The suite keeps its own copy, so a feed sharing memory with it is caught.
	stored.Meta = maps.Clone(stored.Meta)
	if stored.Media != nil {
		media := *stored.Media
		stored.Media = &media
	}
	return seq, stored
}

// appendUser appends a user message and returns its sequence number and
// identity (ID and revision).
func appendUser(t *testing.T, h FeedHarness, content string) (int64, string) {
	t.Helper()
	seq, stored := appendMessage(t, h, einorun.Message{ID: llm.NewModelCallID(), Revision: "1", Role: einorun.RoleUser, Content: content})
	return seq, identity(stored)
}

// identity returns the key the runtime tells messages apart by.
func identity(m einorun.Message) string { return m.ID + "@" + m.Revision }

// sameMessage names the first field in which a and b differ, or returns "".
func sameMessage(a, b einorun.Message) string {
	switch {
	case a.ID != b.ID || a.Revision != b.Revision:
		return "ID or Revision"
	case a.Role != b.Role || a.Content != b.Content:
		return "Role or Content"
	case (a.Media == nil) != (b.Media == nil) || (a.Media != nil && *a.Media != *b.Media):
		return "Media"
	case !maps.Equal(a.Meta, b.Meta):
		return "Meta"
	}
	return ""
}

func testPendingClaim(t *testing.T, h FeedHarness) {
	ctx := context.Background()
	if latest, err := h.Feed.Pending(ctx, 0); err != nil || latest != 0 {
		t.Fatalf("empty pending %d %v", latest, err)
	}
	first, firstID := appendUser(t, h, "m1")
	second, secondID := appendUser(t, h, "m2")
	if firstID == secondID {
		t.Fatalf("two messages share the identity %s", firstID)
	}
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
	if err != nil || claim.EndSeq != second || len(claim.Messages) < 2 {
		t.Fatalf("claim snapshot %+v %v", claim, err)
	}
	last, previous := claim.Messages[len(claim.Messages)-1], claim.Messages[len(claim.Messages)-2]
	if identity(last) != secondID || identity(previous) != firstID {
		t.Fatalf("claimed identities %s, %s, want %s, %s", identity(previous), identity(last), firstID, secondID)
	}
}

func testMessageRoundTrip(t *testing.T, h FeedHarness) {
	ctx := context.Background()
	seen := map[string]bool{}
	for _, in := range []einorun.Message{
		{ID: llm.NewModelCallID(), Revision: "r2", Role: einorun.RoleAssistant, Content: "answer"},
		{ID: llm.NewModelCallID(), Revision: "r1", Role: einorun.RoleUser, Content: "look",
			Media: &einorun.MediaRef{Key: "k", MIME: "image/png", SHA256: "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08", Size: 4},
			Meta:  map[string]string{"fact": "true"}},
	} {
		seq, stored := appendMessage(t, h, in)
		claim, err := h.Feed.Claim(ctx, seq)
		if err != nil {
			t.Fatal(err)
		}
		if seen[identity(stored)] {
			t.Fatalf("two messages share the identity %s", identity(stored))
		}
		seen[identity(stored)] = true
		out := claim.Messages[len(claim.Messages)-1]
		if diff := sameMessage(out, stored); diff != "" {
			t.Fatalf("claimed message differs from the stored one in %s: %+v, want %+v", diff, out, stored)
		}
		// Changing the returned snapshot does not change the conversation.
		if out.Meta != nil {
			for k := range out.Meta {
				out.Meta[k] += "-changed"
			}
		}
		if out.Media != nil {
			out.Media.Key += "-changed"
		}
		again, err := h.Feed.Claim(ctx, seq)
		if err != nil {
			t.Fatal(err)
		}
		if diff := sameMessage(again.Messages[len(again.Messages)-1], stored); diff != "" {
			t.Fatalf("snapshot shares memory with the feed: %s", diff)
		}
	}
}

func testReplay(t *testing.T, h FeedHarness) {
	ctx := context.Background()
	seq, _ := appendUser(t, h, "m1")
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
	first, _ := appendUser(t, h, "m1")
	if err := h.Consume(ctx, first); err != nil {
		t.Fatal(err)
	}
	if latest, err := h.Feed.Pending(ctx, 0); err != nil || latest != 0 {
		t.Fatalf("pending after consumption %d %v", latest, err)
	}
	if _, err := h.Feed.Claim(ctx, first); err == nil {
		t.Fatal("claimed consumed input")
	}
	second, _ := appendUser(t, h, "m2")
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
