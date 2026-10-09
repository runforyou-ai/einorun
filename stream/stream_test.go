package stream

import (
	"errors"
	"testing"
)

func TestApply(t *testing.T) {
	s := Snapshot{Stream: "r1"}
	ok, err := s.Apply(Delta{Stream: "r1", Base: 0, Sequence: 1, Operations: []Operation{
		{Kind: OpUpsertBlock, Block: &Block{ID: "b1", Kind: KindContent, Text: "he"}},
		{Kind: OpAppendText, BlockID: "b1", Text: "llo"},
		{Kind: OpAppendCandidate, Text: "draft"},
		{Kind: OpSetPlan, Plan: []PlanTask{{ID: "1", Subject: "a", Status: PlanPending}}},
	}})
	if !ok || err != nil {
		t.Fatalf("apply: %v %v", ok, err)
	}
	if s.Sequence != 1 || s.Blocks[0].Text != "hello" || s.Candidate != "draft" || len(s.Plan) != 1 {
		t.Fatalf("unexpected snapshot %+v", s)
	}

	// A duplicate is ignored without an error.
	ok, err = s.Apply(Delta{Stream: "r1", Base: 0, Sequence: 1})
	if ok || err != nil {
		t.Fatalf("duplicate: %v %v", ok, err)
	}
	if _, err := s.Apply(Delta{Stream: "r2", Base: 1, Sequence: 2}); !errors.Is(err, ErrMismatch) {
		t.Fatalf("mismatch: %v", err)
	}
	if _, err := s.Apply(Delta{Stream: "r1", Base: 2, Sequence: 3}); !errors.Is(err, ErrGap) {
		t.Fatalf("gap: %v", err)
	}

	// A failing operation leaves the snapshot unchanged.
	before := s.Clone()
	if _, err := s.Apply(Delta{Stream: "r1", Base: 1, Sequence: 2, Operations: []Operation{
		{Kind: OpClearCandidate},
		{Kind: OpAppendText, BlockID: "missing", Text: "x"},
	}}); err == nil {
		t.Fatal("expected an error")
	}
	if s.Sequence != before.Sequence || s.Candidate != before.Candidate {
		t.Fatalf("snapshot changed: %+v", s)
	}

	ok, err = s.Apply(Delta{Stream: "r1", Base: 1, Sequence: 2, Operations: []Operation{
		{Kind: OpRemoveBlocks, BlockIDs: []string{"b1"}},
		{Kind: OpClearCandidate},
	}})
	if !ok || err != nil || len(s.Blocks) != 0 || s.Candidate != "" {
		t.Fatalf("remove: %v %v %+v", ok, err, s)
	}
}

func TestMerge(t *testing.T) {
	a := Delta{Stream: "r", Base: 0, Sequence: 1, Operations: []Operation{
		{Kind: OpUpsertBlock, Block: &Block{ID: "b", Text: "a"}},
	}}
	b := Delta{Stream: "r", Base: 1, Sequence: 2, Operations: []Operation{
		{Kind: OpAppendText, BlockID: "b", Text: "b"},
		{Kind: OpAppendText, BlockID: "b", Text: "c"},
		{Kind: OpAppendCandidate, Text: "x"},
		{Kind: OpAppendCandidate, Text: "y"},
		{Kind: OpSetPlan, Plan: []PlanTask{{ID: "1"}}},
		{Kind: OpSetPlan, Plan: []PlanTask{{ID: "2"}}},
	}}
	merged, ok := MergeDeltas(a, b)
	if !ok || merged.Base != 0 || merged.Sequence != 2 {
		t.Fatalf("merge: %v %+v", ok, merged)
	}
	if len(merged.Operations) != 3 || merged.Operations[0].Block.Text != "abc" ||
		merged.Operations[1].Text != "xy" || merged.Operations[2].Plan[0].ID != "2" {
		t.Fatalf("operations: %+v", merged.Operations)
	}
	if a.Operations[0].Block.Text != "a" {
		t.Fatal("merge modified its input")
	}
	if merged.TextBytes() != len("abc")+len("xy") {
		t.Fatalf("text bytes %d", merged.TextBytes())
	}
	if _, ok := MergeDeltas(b, a); ok {
		t.Fatal("merged deltas that do not connect")
	}
}
