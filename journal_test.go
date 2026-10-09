package einorun

import (
	"errors"
	"fmt"
	"testing"
)

func TestRejectCall(t *testing.T) {
	err := fmt.Errorf("submit: %w", RejectCall("nobody can approve this"))
	if !errors.Is(err, ErrCallRejected) {
		t.Fatal("rejection does not match ErrCallRejected")
	}
	var rejection *CallRejection
	if !errors.As(err, &rejection) || rejection.Reason != "nobody can approve this" {
		t.Fatalf("reason %+v", rejection)
	}
}

func TestSettled(t *testing.T) {
	for _, s := range []CallStatus{StatusQueued, StatusRunning, StatusWaiting, StatusAwaitingDecision} {
		if s.Settled() {
			t.Fatalf("%s settled", s)
		}
	}
	for _, s := range []CallStatus{StatusSucceeded, StatusFailed, StatusInterrupted, StatusNeedsReview, StatusRejected, "expired"} {
		if !s.Settled() {
			t.Fatalf("%s not settled", s)
		}
	}
}

func TestOverlayExternalClearsMedia(t *testing.T) {
	stored := ToolCall{ID: "a", Status: StatusRunning, Media: []MediaRef{{Key: "old"}}, Name: "n", Notes: map[string]string{"k": "v"}}
	got := OverlayExternal(stored, ToolCall{ID: "a", Status: StatusSucceeded, Media: []MediaRef{}})
	if len(got.Media) != 0 || got.Name != "n" || got.Notes["k"] != "v" || got.Status != StatusSucceeded {
		t.Fatalf("overlay %+v", got)
	}
}

func TestMergeCallFirstWrite(t *testing.T) {
	result := "r"
	in := ToolCall{ID: "a", Rev: 1, Status: StatusSucceeded, Result: &result, Notes: map[string]string{"k": "v"}}
	got := MergeCall(nil, in)
	in.Notes["k"] = "changed"
	*in.Result = "changed"
	if got.Notes["k"] != "v" || *got.Result != "r" {
		t.Fatal("MergeCall shares memory with its input")
	}
}
