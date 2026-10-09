package einorun

import (
	"context"
	"encoding/json"

	"github.com/cloudwego/eino/schema"
)

// Guard reviews candidate text before a run delivers it. A guard reviews
// direct text only, never completion results.
type Guard interface {
	// Review is called for every candidate text, including candidates that
	// turn out to be superseded by new input. An error aborts the run.
	Review(ctx context.Context, turn TurnView) (Verdict, error)
}

// TurnView is what a guard reviews.
type TurnView struct {
	// Claim is the snapshot claimed for this turn.
	Claim Claim
	// ModelInput is what the model saw when it produced Text, after all
	// rewrites and retries.
	ModelInput []*schema.AgenticMessage
	// Calls are this turn's calls of the main agent.
	Calls []CallOutcome
	Text  string
	// Superseded is true when new input is pending: the verdict is ignored,
	// notes still apply.
	Superseded bool
	// BudgetSpent is true when the text came at the end of the iteration
	// budget.
	BudgetSpent     bool
	CorrectionsLeft int
}

// VerdictKind is a guard's decision.
type VerdictKind int

const (
	// Accept delivers the text.
	Accept VerdictKind = iota
	// Correct reruns the turn with the rejected text and Prompt, using one
	// correction. Neither enters later history; the rerun keeps the remaining
	// iteration budget and is not a new turn. Without corrections left the
	// completion fallback applies.
	Correct
	// Finish ends the run with Value as a guard completion.
	Finish
)

// Verdict is a guard's decision and the notes it writes.
type Verdict struct {
	Kind   VerdictKind
	Prompt string
	Value  json.RawMessage
	Fixed  bool
	// Notes are written to calls, keyed by call record ID.
	Notes map[string]map[string]string
}
