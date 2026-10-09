package einorun

import "context"

// Role is the role of an input message.
type Role string

const (
	// RoleUser is input from the user side of the conversation.
	RoleUser Role = "user"
	// RoleAssistant is a reply already delivered on the assistant side.
	RoleAssistant Role = "assistant"
)

// Message is one message of the conversation a run works on.
type Message struct {
	// ID and Revision identify the message content. The runtime adds a message
	// to the model history once per ID and Revision; a new revision of a
	// message enters the history again.
	ID       string
	Revision string
	Role     Role
	Content  string
	// Media is an attachment the runtime may pass to the model directly, read
	// lazily within the run's media budget.
	Media *MediaRef
	// Meta is passed to guards and extensions and never interpreted by the
	// runtime.
	Meta map[string]string
}

// Claim is the result of claiming input.
type Claim struct {
	// Messages is the conversation up to EndSeq: a snapshot, not only the
	// messages that arrived since the last claim.
	Messages []Message
	// EndSeq is the boundary actually claimed. It may be lower than requested;
	// the input above it stays pending.
	EndSeq int64
}

// Feed is the durable input of a run. Input is numbered with increasing
// sequence numbers.
//
// Contract:
//   - Watch returns once the subscription is in effect. Signals only wake the
//     runtime, which then calls Pending; signals may be coalesced or lost, and
//     the runtime also polls.
//   - Pending returns the highest sequence number above after that can be
//     claimed now, or 0 when there is none.
//   - Claim binds input up to through to the run and returns the conversation
//     snapshot up to the boundary it claimed. Claiming is not consuming: the
//     host consumes input when it delivers the run's result.
//   - Claiming the same boundary again returns the same EndSeq as long as the
//     input has not been consumed; the runtime relies on this to replay its
//     last claim after a restart.
//   - Claim fails when the snapshot is empty or the boundary is at or below
//     the input already consumed.
type Feed interface {
	Watch(ctx context.Context) (signals <-chan struct{}, stop func(), err error)
	Pending(ctx context.Context, after int64) (int64, error)
	Claim(ctx context.Context, through int64) (Claim, error)
}
