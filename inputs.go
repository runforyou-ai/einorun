package einorun

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/cloudwego/eino/adk"
	"github.com/cloudwego/eino/schema"
)

// fallbackPoll is how often the feed is read when no signal arrives.
const fallbackPoll = 5 * time.Second

// ErrEmptyResponse reports a turn that produced neither text nor a
// completion, even after the model call was retried.
var ErrEmptyResponse = errors.New("einorun: the model produced no response")

// trigger is a signal for the turn loop.
type trigger struct {
	seq        int64
	correction bool // rerun the turn with a correction; claims no input
	resume     bool // continue the turn restored from the checkpoint
}

// inputs feeds durable input into the turn loop: it delivers new input,
// keeps the claimed boundary and decides when the loop stops.
type inputs struct {
	feed Feed
	loop *adk.TurnLoop[trigger, *schema.AgenticMessage]
	// hold keeps new input from preempting the turn, once a fixed completion
	// is active.
	hold func() bool

	mu         sync.Mutex
	maxPushed  int64
	claimedSeq int64
	closed     bool
	resuming   bool
}

// resumeFrom starts from the claimed boundary of a checkpoint; the first
// signal continues the restored turn.
func (i *inputs) resumeFrom(claimedSeq int64) {
	i.mu.Lock()
	defer i.mu.Unlock()
	i.claimedSeq, i.maxPushed, i.resuming = claimedSeq, claimedSeq, true
}

// run subscribes to input, pushes the initial input and waits for the loop
// and the watcher to end.
func (i *inputs) run(ctx context.Context) error {
	signals, stop, err := i.feed.Watch(ctx)
	if err != nil {
		return err
	}
	defer stop()
	if i.resuming {
		if accepted, _ := i.loop.Push(trigger{resume: true}); !accepted {
			return errors.New("einorun: turn loop rejected the resume")
		}
	}
	if err := i.poll(ctx, false); err != nil {
		return err
	}
	if !i.resuming && i.maxPushed == 0 {
		return errors.New("einorun: run has no pending input")
	}
	runCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	i.loop.Run(runCtx)
	watched := make(chan error, 1)
	go func() { watched <- i.watch(runCtx, signals) }()
	exit := i.loop.Wait()
	cancel()
	if err := <-watched; err != nil {
		return err
	}
	return exit.ExitReason
}

// watch pushes new input on signals and on the fallback poll.
func (i *inputs) watch(ctx context.Context, signals <-chan struct{}) error {
	ticker := time.NewTicker(fallbackPoll)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return nil
		case <-signals:
		case <-ticker.C:
		}
		if err := i.poll(ctx, true); err != nil {
			i.mu.Lock()
			defer i.mu.Unlock()
			if i.closed || ctx.Err() != nil {
				return nil
			}
			i.closed = true
			i.loop.Stop(adk.WithImmediate())
			return err
		}
	}
}

// poll pushes the latest pending input, preempting the turn at a safe point
// when preempt is set.
func (i *inputs) poll(ctx context.Context, preempt bool) error {
	i.mu.Lock()
	after, closed := i.maxPushed, i.closed
	i.mu.Unlock()
	if closed {
		return nil
	}
	latest, err := i.feed.Pending(ctx, after)
	if err != nil {
		return err
	}
	var ack <-chan struct{}
	i.mu.Lock()
	if !i.closed && latest > i.maxPushed {
		var accepted bool
		if preempt && (i.hold == nil || !i.hold()) {
			accepted, ack = i.loop.Push(trigger{seq: latest}, adk.WithPreempt[trigger, *schema.AgenticMessage](adk.AnySafePoint))
		} else {
			accepted, _ = i.loop.Push(trigger{seq: latest})
		}
		if accepted {
			i.maxPushed = latest
		}
	}
	i.mu.Unlock()
	if ack != nil {
		<-ack
	}
	return nil
}

// claim claims input up to through and records the boundary. Pending only
// reports input that can be claimed, so a claim that does not advance the
// boundary breaks the Feed contract.
func (i *inputs) claim(ctx context.Context, through int64) (Claim, error) {
	claim, err := i.feed.Claim(ctx, through)
	if err != nil {
		return Claim{}, err
	}
	if len(claim.Messages) == 0 {
		return Claim{}, errors.New("einorun: feed claimed no messages")
	}
	i.mu.Lock()
	defer i.mu.Unlock()
	if claim.EndSeq <= i.claimedSeq {
		return Claim{}, fmt.Errorf("einorun: feed claimed up to %d, not beyond the claimed %d", claim.EndSeq, i.claimedSeq)
	}
	i.claimedSeq = claim.EndSeq
	switch {
	case claim.EndSeq >= through || i.maxPushed > through:
		// Input pushed while claiming is already queued for the next turn.
		i.maxPushed = max(i.maxPushed, claim.EndSeq)
	default:
		// Input above a partial claim stays pending and is pushed again.
		i.maxPushed = claim.EndSeq
	}
	return claim, nil
}

// replay claims the checkpointed boundary again after a restart.
func (i *inputs) replay(ctx context.Context) (Claim, error) {
	i.mu.Lock()
	seq := i.claimedSeq
	i.mu.Unlock()
	claim, err := i.feed.Claim(ctx, seq)
	if err != nil {
		return Claim{}, err
	}
	if claim.EndSeq != seq || len(claim.Messages) == 0 {
		return Claim{}, errors.New("einorun: feed did not replay the claimed boundary")
	}
	return claim, nil
}

// finish decides whether the run ends with the turn's outcome: a fixed
// completion ends it at once; otherwise it ends unless new input is pending.
func (i *inputs) finish(ctx context.Context, turn *adk.TurnContext[trigger, *schema.AgenticMessage], completion *Completion, text string) (bool, error) {
	if completion != nil && completion.Fixed {
		i.mu.Lock()
		defer i.mu.Unlock()
		if !i.closed {
			i.closed = true
			i.loop.Stop()
		}
		return true, nil
	}
	if pending, err := i.pending(ctx, turn); pending || err != nil {
		return false, err
	}
	i.mu.Lock()
	defer i.mu.Unlock()
	if i.closed || i.maxPushed > i.claimedSeq {
		return false, nil
	}
	if completion == nil && text == "" {
		return false, ErrEmptyResponse
	}
	i.closed = true
	i.loop.Stop()
	return true, nil
}

// pending reports whether the turn was preempted or input is waiting to be
// claimed.
func (i *inputs) pending(ctx context.Context, turn *adk.TurnContext[trigger, *schema.AgenticMessage]) (bool, error) {
	if turn != nil {
		select {
		case <-turn.Preempted:
			return true, nil
		default:
		}
	}
	if err := i.poll(ctx, false); err != nil {
		return false, err
	}
	i.mu.Lock()
	defer i.mu.Unlock()
	return i.closed || i.maxPushed > i.claimedSeq, nil
}

// rerun pushes a correction signal: the turn runs again on the same input.
func (i *inputs) rerun() error {
	i.mu.Lock()
	defer i.mu.Unlock()
	if i.closed {
		return nil
	}
	if accepted, _ := i.loop.Push(trigger{correction: true}); !accepted {
		return errors.New("einorun: turn loop rejected the correction")
	}
	return nil
}

// boundary returns the claimed boundary.
func (i *inputs) boundary() int64 {
	i.mu.Lock()
	defer i.mu.Unlock()
	return i.claimedSeq
}
