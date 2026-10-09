// Package inmem provides in-memory implementations of einorun.Feed and
// einorun.Journal. They follow the same contracts as persistent
// implementations and suit tests, examples and runs that need no durability.
package inmem

import (
	"context"
	"fmt"
	"slices"
	"sync"

	"github.com/runforyou-ai/einorun"
)

// Journal keeps a run's records in memory.
type Journal struct {
	mu         sync.Mutex
	calls      map[string]*einorun.ToolCall
	blocks     map[string]einorun.Block // calls are linked by Block.Call.ID
	state      []byte
	usage      einorun.Usage
	plan       []einorun.PlanTask
	completion *einorun.Completion
}

// NewJournal returns an empty journal.
func NewJournal() *Journal {
	return &Journal{calls: map[string]*einorun.ToolCall{}, blocks: map[string]einorun.Block{}}
}

// SaveStep applies a step.
func (j *Journal) SaveStep(_ context.Context, step einorun.Step) error {
	for _, call := range step.Changes.Calls {
		if err := call.Validate(); err != nil {
			return err
		}
	}
	j.mu.Lock()
	defer j.mu.Unlock()
	for _, id := range step.Changes.RemovedBlocks {
		delete(j.blocks, id)
	}
	for _, id := range step.Changes.RemovedCalls {
		delete(j.calls, id)
	}
	for _, call := range step.Changes.Calls {
		j.mergeLocked(call)
	}
	for _, block := range step.Changes.Blocks {
		// Blocks only link to their call.
		if block.Call != nil {
			block.Call = &einorun.ToolCall{ID: block.Call.ID}
		}
		j.blocks[block.ID] = block
	}
	j.state = slices.Clone(step.State)
	j.usage = step.Usage
	j.plan = slices.Clone(step.Plan)
	j.completion = cloneCompletion(step.Completion)
	return nil
}

// SaveToolCall merges one call.
func (j *Journal) SaveToolCall(_ context.Context, call einorun.ToolCall) error {
	if err := call.Validate(); err != nil {
		return err
	}
	j.mu.Lock()
	defer j.mu.Unlock()
	j.mergeLocked(call)
	return nil
}

// External applies a write made outside the runtime, as a host does when it
// dispatches a call to an external executor or settles it, with
// einorun.OverlayExternal semantics.
func (j *Journal) External(_ context.Context, update einorun.ToolCall) error {
	j.mu.Lock()
	defer j.mu.Unlock()
	stored, ok := j.calls[update.ID]
	if !ok {
		return fmt.Errorf("inmem: no call %s", update.ID)
	}
	merged := einorun.OverlayExternal(*stored, update)
	j.calls[update.ID] = &merged
	return nil
}

// Resume returns the saved checkpoint and the latest records, ready to hand
// back to a run.
func (j *Journal) Resume() einorun.Resume {
	j.mu.Lock()
	defer j.mu.Unlock()
	resume := einorun.Resume{State: slices.Clone(j.state), Plan: slices.Clone(j.plan), Completion: cloneCompletion(j.completion)}
	linked := map[string]bool{}
	for _, block := range j.blocks {
		if block.Call != nil {
			linked[block.Call.ID] = true
			if call, ok := j.calls[block.Call.ID]; ok {
				clone := call.Clone()
				block.Call = &clone
			}
		}
		resume.Blocks = append(resume.Blocks, block)
	}
	slices.SortFunc(resume.Blocks, func(a, b einorun.Block) int { return int(a.Position - b.Position) })
	for id, call := range j.calls {
		if !linked[id] {
			resume.Calls = append(resume.Calls, call.Clone())
		}
	}
	slices.SortFunc(resume.Calls, func(a, b einorun.ToolCall) int {
		if a.ID < b.ID {
			return -1
		}
		if a.ID > b.ID {
			return 1
		}
		return 0
	})
	return resume
}

// Call returns the stored record of a call.
func (j *Journal) Call(id string) (einorun.ToolCall, bool) {
	j.mu.Lock()
	defer j.mu.Unlock()
	call, ok := j.calls[id]
	if !ok {
		return einorun.ToolCall{}, false
	}
	return call.Clone(), true
}

// Usage returns the usage of the last saved step.
func (j *Journal) Usage() einorun.Usage {
	j.mu.Lock()
	defer j.mu.Unlock()
	return j.usage
}

// mergeLocked merges call into the stored record.
func (j *Journal) mergeLocked(call einorun.ToolCall) {
	merged := einorun.MergeCall(j.calls[call.ID], call)
	j.calls[call.ID] = &merged
}

// cloneCompletion copies c.
func cloneCompletion(c *einorun.Completion) *einorun.Completion {
	if c == nil {
		return nil
	}
	clone := *c
	clone.Value = slices.Clone(c.Value)
	return &clone
}
