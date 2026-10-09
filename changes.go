package einorun

import (
	"maps"
	"slices"
	"sync"
	"time"

	"github.com/runforyou-ai/einorun/stream"
)

// changeSet tracks the process changes since the last successful save. Each
// registration gets an increasing sequence number; a successful save clears
// only the entries of its snapshot that did not change again since.
type changeSet struct {
	seq           uint64
	blocks        map[string]uint64
	calls         map[string]uint64
	removedBlocks map[string]uint64
	removedCalls  map[string]uint64
}

// changeMarks are the sequence numbers of a snapshot's entries.
type changeMarks struct {
	blocks, calls, removedBlocks, removedCalls map[string]uint64
}

// newChangeSet returns an empty change set.
func newChangeSet() changeSet {
	return changeSet{blocks: map[string]uint64{}, calls: map[string]uint64{}, removedBlocks: map[string]uint64{}, removedCalls: map[string]uint64{}}
}

// block registers a new or changed block.
func (c *changeSet) block(id string) {
	c.seq++
	c.blocks[id] = c.seq
	delete(c.removedBlocks, id)
}

// call registers a new or changed call.
func (c *changeSet) call(id string) {
	c.seq++
	c.calls[id] = c.seq
	delete(c.removedCalls, id)
}

// remove registers the removal of a block and its call.
func (c *changeSet) remove(block Block) {
	c.seq++
	delete(c.blocks, block.ID)
	c.removedBlocks[block.ID] = c.seq
	if block.Call != nil {
		delete(c.calls, block.Call.ID)
		c.removedCalls[block.Call.ID] = c.seq
	}
}

// marks returns the current sequence numbers.
func (c *changeSet) marks() changeMarks {
	return changeMarks{blocks: maps.Clone(c.blocks), calls: maps.Clone(c.calls), removedBlocks: maps.Clone(c.removedBlocks), removedCalls: maps.Clone(c.removedCalls)}
}

// commit clears the entries saved in a snapshot that did not change since.
// Calls carry revisions, so a snapshot that raced with a later SaveToolCall
// cannot overwrite it; their entries are cleared like the others.
func (c *changeSet) commit(saved changeMarks) {
	for _, pair := range [][2]map[string]uint64{
		{c.blocks, saved.blocks}, {c.calls, saved.calls}, {c.removedBlocks, saved.removedBlocks}, {c.removedCalls, saved.removedCalls},
	} {
		maps.DeleteFunc(pair[0], func(id string, seq uint64) bool { return pair[1][id] == seq })
	}
}

// publisher merges stream operations over a fixed period and hands them to
// the sink serially.
type publisher struct {
	stream   string
	sequence int64
	sink     func(stream.Delta)
	mu       sync.Mutex
	pending  []stream.Operation
	stop     chan struct{}
	done     chan struct{}
}

// publishInterval is how often operations are merged into a delta.
const publishInterval = 50 * time.Millisecond

// start starts publishing; without a sink nothing is published.
func (p *publisher) start() {
	if p.sink == nil {
		return
	}
	p.stop, p.done = make(chan struct{}), make(chan struct{})
	go func() {
		defer close(p.done)
		ticker := time.NewTicker(publishInterval)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				p.flush()
			case <-p.stop:
				p.flush()
				return
			}
		}
	}()
}

// close publishes the remaining operations and waits for the publisher to
// stop. The sink is not called afterwards.
func (p *publisher) close() {
	if p.stop == nil {
		return
	}
	close(p.stop)
	<-p.done
	p.stop = nil
}

// add queues operations.
func (p *publisher) add(ops ...stream.Operation) {
	if p.sink == nil || len(ops) == 0 {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.pending = append(p.pending, ops...)
}

// flush publishes the queued operations as the next delta.
func (p *publisher) flush() {
	p.mu.Lock()
	if len(p.pending) == 0 {
		p.mu.Unlock()
		return
	}
	delta := stream.Delta{Stream: p.stream, Base: p.sequence, Sequence: p.sequence + 1, Operations: stream.MergeOperations(p.pending)}
	p.sequence++
	p.pending = nil
	p.mu.Unlock()
	p.sink(delta)
}

// view returns the display form of a block.
func (r *recorder) view(b Block) *stream.Block {
	view := &stream.Block{ID: b.ID, Position: b.Position, ModelCallID: b.ModelCallID, Kind: stream.BlockKind(b.Kind), Text: b.Text}
	if call := b.Call; call != nil {
		view.Call = &stream.CallView{CallID: call.CallID, Name: call.Name, Status: string(call.Status), StartedAt: call.StartedAt, CompletedAt: call.CompletedAt}
		if r.describe != nil {
			view.Call.Description = r.describe(*call)
		}
		view.Call.Activity = r.activity[call.ID]
	}
	return view
}

// sortedKeys returns the keys of m in order.
func sortedKeys(m map[string]uint64) []string {
	return slices.Sorted(maps.Keys(m))
}
