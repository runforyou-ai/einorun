package inmem

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"sync"

	"github.com/runforyou-ai/einorun"
)

// Feed keeps a conversation in memory. Each appended message gets the next
// sequence number; claims return the whole conversation up to the boundary.
type Feed struct {
	mu       sync.Mutex
	messages []einorun.Message // messages[i] has sequence i+1
	consumed int64
	watchers map[chan struct{}]struct{}
}

// NewFeed returns an empty feed.
func NewFeed() *Feed {
	return &Feed{watchers: map[chan struct{}]struct{}{}}
}

// Append adds a message and returns its sequence number.
func (f *Feed) Append(message einorun.Message) int64 {
	f.mu.Lock()
	f.messages = append(f.messages, cloneMessage(message))
	seq := int64(len(f.messages))
	for ch := range f.watchers {
		select {
		case ch <- struct{}{}:
		default:
		}
	}
	f.mu.Unlock()
	return seq
}

// Consume marks input up to seq as consumed, as a host does when it delivers a
// run's result.
func (f *Feed) Consume(seq int64) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.consumed = max(f.consumed, seq)
}

// Watch subscribes to new input.
func (f *Feed) Watch(context.Context) (<-chan struct{}, func(), error) {
	ch := make(chan struct{}, 1)
	f.mu.Lock()
	f.watchers[ch] = struct{}{}
	f.mu.Unlock()
	var once sync.Once
	return ch, func() {
		once.Do(func() {
			f.mu.Lock()
			delete(f.watchers, ch)
			f.mu.Unlock()
		})
	}, nil
}

// Pending returns the latest sequence number above after, or 0.
func (f *Feed) Pending(_ context.Context, after int64) (int64, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if latest := int64(len(f.messages)); latest > max(after, f.consumed) {
		return latest, nil
	}
	return 0, nil
}

// Claim returns the conversation up to through, or up to the latest message
// when through is beyond it.
func (f *Feed) Claim(_ context.Context, through int64) (einorun.Claim, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	end := min(through, int64(len(f.messages)))
	if end <= 0 {
		return einorun.Claim{}, errors.New("inmem: nothing to claim")
	}
	if end <= f.consumed {
		return einorun.Claim{}, fmt.Errorf("inmem: input up to %d is already consumed", f.consumed)
	}
	messages := make([]einorun.Message, end)
	for i, m := range f.messages[:end] {
		messages[i] = cloneMessage(m)
	}
	return einorun.Claim{Messages: messages, EndSeq: end}, nil
}

// cloneMessage copies the media reference and meta of m.
func cloneMessage(m einorun.Message) einorun.Message {
	if m.Media != nil {
		media := *m.Media
		m.Media = &media
	}
	m.Meta = maps.Clone(m.Meta)
	return m
}
