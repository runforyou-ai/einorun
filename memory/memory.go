// Package memory gives runs long-term memory: Recall is an extension that
// picks the memories relevant to each claimed turn and shows them to the
// model, and Extract derives memory changes from a conversation. The host
// stores the entries; this package never writes them.
package memory

import (
	"context"
	"time"

	"github.com/runforyou-ai/einorun"
)

// Entry is one memory.
type Entry struct {
	// Key identifies the entry. Extract writes keys of lowercase letters,
	// digits and hyphens; entries from a Source may use any non-empty key.
	Key string `json:"key"`
	// Name is a short title and Description one sentence; together they
	// decide whether the entry is relevant.
	Name        string `json:"name"`
	Description string `json:"description"`
	Body        string `json:"body"`
	// UpdatedAt orders entries, newest first.
	UpdatedAt time.Time `json:"updatedAt,omitzero"`
}

// Source provides the memories a run may recall. It is called once per run.
type Source interface {
	Memories(ctx context.Context, run einorun.RunScope) ([]Entry, error)
}

// SourceFunc adapts a function to Source.
type SourceFunc func(ctx context.Context, run einorun.RunScope) ([]Entry, error)

// Memories calls f.
func (f SourceFunc) Memories(ctx context.Context, run einorun.RunScope) ([]Entry, error) {
	return f(ctx, run)
}

// Message is one message of the conversation memories are extracted from.
type Message struct {
	Role    einorun.Role `json:"role"`
	Content string       `json:"content"`
}
