// Package inmem provides in-memory implementations of einorun.Feed and
// einorun.Journal. They follow the same contracts as persistent
// implementations and suit tests, examples and runs that need no durability.
package inmem

import "github.com/runforyou-ai/einorun"

// Journal keeps a run's records in memory.
type Journal = einorun.MemoryJournal

// NewJournal returns an empty journal.
func NewJournal() *Journal { return einorun.NewMemoryJournal() }
