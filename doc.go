// Package einorun runs agents built on Eino as durable, resumable runs.
//
// A run reads its input from a Feed, records its process in a Journal and can
// be resumed from its last checkpoint after a crash or after waiting for an
// external result. Hosts keep their business rules (instructions, tools,
// approvals, external executors) and plug them in through tool specs, guards
// and extensions; einorun owns the run's state machine.
//
// See docs/design.md in the repository for the design and its invariants.
package einorun
