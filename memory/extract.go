package memory

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"slices"
	"strings"
	"unicode/utf8"

	"github.com/runforyou-ai/einorun/llm"
)

// ExtractLimits bound the entries Extract accepts. Zero values take the
// defaults.
type ExtractLimits struct {
	// Name, Description and Body are the most characters of each field
	// (default 60, 200 and 2000).
	Name        int
	Description int
	Body        int
}

// withDefaults fills zero limits.
func (l ExtractLimits) withDefaults() ExtractLimits {
	l.Name = cmp.Or(l.Name, 60)
	l.Description = cmp.Or(l.Description, 200)
	l.Body = cmp.Or(l.Body, 2000)
	return l
}

// ExtractRequest is one extraction.
type ExtractRequest struct {
	// Instruction replaces the default criteria of what to remember and what
	// not to. The task, rules and output format always follow it.
	Instruction string
	// Entries are the existing memories, given to the model in full.
	Entries []Entry
	// Earlier is conversation already extracted, for context only; Recent is
	// the conversation to extract from.
	Earlier []Message
	Recent  []Message
	// Language selects the default text.
	Language llm.Language
	Limits   ExtractLimits
}

// Changes are the memory changes of an extraction. A key appears in at most
// one of Saved and Deleted.
type Changes struct {
	// Saved are new entries and complete rewrites of existing ones, by key.
	// UpdatedAt is left for the host.
	Saved []Entry
	// Deleted are keys of existing entries to delete.
	Deleted []string
	// Skipped are entries the model proposed that break the limits; they are
	// not in Saved.
	Skipped []Skipped
}

// Skipped is a proposed entry that was not accepted.
type Skipped struct {
	Entry  Entry
	Reason string
}

// keyPattern is the form of keys Extract writes.
var keyPattern = regexp.MustCompile(`^[a-z0-9]+(-[a-z0-9]+)*$`)

// maxKeyLength is the longest key Extract writes.
const maxKeyLength = 64

// extraction is the model's answer.
type extraction struct {
	Save   []proposed `json:"save"`
	Delete []string   `json:"delete"`
}

// proposed is an entry the model proposes to save.
type proposed struct {
	Key         string `json:"key"`
	Name        string `json:"name"`
	Description string `json:"description"`
	Body        string `json:"body"`
}

// Extract asks the model which memories to save and delete given a recent
// conversation, in one structured call with thinking off. It returns the
// changes and the usage, also on error. Nothing is written: the host applies
// the changes. Without recent messages the model is not called.
func Extract(ctx context.Context, factory llm.ModelFactory, request ExtractRequest) (Changes, llm.Usage, error) {
	if len(request.Recent) == 0 {
		return Changes{}, llm.Usage{}, nil
	}
	text := catalogFor(request.Language)
	limits := request.Limits.withDefaults()
	criteria := cmp.Or(request.Instruction, text.extractCriteria)
	instruction := criteria + "\n\n" + fmt.Sprintf(text.extractRules, limits.Name, limits.Description, limits.Body)
	input, err := extractionInput(text, request)
	if err != nil {
		return Changes{}, llm.Usage{}, err
	}
	answer, usage, err := llm.GenerateObject[extraction](ctx, factory, llm.GenerateRequest{
		Instruction: instruction, Input: input, Language: request.Language,
	})
	if err != nil {
		return Changes{}, usage, err
	}
	existing := map[string]bool{}
	for _, e := range request.Entries {
		existing[e.Key] = true
	}
	var changes Changes
	saved := map[string]int{}
	for _, p := range answer.Save {
		entry := Entry{Key: strings.TrimSpace(p.Key), Name: strings.TrimSpace(p.Name),
			Description: strings.TrimSpace(p.Description), Body: strings.TrimSpace(p.Body)}
		if reason := problem(entry, existing[entry.Key], limits); reason != "" {
			changes.Skipped = append(changes.Skipped, Skipped{Entry: entry, Reason: reason})
			continue
		}
		// A later proposal for the same key replaces the earlier one.
		if i, ok := saved[entry.Key]; ok {
			changes.Saved[i] = entry
			continue
		}
		saved[entry.Key] = len(changes.Saved)
		changes.Saved = append(changes.Saved, entry)
	}
	for _, key := range answer.Delete {
		key = strings.TrimSpace(key)
		if _, ok := saved[key]; ok || !existing[key] || slices.Contains(changes.Deleted, key) {
			continue
		}
		changes.Deleted = append(changes.Deleted, key)
	}
	return changes, usage, nil
}

// problem returns why a proposed entry is not accepted, or "". Existing keys
// are accepted as they are.
func problem(e Entry, exists bool, limits ExtractLimits) string {
	switch {
	case !exists && (len(e.Key) > maxKeyLength || !keyPattern.MatchString(e.Key)):
		return fmt.Sprintf("key must be lowercase letters, digits and hyphens, at most %d characters", maxKeyLength)
	case e.Name == "" || utf8.RuneCountInString(e.Name) > limits.Name:
		return fmt.Sprintf("name must be 1 to %d characters", limits.Name)
	case e.Description == "" || utf8.RuneCountInString(e.Description) > limits.Description:
		return fmt.Sprintf("description must be 1 to %d characters", limits.Description)
	case e.Body == "" || utf8.RuneCountInString(e.Body) > limits.Body:
		return fmt.Sprintf("body must be 1 to %d characters", limits.Body)
	}
	return ""
}

// extractionInput renders the existing memories and the conversation.
func extractionInput(text *catalog, request ExtractRequest) (string, error) {
	var input strings.Builder
	input.WriteString(text.extractEntries + "\n")
	if len(request.Entries) == 0 {
		input.WriteString(text.extractNone + "\n")
	} else {
		entries := make([]proposed, len(request.Entries))
		for i, e := range request.Entries {
			entries[i] = proposed{Key: e.Key, Name: e.Name, Description: e.Description, Body: e.Body}
		}
		encoded, err := json.Marshal(entries)
		if err != nil {
			return "", fmt.Errorf("memory: encode entries: %w", err)
		}
		input.Write(encoded)
		input.WriteString("\n")
	}
	for _, part := range []struct {
		title    string
		messages []Message
	}{{text.extractEarlier, request.Earlier}, {text.extractRecent, request.Recent}} {
		messages := part.messages
		if messages == nil {
			messages = []Message{}
		}
		encoded, err := json.Marshal(messages)
		if err != nil {
			return "", fmt.Errorf("memory: encode messages: %w", err)
		}
		input.WriteString("\n" + part.title + "\n")
		input.Write(encoded)
		input.WriteString("\n")
	}
	return input.String(), nil
}
