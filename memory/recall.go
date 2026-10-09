package memory

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"slices"
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/cloudwego/eino/adk"
	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"

	"github.com/runforyou-ai/einorun"
	"github.com/runforyou-ai/einorun/llm"
)

// Limits bound what Recall puts in front of the model. Zero values take the
// defaults.
type Limits struct {
	// IndexEntries and IndexBytes bound the index in the instruction
	// (default 200 entries, 16 KiB); the newest entries are listed.
	IndexEntries int
	IndexBytes   int
	// Candidates is the number of newest entries considered for selection
	// (default 200).
	Candidates int
	// Selected is the most memories shown for a turn (default 5).
	Selected int
	// EntryBytes bounds one shown memory (default 8 KiB) and SelectedBytes
	// all memories shown for a turn (default 32 KiB).
	EntryBytes    int
	SelectedBytes int
	// ConversationRunes bounds the conversation the selection sees, newest
	// messages first (default 6000).
	ConversationRunes int
}

// withDefaults fills zero limits.
func (l Limits) withDefaults() Limits {
	l.IndexEntries = cmp.Or(l.IndexEntries, 200)
	l.IndexBytes = cmp.Or(l.IndexBytes, 16<<10)
	l.Candidates = cmp.Or(l.Candidates, 200)
	l.Selected = cmp.Or(l.Selected, 5)
	l.EntryBytes = cmp.Or(l.EntryBytes, 8<<10)
	l.SelectedBytes = cmp.Or(l.SelectedBytes, 32<<10)
	l.ConversationRunes = cmp.Or(l.ConversationRunes, 6000)
	return l
}

// RecallOptions configure Recall.
type RecallOptions struct {
	// Name is the extension's name (default "memory").
	Name string
	// Instruction replaces the default text that tells the agent how to
	// treat memories. The index always follows it.
	Instruction string
	Limits      Limits
}

// Recall returns an extension that gives the main agent the run's memories.
// Each run loads the entries from source once; the instruction lists an
// index of them. On every claim of new input the run's model picks the
// entries relevant to the conversation, and each model call of the turn sees
// them in a reminder before the latest input. The reminder is not part of
// the history or the checkpoint; the picked keys are. A Source error fails
// the run; a failed selection shows no memories for that turn. The selection
// usage counts toward the run.
//
// The index is part of the instruction: hosts that bound the context should
// include it in ContextPolicy.InstructionTokens.
func Recall(source Source, options RecallOptions) einorun.Extension {
	options.Name = cmp.Or(options.Name, "memory")
	options.Limits = options.Limits.withDefaults()
	return &recallPrototype{source: source, options: options}
}

// recallPrototype creates the per-run instances.
type recallPrototype struct {
	source  Source
	options RecallOptions
}

// Name returns the extension's name.
func (p *recallPrototype) Name() string { return p.options.Name }

// Instance loads the run's memories.
func (p *recallPrototype) Instance(ctx context.Context, run einorun.RunScope) (einorun.Extension, error) {
	loaded, err := p.source.Memories(ctx, run)
	if err != nil {
		return nil, fmt.Errorf("memory: load memories: %w", err)
	}
	entries := make([]Entry, 0, len(loaded))
	seen := map[string]bool{}
	for _, e := range loaded {
		if e.Key == "" || seen[e.Key] {
			continue
		}
		seen[e.Key] = true
		entries = append(entries, e)
	}
	slices.SortStableFunc(entries, func(a, b Entry) int { return b.UpdatedAt.Compare(a.UpdatedAt) })
	text := catalogFor(run.Language)
	return &recall{
		name: p.options.Name, usage: cmp.Or(p.options.Instruction, text.usage), limits: p.options.Limits,
		text: text, language: run.Language, model: run.Model, entries: entries,
	}, nil
}

// recall is the extension instance of one run.
type recall struct {
	name     string
	usage    string
	limits   Limits
	text     *catalog
	language llm.Language
	model    llm.ModelFactory
	entries  []Entry // newest first

	mu       sync.Mutex
	selected []string
	used     llm.Usage
}

// Name returns the extension's name.
func (r *recall) Name() string { return r.name }

// Instruction adds the usage text and the index to the main agent.
func (r *recall) Instruction(scope einorun.AgentScope) string {
	if !scope.Main {
		return ""
	}
	var b strings.Builder
	b.WriteString(r.usage + "\n\n" + r.text.indexTitle + "\n")
	if len(r.entries) == 0 {
		b.WriteString(r.text.indexEmpty)
		return b.String()
	}
	size := 0
	for i, e := range r.entries {
		line := fmt.Sprintf("- %s: %s — %s\n", e.Key, e.Name, e.Description)
		if i == r.limits.IndexEntries || size+len(line) > r.limits.IndexBytes {
			break
		}
		b.WriteString(line)
		size += len(line)
	}
	return strings.TrimRight(b.String(), "\n")
}

// selection is the model's pick.
type selection struct {
	Keys []string `json:"keys"`
}

// OnClaim picks the memories relevant to the claimed conversation.
func (r *recall) OnClaim(ctx context.Context, claim einorun.Claim) {
	if len(r.entries) == 0 || r.model == nil {
		return
	}
	candidates := r.entries[:min(len(r.entries), r.limits.Candidates)]
	var input strings.Builder
	input.WriteString(r.text.selectMemories + "\n")
	for _, e := range candidates {
		fmt.Fprintf(&input, "- %s: %s — %s\n", e.Key, e.Name, e.Description)
	}
	input.WriteString("\n" + r.text.selectConversation + "\n" + conversation(claim.Messages, r.limits.ConversationRunes))
	picked, usage, err := llm.GenerateObject[selection](ctx, r.model, llm.GenerateRequest{
		Instruction: fmt.Sprintf(r.text.selectInstruction, r.limits.Selected), Input: input.String(), Language: r.language,
	})
	var keys []string
	if err != nil {
		slog.WarnContext(ctx, "einorun: selecting memories failed, showing none this turn", "error", err)
	} else {
		known := map[string]bool{}
		for _, e := range candidates {
			known[e.Key] = true
		}
		for _, key := range picked.Keys {
			if known[key] && !slices.Contains(keys, key) && len(keys) < r.limits.Selected {
				keys = append(keys, key)
			}
		}
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.used.Add(usage)
	r.selected = keys
}

// conversation renders the newest messages within budget runes, oldest
// first; the latest message is kept, shortened when needed.
func conversation(messages []einorun.Message, budget int) string {
	var lines []string
	total := 0
	for i := len(messages) - 1; i >= 0; i-- {
		line := string(messages[i].Role) + ": " + messages[i].Content
		n := utf8.RuneCountInString(line)
		if total+n > budget {
			if len(lines) == 0 {
				lines = append(lines, string([]rune(line)[:budget]))
			}
			break
		}
		lines = append(lines, line)
		total += n
	}
	slices.Reverse(lines)
	return strings.Join(lines, "\n")
}

// ModelMiddlewares shows the selected memories to the main agent's model
// calls.
func (r *recall) ModelMiddlewares(scope einorun.AgentScope) []adk.TypedChatModelAgentMiddleware[*schema.AgenticMessage] {
	if !scope.Main {
		return nil
	}
	return []adk.TypedChatModelAgentMiddleware[*schema.AgenticMessage]{&reminderMiddleware{recall: r}}
}

// reminder renders the selected memories, or "" when there are none.
func (r *recall) reminder() string {
	r.mu.Lock()
	keys := slices.Clone(r.selected)
	r.mu.Unlock()
	if len(keys) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("<memory-reminder>\n" + r.text.reminder + "\n")
	size, shown := 0, 0
	for _, key := range keys {
		i := slices.IndexFunc(r.entries, func(e Entry) bool { return e.Key == key })
		if i < 0 {
			continue
		}
		e := r.entries[i]
		body := e.Description + "\n" + e.Body
		if len(body) > r.limits.EntryBytes {
			body = truncate(body, r.limits.EntryBytes) + "\n" + r.text.truncated
		}
		item := fmt.Sprintf("<memory key=%q name=%q>\n%s\n</memory>\n", e.Key, e.Name, body)
		if size+len(item) > r.limits.SelectedBytes {
			break
		}
		b.WriteString(item)
		size += len(item)
		shown++
	}
	if shown == 0 {
		return ""
	}
	b.WriteString("</memory-reminder>")
	return b.String()
}

// truncate cuts s to at most n bytes on a rune boundary.
func truncate(s string, n int) string {
	for n > 0 && !utf8.RuneStart(s[n]) {
		n--
	}
	return s[:n]
}

// Save keeps the selected keys.
func (r *recall) Save() (json.RawMessage, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return json.Marshal(recallState{Selected: r.selected})
}

// Restore restores the selected keys.
func (r *recall) Restore(data json.RawMessage) error {
	var s recallState
	if err := json.Unmarshal(data, &s); err != nil {
		return err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.selected = s.Selected
	return nil
}

// recallState is the checkpoint state of a recall instance.
type recallState struct {
	Selected []string `json:"selected,omitempty"`
}

// Usage returns the usage of the selections.
func (r *recall) Usage() llm.Usage {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.used
}

// reminderMiddleware adds the reminder to every model call.
type reminderMiddleware struct {
	adk.TypedBaseChatModelAgentMiddleware[*schema.AgenticMessage]
	recall *recall
}

// WrapModel wraps the model.
func (m *reminderMiddleware) WrapModel(_ context.Context, base model.BaseModel[*schema.AgenticMessage], _ *adk.TypedModelContext[*schema.AgenticMessage]) (model.BaseModel[*schema.AgenticMessage], error) {
	return &reminderModel{base: base, recall: m.recall}, nil
}

// reminderModel puts the reminder before the latest input of each call.
type reminderModel struct {
	base   model.BaseModel[*schema.AgenticMessage]
	recall *recall
}

// Generate calls the model with the reminder.
func (m *reminderModel) Generate(ctx context.Context, input []*schema.AgenticMessage, opts ...model.Option) (*schema.AgenticMessage, error) {
	return m.base.Generate(ctx, m.withReminder(input), opts...)
}

// Stream calls the model with the reminder.
func (m *reminderModel) Stream(ctx context.Context, input []*schema.AgenticMessage, opts ...model.Option) (*schema.StreamReader[*schema.AgenticMessage], error) {
	return m.base.Stream(ctx, m.withReminder(input), opts...)
}

// withReminder returns input with the reminder before the latest user
// message that is not a tool result, or after the system instruction when
// there is none.
func (m *reminderModel) withReminder(input []*schema.AgenticMessage) []*schema.AgenticMessage {
	text := m.recall.reminder()
	if text == "" {
		return input
	}
	at := -1
	for i := len(input) - 1; i >= 0; i-- {
		if input[i].Role == schema.AgenticRoleTypeUser && !slices.ContainsFunc(input[i].ContentBlocks, func(b *schema.ContentBlock) bool {
			return b != nil && b.Type == schema.ContentBlockTypeFunctionToolResult
		}) {
			at = i
			break
		}
	}
	if at < 0 {
		at = 0
		for at < len(input) && input[at].Role == schema.AgenticRoleTypeSystem {
			at++
		}
	}
	return slices.Concat(input[:at:at], []*schema.AgenticMessage{schema.UserAgenticMessage(text)}, input[at:])
}
