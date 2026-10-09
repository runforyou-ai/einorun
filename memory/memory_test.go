package memory_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/schema"

	"github.com/runforyou-ai/einorun"
	"github.com/runforyou-ai/einorun/inmem"
	"github.com/runforyou-ai/einorun/llm"
	"github.com/runforyou-ai/einorun/memory"
)

// fake is a model whose structured calls answer with answers in order and
// whose other calls answer with steps in order; it records the inputs.
type fake struct {
	mu         sync.Mutex
	answers    []string
	steps      []func() *schema.AgenticMessage
	structured [][]*schema.AgenticMessage
	agent      [][]*schema.AgenticMessage
}

func (f *fake) factory(_ context.Context, options llm.ModelOptions) (model.AgenticModel, error) {
	return &fakeModel{fake: f, structured: options.Output != nil}, nil
}

type fakeModel struct {
	fake       *fake
	structured bool
}

func (m *fakeModel) Generate(_ context.Context, input []*schema.AgenticMessage, _ ...model.Option) (*schema.AgenticMessage, error) {
	f := m.fake
	f.mu.Lock()
	defer f.mu.Unlock()
	if m.structured {
		f.structured = append(f.structured, input)
		if len(f.answers) == 0 {
			return nil, errors.New("no answer left")
		}
		answer := f.answers[0]
		f.answers = f.answers[1:]
		return text(answer, 5, 1), nil
	}
	f.agent = append(f.agent, input)
	if len(f.steps) == 0 {
		return nil, errors.New("no step left")
	}
	step := f.steps[0]
	f.steps = f.steps[1:]
	return step(), nil
}

func (m *fakeModel) Stream(ctx context.Context, input []*schema.AgenticMessage, opts ...model.Option) (*schema.StreamReader[*schema.AgenticMessage], error) {
	message, err := m.Generate(ctx, input, opts...)
	if err != nil {
		return nil, err
	}
	return schema.StreamReaderFromArray([]*schema.AgenticMessage{message}), nil
}

func text(s string, in, out int) *schema.AgenticMessage {
	return &schema.AgenticMessage{Role: schema.AgenticRoleTypeAssistant,
		ContentBlocks: []*schema.ContentBlock{schema.NewContentBlock(&schema.AssistantGenText{Text: s})},
		ResponseMeta:  &schema.AgenticResponseMeta{TokenUsage: &schema.TokenUsage{PromptTokens: in, CompletionTokens: out}}}
}

func ok() func() *schema.AgenticMessage {
	return func() *schema.AgenticMessage { return text("ok", 10, 2) }
}

func callTool(name string) func() *schema.AgenticMessage {
	return func() *schema.AgenticMessage {
		return &schema.AgenticMessage{Role: schema.AgenticRoleTypeAssistant, ContentBlocks: []*schema.ContentBlock{
			schema.NewContentBlock(&schema.FunctionToolCall{CallID: "c1", Name: name, Arguments: "{}"})},
			ResponseMeta: &schema.AgenticResponseMeta{TokenUsage: &schema.TokenUsage{PromptTokens: 10, CompletionTokens: 2}}}
	}
}

// messageText returns the text of a message.
func messageText(m *schema.AgenticMessage) string {
	var parts []string
	for _, b := range m.ContentBlocks {
		switch {
		case b.UserInputText != nil:
			parts = append(parts, b.UserInputText.Text)
		case b.FunctionToolResult != nil:
			parts = append(parts, "[result]")
		}
	}
	return strings.Join(parts, "\n")
}

var entries = []memory.Entry{
	{Key: "tone", Name: "Tone", Description: "How replies should sound", Body: "Keep replies short.", UpdatedAt: time.Date(2026, 1, 2, 0, 0, 0, 0, time.UTC)},
	{Key: "stack", Name: "Stack", Description: "Tools in use", Body: "Uses Go.", UpdatedAt: time.Date(2026, 1, 3, 0, 0, 0, 0, time.UTC)},
}

func source(list []memory.Entry) memory.Source {
	return memory.SourceFunc(func(context.Context, einorun.RunScope) ([]memory.Entry, error) { return list, nil })
}

func runWith(t *testing.T, f *fake, ext einorun.Extension) einorun.Result {
	t.Helper()
	feed := inmem.NewFeed()
	feed.Append(einorun.Message{ID: "m1", Revision: "1", Role: einorun.RoleUser, Content: "write me a reply"})
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	result, err := einorun.New(einorun.Config{Language: llm.English}).Run(ctx, einorun.Request{
		RunID: "r1", Instruction: "be kind", Model: einorun.Model{New: f.factory}, Feed: feed,
		Extensions: []einorun.Extension{ext},
	})
	if err != nil {
		t.Fatal(err)
	}
	return result
}

func TestRecallShowsSelectedMemories(t *testing.T) {
	f := &fake{answers: []string{`{"keys":["tone","unknown"]}`}, steps: []func() *schema.AgenticMessage{ok()}}
	result := runWith(t, f, memory.Recall(source(entries), memory.RecallOptions{}))
	if len(f.structured) != 1 || !strings.Contains(messageText(f.structured[0][1]), "user: write me a reply") {
		t.Fatalf("selection input %v", f.structured)
	}
	input := f.agent[0]
	system := messageText(input[0])
	if !strings.Contains(system, "be kind") || !strings.Contains(system, "- \"stack\": Stack — Tools in use\n- \"tone\": Tone") {
		t.Fatalf("instruction %q", system)
	}
	reminder := messageText(input[len(input)-2])
	if !strings.Contains(reminder, `<memory key="tone" name="Tone">`) || !strings.Contains(reminder, "Keep replies short.") || strings.Contains(reminder, "Uses Go.") {
		t.Fatalf("reminder %q", reminder)
	}
	if messageText(input[len(input)-1]) != "write me a reply" {
		t.Fatalf("latest input %q", messageText(input[len(input)-1]))
	}
	if result.Usage.Total != 12+6 {
		t.Fatalf("usage %+v", result.Usage)
	}
}

func TestRecallKeepsTheReminderBeforeTheInput(t *testing.T) {
	f := &fake{answers: []string{`{"keys":["tone"]}`}, steps: []func() *schema.AgenticMessage{callTool("look"), ok()}}
	feed := inmem.NewFeed()
	feed.Append(einorun.Message{ID: "m1", Revision: "1", Role: einorun.RoleUser, Content: "write me a reply"})
	// The second call is the last of the budget and ends with the runtime's
	// closing notice, which is not input.
	_, err := einorun.New(einorun.Config{Language: llm.English}).Run(context.Background(), einorun.Request{
		Model: einorun.Model{New: f.factory}, Feed: feed, Limits: einorun.Limits{MaxIterations: 2},
		Extensions: []einorun.Extension{memory.Recall(source(entries), memory.RecallOptions{})},
		Tools:      []einorun.ToolSpec{{Tool: &lookTool{}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	second := f.agent[1]
	reminders := 0
	at := -1
	for i, m := range second {
		if m.Role == schema.AgenticRoleTypeUser && strings.Contains(messageText(m), "<memory-reminder>") {
			reminders++
			at = i
		}
	}
	if reminders != 1 || messageText(second[at+1]) != "write me a reply" || messageText(second[at+3]) != "[result]" || at+4 != len(second)-1 {
		for _, m := range second {
			t.Logf("%s %q", m.Role, messageText(m))
		}
		t.Fatalf("second input %d %d", reminders, at)
	}
}

func TestExtractNeverDeletesARewrite(t *testing.T) {
	f := &fake{answers: []string{`{"save":[{"key":"stack","name":"Stack","description":"Tools","body":""}],"delete":["stack"]}`}}
	changes, _, err := memory.Extract(context.Background(), f.factory, memory.ExtractRequest{
		Entries: entries, Recent: []memory.Message{{Role: einorun.RoleUser, Content: "we moved to Rust"}},
	})
	if err != nil || len(changes.Deleted) != 0 || len(changes.Saved) != 0 || len(changes.Skipped) != 1 {
		t.Fatalf("changes %+v %v", changes, err)
	}
}

func TestRecallKeepsKeysAsGiven(t *testing.T) {
	spaced := []memory.Entry{{Key: " tone ", Name: "Tone", Description: "d", Body: "spaced body"}, {Key: "tone", Name: "Tone", Description: "d", Body: "plain body"}}
	f := &fake{answers: []string{`{"keys":[" tone "]}`}, steps: []func() *schema.AgenticMessage{ok()}}
	runWith(t, f, memory.Recall(source(spaced), memory.RecallOptions{}))
	if candidates := messageText(f.structured[0][1]); !strings.Contains(candidates, `- " tone ": Tone`) || !strings.Contains(candidates, `- "tone": Tone`) {
		t.Fatalf("candidates %q", candidates)
	}
	reminder := messageText(f.agent[0][len(f.agent[0])-2])
	if !strings.Contains(reminder, "spaced body") || strings.Contains(reminder, "plain body") {
		t.Fatalf("reminder %q", reminder)
	}
}

func TestRecallFailedSelectionShowsNone(t *testing.T) {
	f := &fake{answers: []string{"not json", "still not json"}, steps: []func() *schema.AgenticMessage{ok()}}
	result := runWith(t, f, memory.Recall(source(entries), memory.RecallOptions{}))
	for _, m := range f.agent[0] {
		if m.Role == schema.AgenticRoleTypeUser && strings.Contains(messageText(m), "<memory-reminder>") {
			t.Fatal("reminder after a failed selection")
		}
	}
	if result.Usage.Total != 12+12 {
		t.Fatalf("usage %+v", result.Usage)
	}
}

func TestRecallWithoutMemories(t *testing.T) {
	f := &fake{steps: []func() *schema.AgenticMessage{ok()}}
	runWith(t, f, memory.Recall(source(nil), memory.RecallOptions{Instruction: "Custom usage."}))
	if len(f.structured) != 0 {
		t.Fatal("selection without memories")
	}
	if system := messageText(f.agent[0][0]); !strings.Contains(system, "Custom usage.\n\n## Memory index\n(none)") {
		t.Fatalf("instruction %q", system)
	}
}

func TestRecallSourceErrorFailsTheRun(t *testing.T) {
	failing := memory.SourceFunc(func(context.Context, einorun.RunScope) ([]memory.Entry, error) {
		return nil, errors.New("store down")
	})
	feed := inmem.NewFeed()
	feed.Append(einorun.Message{ID: "m1", Revision: "1", Role: einorun.RoleUser, Content: "hi"})
	f := &fake{}
	_, err := einorun.New(einorun.Config{}).Run(context.Background(), einorun.Request{
		Model: einorun.Model{New: f.factory}, Feed: feed, Extensions: []einorun.Extension{memory.Recall(failing, memory.RecallOptions{})},
	})
	if err == nil || !strings.Contains(err.Error(), "store down") {
		t.Fatalf("err %v", err)
	}
}

func TestRecallStateRoundTrip(t *testing.T) {
	ctx := context.Background()
	f := &fake{answers: []string{`{"keys":["stack"]}`}}
	prototype := memory.Recall(source(entries), memory.RecallOptions{})
	scope := einorun.RunScope{Language: llm.English, Model: f.factory}
	first, err := prototype.(einorun.Instantiable).Instance(ctx, scope)
	if err != nil {
		t.Fatal(err)
	}
	first.(einorun.ClaimObserver).OnClaim(ctx, einorun.Claim{Messages: []einorun.Message{{Role: einorun.RoleUser, Content: "go?"}}})
	saved, err := first.(einorun.Stateful).Save()
	if err != nil {
		t.Fatal(err)
	}
	var state map[string][]string
	if err := json.Unmarshal(saved, &state); err != nil || fmt.Sprint(state["selected"]) != "[stack]" {
		t.Fatalf("state %s %v", saved, err)
	}
	second, err := prototype.(einorun.Instantiable).Instance(ctx, scope)
	if err != nil {
		t.Fatal(err)
	}
	if err := second.(einorun.Stateful).Restore(saved); err != nil {
		t.Fatal(err)
	}
	if second.(einorun.UsageReporter).Usage().Total != 0 || first.(einorun.UsageReporter).Usage().Total != 6 {
		t.Fatal("usage is per instance")
	}
	if got := second.(einorun.ModelMiddleware).ModelMiddlewares(einorun.AgentScope{Name: "sub"}); got != nil {
		t.Fatal("sub-agents get no reminder")
	}
}

// lookTool is a tool without arguments.
type lookTool struct{}

func (*lookTool) Info(context.Context) (*schema.ToolInfo, error) {
	return &schema.ToolInfo{Name: "look", Desc: "look"}, nil
}

func (*lookTool) InvokableRun(context.Context, string, ...tool.Option) (string, error) {
	return "seen", nil
}

func TestExtract(t *testing.T) {
	f := &fake{answers: []string{`{"save":[
		{"key":"tone","name":"Tone","description":"How replies sound","body":"Keep replies very short."},
		{"key":"Bad Key","name":"x","description":"y","body":"z"},
		{"key":"deadline","name":"Deadline","description":"Launch date","body":""},
		{"key":"deadline","name":"Deadline","description":"Launch date","body":"Launch on 2026-11-01."}],
		"delete":["stack","tone","unknown","stack"]}`}}
	changes, usage, err := memory.Extract(context.Background(), f.factory, memory.ExtractRequest{
		Entries: entries, Earlier: []memory.Message{{Role: einorun.RoleUser, Content: "hello"}},
		Recent: []memory.Message{{Role: einorun.RoleUser, Content: "shorter please, and we launch Nov 1"}}, Language: llm.Chinese,
	})
	if err != nil {
		t.Fatal(err)
	}
	if usage.Total != 6 {
		t.Fatalf("usage %+v", usage)
	}
	if len(changes.Saved) != 2 || changes.Saved[0].Body != "Keep replies very short." || changes.Saved[1].Key != "deadline" {
		t.Fatalf("saved %+v", changes.Saved)
	}
	if fmt.Sprint(changes.Deleted) != "[stack]" {
		t.Fatalf("deleted %v", changes.Deleted)
	}
	if len(changes.Skipped) != 2 || !strings.HasPrefix(changes.Skipped[0].Reason, "key") || !strings.HasPrefix(changes.Skipped[1].Reason, "body") {
		t.Fatalf("skipped %+v", changes.Skipped)
	}
	input := f.structured[0]
	if system := messageText(input[0]); !strings.Contains(system, "## 应该记住") || !strings.Contains(system, "不超过 2000 个字符") {
		t.Fatalf("instruction %q", system)
	}
	if user := messageText(input[1]); !strings.Contains(user, `"key":"tone"`) || !strings.Contains(user, "## 新消息\n[{\"role\":\"user\",\"content\":\"shorter please") {
		t.Fatalf("input %q", user)
	}
}

func TestExtractCustomCriteriaAndNoInput(t *testing.T) {
	f := &fake{answers: []string{`{"save":[],"delete":[]}`}}
	changes, _, err := memory.Extract(context.Background(), f.factory, memory.ExtractRequest{})
	if err != nil || len(f.structured) != 0 || len(changes.Saved) != 0 {
		t.Fatalf("extract without messages: %v %d", err, len(f.structured))
	}
	_, _, err = memory.Extract(context.Background(), f.factory, memory.ExtractRequest{
		Instruction: "Remember only food preferences.", Recent: []memory.Message{{Role: einorun.RoleUser, Content: "I like tea"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	system := messageText(f.structured[0][0])
	if !strings.HasPrefix(system, "Remember only food preferences.\n\n## Rules") || strings.Contains(system, "What to remember") {
		t.Fatalf("instruction %q", system)
	}
}

// awaitTool hands its call to an external executor.
type awaitTool struct{}

func (*awaitTool) Info(context.Context) (*schema.ToolInfo, error) {
	return &schema.ToolInfo{Name: "dispatch", Desc: "dispatch"}, nil
}

func (*awaitTool) InvokableRun(context.Context, string, ...tool.Option) (string, error) {
	return "", einorun.Await(json.RawMessage(`{}`))
}

func TestRecallAcrossResume(t *testing.T) {
	f := &fake{answers: []string{`{"keys":["tone"]}`}, steps: []func() *schema.AgenticMessage{callTool("dispatch"), ok()}}
	feed := inmem.NewFeed()
	feed.Append(einorun.Message{ID: "m1", Revision: "1", Role: einorun.RoleUser, Content: "write me a reply"})
	journal := inmem.NewJournal()
	request := einorun.Request{RunID: "r1", Model: einorun.Model{New: f.factory}, Feed: feed, Journal: journal,
		Extensions: []einorun.Extension{memory.Recall(source(entries), memory.RecallOptions{})},
		Tools:      []einorun.ToolSpec{{Tool: &awaitTool{}}}}
	runtime := einorun.New(einorun.Config{Language: llm.English})
	ctx := context.Background()
	result, err := runtime.Run(ctx, request)
	if err != nil || !result.Suspended {
		t.Fatalf("first start %+v %v", result, err)
	}
	resume := journal.Resume()
	if strings.Contains(string(resume.State), "Keep replies short.") {
		t.Fatal("the reminder is in the checkpoint")
	}
	output := "done"
	if err := journal.External(ctx, einorun.ToolCall{ID: result.Blocks[0].Call.ID, Status: einorun.StatusSucceeded, Result: &output}); err != nil {
		t.Fatal(err)
	}
	resume = journal.Resume()
	request.Resume = &resume
	result, err = runtime.Run(ctx, request)
	if err != nil {
		t.Fatal(err)
	}
	if len(f.structured) != 1 {
		t.Fatalf("selections %d", len(f.structured))
	}
	if !strings.Contains(messageText(f.agent[1][len(f.agent[1])-4]), "Keep replies short.") {
		for _, m := range f.agent[1] {
			t.Logf("%s %q", m.Role, messageText(m))
		}
		t.Fatal("no reminder after resume")
	}
	if result.Usage.Total != 12+6+12 {
		t.Fatalf("usage %+v", result.Usage)
	}
}

func TestRecallNegativeLimits(t *testing.T) {
	f := &fake{answers: []string{`{"keys":["tone"]}`}, steps: []func() *schema.AgenticMessage{ok()}}
	limits := memory.Limits{IndexEntries: -1, IndexBytes: -1, Candidates: -1, Selected: -1, EntryBytes: -1, SelectedBytes: -1, ConversationRunes: -1}
	runWith(t, f, memory.Recall(source(entries), memory.RecallOptions{Limits: limits}))
	if !strings.Contains(messageText(f.agent[0][len(f.agent[0])-2]), "Keep replies short.") {
		t.Fatal("no reminder with default limits")
	}
}

func TestExtractKeepsExistingKeys(t *testing.T) {
	spaced := []memory.Entry{{Key: " tone ", Name: "Tone", Description: "d", Body: "b"}, {Key: "old", Name: "Old", Description: "d", Body: "b"}}
	f := &fake{answers: []string{`{"save":[{"key":" tone ","name":"Tone","description":"d","body":"new"}],"delete":[" old "]}`}}
	now := time.Date(2026, 10, 9, 8, 0, 0, 0, time.UTC)
	changes, _, err := memory.Extract(context.Background(), f.factory, memory.ExtractRequest{
		Entries: spaced, Now: now, Recent: []memory.Message{{Role: einorun.RoleUser, Content: "x", At: now}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(changes.Saved) != 1 || changes.Saved[0].Key != " tone " || fmt.Sprint(changes.Deleted) != "[old]" {
		t.Fatalf("changes %+v", changes)
	}
	if input := messageText(f.structured[0][1]); !strings.HasPrefix(input, "Current time: 2026-10-09T08:00:00Z\n\n") || !strings.Contains(input, `"at":"2026-10-09T08:00:00Z"`) {
		t.Fatalf("input %q", input)
	}
}
