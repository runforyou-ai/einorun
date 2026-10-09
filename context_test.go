package einorun_test

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"strings"
	"sync"
	"testing"

	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"

	"github.com/runforyou-ai/einorun"
	"github.com/runforyou-ai/einorun/inmem"
	"github.com/runforyou-ai/einorun/llm"
)

// texts returns every text the model saw, including tool results.
func texts(input []*schema.AgenticMessage) string {
	var parts []string
	for _, m := range input {
		parts = append(parts, llmText(m))
		for _, b := range m.ContentBlocks {
			if b.FunctionToolResult != nil {
				for _, c := range b.FunctionToolResult.Content {
					if c.Text != nil {
						parts = append(parts, c.Text.Text)
					}
				}
			}
		}
	}
	return strings.Join(parts, "\n")
}

func TestLargeResultsAreOffloaded(t *testing.T) {
	big := strings.Repeat("0123456789", 2000)
	var seen string
	m := &scripted{steps: []step{
		call(invocation{"dump", "{}"}),
		func(input []*schema.AgenticMessage) *schema.AgenticMessage {
			seen = texts(input)
			return say("ok")(input)
		},
	}}
	feed := inmem.NewFeed()
	user(feed, "m1", "dump")
	dump := &fn{name: "dump", run: func(context.Context, string) (string, error) { return big, nil }}
	result, err := run(t, einorun.Request{Model: einorun.Model{New: m.factory, ContextWindow: 4000}, Feed: feed,
		Tools: []einorun.ToolSpec{{Tool: dump}}})
	if err != nil || result.Text != "ok" {
		t.Fatalf("%+v %v", result, err)
	}
	if strings.Contains(seen, big) || !strings.Contains(seen, einorun.OffloadedPath("")) {
		t.Fatalf("the model saw the whole result or no offload note: %d bytes", len(seen))
	}
}

// summaryAware answers summary calls (with an output limit) separately.
type summaryAware struct {
	main      *scripted
	summaries int
}

func (s *summaryAware) factory(ctx context.Context, options llm.ModelOptions) (model.AgenticModel, error) {
	if options.MaxOutputTokens > 0 {
		return &summaryModel{parent: s}, nil
	}
	return s.main, nil
}

type summaryModel struct{ parent *summaryAware }

func (m *summaryModel) Generate(_ context.Context, _ []*schema.AgenticMessage, _ ...model.Option) (*schema.AgenticMessage, error) {
	m.parent.summaries++
	return say("<analysis>scratch</analysis>the user asked many things")(nil), nil
}

func (m *summaryModel) Stream(ctx context.Context, input []*schema.AgenticMessage, opts ...model.Option) (*schema.StreamReader[*schema.AgenticMessage], error) {
	out, err := m.Generate(ctx, input, opts...)
	if err != nil {
		return nil, err
	}
	return schema.StreamReaderFromArray([]*schema.AgenticMessage{out}), nil
}

func TestLongContextIsSummarized(t *testing.T) {
	var seen string
	main := &scripted{steps: []step{
		call(invocation{"lookup", `{"x":"1"}`}),
		call(invocation{"lookup", `{"x":"2"}`}),
		call(invocation{"lookup", `{"x":"3"}`}),
		call(invocation{"lookup", `{"x":"4"}`}),
		func(input []*schema.AgenticMessage) *schema.AgenticMessage {
			seen = texts(input)
			return say("done")(input)
		},
	}}
	s := &summaryAware{main: main}
	feed := inmem.NewFeed()
	for i := range 30 {
		user(feed, "old"+string(rune('a'+i)), strings.Repeat("earlier conversation ", 40))
	}
	user(feed, "new", "latest question")
	padded := &fn{name: "lookup", run: func(context.Context, string) (string, error) { return strings.Repeat("r", 2300), nil }}
	result, err := run(t, einorun.Request{Model: einorun.Model{New: s.factory, ContextWindow: 8000}, Feed: feed,
		Tools: []einorun.ToolSpec{{Tool: padded}}})
	if err != nil || result.Text != "done" {
		t.Fatalf("%+v %v", result, err)
	}
	if s.summaries == 0 || !strings.Contains(seen, "the user asked many things") || strings.Contains(seen, "scratch") || !strings.Contains(seen, "latest question") {
		t.Fatalf("summaries %d, model saw %d bytes", s.summaries, len(seen))
	}
	if result.Usage.Total <= 12*3 {
		t.Fatalf("summary usage missing: %+v", result.Usage)
	}
}

func digest(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func TestToolResultMedia(t *testing.T) {
	image := []byte("png-bytes")
	refs := []einorun.MediaRef{
		{Key: "tool:1", MIME: "image/png", SHA256: digest(image), Size: int64(len(image))},
		{Key: "tool:2", MIME: "image/png", SHA256: "deadbeef", Size: 4},
		{Key: "tool:3", MIME: "application/pdf", Size: 4},
	}
	read := func(_ context.Context, ref einorun.MediaRef) ([]byte, error) {
		if ref.Key == "tool:1" {
			return image, nil
		}
		return []byte("xxxx"), nil
	}
	for _, inResult := range []bool{false, true} {
		var images, notes int
		m := &scripted{steps: []step{
			call(invocation{"screenshot", "{}"}),
			func(input []*schema.AgenticMessage) *schema.AgenticMessage {
				for _, msg := range input {
					for _, b := range msg.ContentBlocks {
						if b.UserInputImage != nil {
							images++
						}
						if b.FunctionToolResult != nil {
							for _, c := range b.FunctionToolResult.Content {
								if c.Image != nil {
									images++
								}
							}
						}
					}
				}
				notes = strings.Count(texts(input), "cannot view") + strings.Count(texts(input), "content changed")
				return say("seen")(input)
			},
		}}
		feed := inmem.NewFeed()
		user(feed, "m1", "look")
		shot := &fn{name: "screenshot", run: func(ctx context.Context, _ string) (string, error) {
			return "captured", einorun.AttachMedia(ctx, refs...)
		}}
		result, err := run(t, einorun.Request{Model: einorun.Model{New: m.factory, Inputs: []llm.Modality{llm.Image}, ToolResultMedia: inResult},
			Feed: feed, ReadMedia: read, Tools: []einorun.ToolSpec{{Tool: shot}}})
		if err != nil {
			t.Fatal(err)
		}
		if images != 1 || notes != 2 {
			t.Fatalf("inResult %v: images %d notes %d", inResult, images, notes)
		}
		call := result.Blocks[0].Call
		if *call.Result != "captured" || len(call.Media) != 3 {
			t.Fatalf("call %+v", call)
		}
	}
}

func TestNewestMediaWinsTheBudget(t *testing.T) {
	data := map[string][]byte{"old": []byte("old-image"), "new": []byte("new-image")}
	read := func(_ context.Context, ref einorun.MediaRef) ([]byte, error) { return data[ref.Key], nil }
	var seen []string
	m := &scripted{steps: []step{
		call(invocation{"shot", `{"x":"old"}`}, invocation{"shot", `{"x":"new"}`}),
		func(input []*schema.AgenticMessage) *schema.AgenticMessage {
			for _, msg := range input {
				for _, b := range msg.ContentBlocks {
					if b.UserInputImage != nil {
						seen = append(seen, b.UserInputImage.Base64Data)
					}
				}
			}
			return say("ok")(input)
		},
	}}
	feed := inmem.NewFeed()
	user(feed, "m1", "look")
	shot := &fn{name: "shot", run: func(ctx context.Context, args string) (string, error) {
		key := "old"
		if strings.Contains(args, "new") {
			key = "new"
		}
		return key, einorun.AttachMedia(ctx, einorun.MediaRef{Key: key, MIME: "image/png"})
	}}
	// A window this small allows one image per run.
	if _, err := run(t, einorun.Request{Model: einorun.Model{New: m.factory, ContextWindow: 6000, Inputs: []llm.Modality{llm.Image}},
		Feed: feed, ReadMedia: read, Tools: []einorun.ToolSpec{{Tool: shot}}}); err != nil {
		t.Fatal(err)
	}
	if len(seen) != 1 || seen[0] != base64.StdEncoding.EncodeToString(data["new"]) {
		t.Fatalf("images %v", seen)
	}
}

// stepLog keeps every saved step.
type stepLog struct {
	*inmem.Journal
	mu     sync.Mutex
	states [][]byte
}

func (j *stepLog) SaveStep(ctx context.Context, step einorun.Step) error {
	j.mu.Lock()
	j.states = append(j.states, step.State)
	j.mu.Unlock()
	return j.Journal.SaveStep(ctx, step)
}

func TestMediaSurvivesACrashBeforeTheStep(t *testing.T) {
	image := []byte("img")
	read := func(context.Context, einorun.MediaRef) ([]byte, error) { return image, nil }
	m := &scripted{steps: []step{call(invocation{"shot", "{}"}, invocation{"dispatch", "{}"})}}
	feed := inmem.NewFeed()
	user(feed, "m1", "look")
	journal := &stepLog{Journal: inmem.NewJournal()}
	shot := &fn{name: "shot", run: func(ctx context.Context, _ string) (string, error) {
		return "captured", einorun.AttachMedia(ctx, einorun.MediaRef{Key: "k", MIME: "image/png", SHA256: digest(image)})
	}}
	dispatch := &fn{name: "dispatch", run: func(context.Context, string) (string, error) { return "", einorun.Await(nil) }}
	request := einorun.Request{Model: einorun.Model{New: m.factory, Inputs: []llm.Modality{llm.Image}}, Feed: feed, Journal: journal,
		ReadMedia: read, Tools: []einorun.ToolSpec{{Tool: shot}, {Tool: dispatch}}}
	if result, err := run(t, request); err != nil || !result.Suspended {
		t.Fatalf("%+v %v", result, err)
	}
	resume := journal.Resume()
	// Crash window: the step saved after the model output, before the tools ran.
	resume.State = journal.states[0]
	for i := range resume.Blocks {
		if c := resume.Blocks[i].Call; c != nil && c.Name == "dispatch" {
			output := "done"
			c.Status, c.Result = einorun.StatusSucceeded, &output
		}
	}
	images := 0
	m.steps = []step{func(input []*schema.AgenticMessage) *schema.AgenticMessage {
		for _, msg := range input {
			for _, b := range msg.ContentBlocks {
				if b.UserInputImage != nil {
					images++
				}
			}
		}
		return say("ok")(input)
	}}
	request.Resume = &resume
	if _, err := run(t, request); err != nil {
		t.Fatal(err)
	}
	if images != 1 {
		t.Fatalf("images after recovery %d", images)
	}
}

func TestMediaFollowsTheWholeBatch(t *testing.T) {
	read := func(context.Context, einorun.MediaRef) ([]byte, error) { return []byte("img"), nil }
	var order []string
	m := &scripted{steps: []step{
		call(invocation{"shot", "{}"}, invocation{"lookup", "{}"}),
		func(input []*schema.AgenticMessage) *schema.AgenticMessage {
			for _, msg := range input {
				switch {
				case len(msg.ContentBlocks) > 0 && msg.ContentBlocks[0].FunctionToolResult != nil:
					order = append(order, "result")
				case len(msg.ContentBlocks) > 0 && msg.ContentBlocks[0].UserInputImage != nil:
					order = append(order, "media")
				}
			}
			return say("ok")(input)
		},
	}}
	feed := inmem.NewFeed()
	user(feed, "m1", "look")
	shot := &fn{name: "shot", run: func(ctx context.Context, _ string) (string, error) {
		return "captured", einorun.AttachMedia(ctx, einorun.MediaRef{Key: "k", MIME: "image/png"})
	}}
	if _, err := run(t, einorun.Request{Model: einorun.Model{New: m.factory, Inputs: []llm.Modality{llm.Image}}, Feed: feed, ReadMedia: read,
		Tools: []einorun.ToolSpec{{Tool: shot}, {Tool: echo("lookup")}}}); err != nil {
		t.Fatal(err)
	}
	if strings.Join(order, ",") != "result,result,media" {
		t.Fatalf("order %v", order)
	}
}

// errDiskUnavailable is the save error of failingSteps.
var errDiskUnavailable = errors.New("disk unavailable")

// failingSteps fails the first step saved after an offload, once.
type failingSteps struct {
	*inmem.Journal
	fail bool
}

func (j *failingSteps) SaveStep(ctx context.Context, step einorun.Step) error {
	if j.fail {
		j.fail = false
		return errDiskUnavailable
	}
	return j.Journal.SaveStep(ctx, step)
}

func TestOffloadSaveFailureEndsTheRun(t *testing.T) {
	m := &scripted{steps: []step{call(invocation{"dump", "{}"}), say("ok")}}
	feed := inmem.NewFeed()
	user(feed, "m1", "dump")
	journal := &failingSteps{Journal: inmem.NewJournal()}
	dump := &fn{name: "dump", run: func(context.Context, string) (string, error) {
		journal.fail = true
		return strings.Repeat("x", 50000), nil
	}}
	_, err := run(t, einorun.Request{Model: einorun.Model{New: m.factory, ContextWindow: 4000}, Feed: feed, Journal: journal,
		Tools: []einorun.ToolSpec{{Tool: dump}}})
	if !errors.Is(err, errDiskUnavailable) {
		t.Fatalf("err %v", err)
	}
	if len(m.inputs) != 1 {
		t.Fatalf("model calls %d, want 1", len(m.inputs))
	}
}
