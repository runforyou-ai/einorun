package llm

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"
	"github.com/cloudwego/eino/schema/claude"
)

// scripted answers each Generate call with the next reply and records inputs.
type scripted struct {
	replies []*schema.AgenticMessage
	inputs  [][]*schema.AgenticMessage
	options []ModelOptions
}

func (s *scripted) factory(_ context.Context, options ModelOptions) (model.AgenticModel, error) {
	s.options = append(s.options, options)
	return s, nil
}

func (s *scripted) Generate(_ context.Context, input []*schema.AgenticMessage, _ ...model.Option) (*schema.AgenticMessage, error) {
	s.inputs = append(s.inputs, input)
	if len(s.replies) == 0 {
		return nil, errors.New("no reply left")
	}
	reply := s.replies[0]
	s.replies = s.replies[1:]
	return reply, nil
}

func (s *scripted) Stream(context.Context, []*schema.AgenticMessage, ...model.Option) (*schema.StreamReader[*schema.AgenticMessage], error) {
	return nil, errors.New("not supported")
}

func reply(text string, in, out int) *schema.AgenticMessage {
	return &schema.AgenticMessage{
		Role:          schema.AgenticRoleTypeAssistant,
		ContentBlocks: []*schema.ContentBlock{schema.NewContentBlock(&schema.AssistantGenText{Text: text})},
		ResponseMeta:  &schema.AgenticResponseMeta{TokenUsage: &schema.TokenUsage{PromptTokens: in, CompletionTokens: out}},
	}
}

type answer struct {
	Title string  `json:"title"`
	Note  *string `json:"note" jsonschema:"nullable"`
}

func TestGenerateObjectRetriesOnce(t *testing.T) {
	model := &scripted{replies: []*schema.AgenticMessage{
		reply("not json", 10, 2),
		reply("```json\n{\"title\":\"ok\",\"note\":null}\n```", 12, 5),
	}}
	got, usage, err := GenerateObject[answer](context.Background(), model.factory, GenerateRequest{Instruction: "i", Input: "q", Language: English})
	if err != nil {
		t.Fatal(err)
	}
	if got.Title != "ok" || got.Note != nil {
		t.Fatalf("decoded %+v", got)
	}
	if usage != (Usage{Input: 22, Output: 7, Total: 29}) {
		t.Fatalf("usage %+v", usage)
	}
	if len(model.inputs[1]) != 4 || !strings.Contains(Text(model.inputs[1][2]), "not json") {
		t.Fatalf("retry input %+v", model.inputs[1])
	}
	for _, options := range model.options {
		if !options.DisableThinking || options.Output == nil || options.Output.Name != "answer" {
			t.Fatalf("options %+v", options)
		}
	}
}

func TestGenerateObjectGivesUpAfterRetry(t *testing.T) {
	model := &scripted{replies: []*schema.AgenticMessage{reply("x", 1, 1), reply("y", 1, 1)}}
	_, usage, err := GenerateObject[answer](context.Background(), model.factory, GenerateRequest{})
	if err == nil || usage.Total != 4 {
		t.Fatalf("err %v usage %+v", err, usage)
	}
}

func TestGenerateObjectStopReasons(t *testing.T) {
	truncated := reply("{", 1, 1)
	SetStopReason(truncated, StopLength)
	model := &scripted{replies: []*schema.AgenticMessage{truncated}}
	if _, _, err := GenerateObject[answer](context.Background(), model.factory, GenerateRequest{}); !errors.Is(err, ErrTruncated) {
		t.Fatalf("truncated: %v", err)
	}
	refused := reply("", 1, 1)
	refused.ResponseMeta.ClaudeExtension = &claude.ResponseMetaExtension{StopReason: "refusal"}
	model = &scripted{replies: []*schema.AgenticMessage{refused}}
	if _, _, err := GenerateObject[answer](context.Background(), model.factory, GenerateRequest{}); !errors.Is(err, ErrRefused) {
		t.Fatalf("refused: %v", err)
	}
}

func TestSchemaFor(t *testing.T) {
	s := SchemaFor[answer]()
	note, ok := s.Schema.Properties.Get("note")
	if !ok || len(note.AnyOf) != 2 || len(note.OneOf) != 0 {
		t.Fatalf("note schema %+v", note)
	}
	if len(s.Schema.Required) != 2 {
		t.Fatalf("required %v", s.Schema.Required)
	}
}

func TestEstimateTokens(t *testing.T) {
	if got := EstimateTokens("你好abcd"); got != 3 {
		t.Fatalf("estimate %d", got)
	}
	if ContextWindow(0) != DefaultContextWindow || ContextWindow(10) != 10 {
		t.Fatal("context window")
	}
}

func TestModelCallID(t *testing.T) {
	ctx := WithModelCallID(context.Background(), "m1")
	if ModelCallID(ctx) != "m1" || ModelCallID(context.Background()) != "" {
		t.Fatal("model call id")
	}
}
