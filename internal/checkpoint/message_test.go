package checkpoint

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/cloudwego/eino/schema"
	openaischema "github.com/cloudwego/eino/schema/openai"
)

func TestRoundTrip(t *testing.T) {
	assistant := &schema.AgenticMessage{Role: schema.AgenticRoleTypeAssistant, ContentBlocks: []*schema.ContentBlock{
		schema.NewContentBlock(&schema.Reasoning{Text: "think", Signature: "sig", OpenAIExtension: &openaischema.ReasoningExtension{
			Content: []*openaischema.ReasoningContent{{Text: "part"}}}}),
		schema.NewContentBlock(&schema.AssistantGenText{Text: "calling"}),
		schema.NewContentBlock(&schema.FunctionToolCall{CallID: "c1", Name: "t", Arguments: `{"a":1}`}),
	}, ResponseMeta: &schema.AgenticResponseMeta{TokenUsage: &schema.TokenUsage{PromptTokens: 3, CompletionTokens: 2, TotalTokens: 5}},
		Extra: map[string]any{"id": "m1", "n": 2, "f": 1.5, "b": true, "raw": []byte("x"), "obj": map[string]any{"k": "v"}, "skip": struct{}{}}}
	user := &schema.AgenticMessage{Role: schema.AgenticRoleTypeUser, ContentBlocks: []*schema.ContentBlock{
		schema.NewContentBlock(&schema.UserInputText{Text: "look"}),
		schema.NewContentBlock(&schema.UserInputImage{Base64Data: "AAA", MIMEType: "image/png"}),
		schema.NewContentBlock(&schema.FunctionToolResult{CallID: "c1", Name: "t", Content: []*schema.FunctionToolResultContentBlock{
			{Type: schema.FunctionToolResultContentBlockTypeText, Text: &schema.UserInputText{Text: "ok"}},
		}}),
	}}
	encoded, err := EncodeMessages([]*schema.AgenticMessage{schema.SystemAgenticMessage("sys"), assistant, user})
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(encoded)
	if err != nil {
		t.Fatal(err)
	}
	var back []Message
	if err := json.Unmarshal(data, &back); err != nil {
		t.Fatal(err)
	}
	decoded, err := DecodeMessages(back)
	if err != nil {
		t.Fatal(err)
	}
	if len(decoded) != 3 {
		t.Fatalf("decoded %d messages", len(decoded))
	}
	got := decoded[1]
	if got.ContentBlocks[0].Reasoning.Signature != "sig" || got.ContentBlocks[0].Reasoning.OpenAIExtension.Content[0].Text != "part" ||
		got.ContentBlocks[2].FunctionToolCall.Arguments != `{"a":1}` || got.ResponseMeta.TokenUsage.TotalTokens != 5 {
		t.Fatalf("assistant %+v", got)
	}
	wantExtra := map[string]any{"id": "m1", "n": 2, "f": 1.5, "b": true, "raw": []byte("x"), "obj": map[string]any{"k": "v"}}
	if !reflect.DeepEqual(got.Extra, wantExtra) {
		t.Fatalf("extra %#v", got.Extra)
	}
	u := decoded[2]
	if u.ContentBlocks[1].UserInputImage.Base64Data != "AAA" || u.ContentBlocks[2].FunctionToolResult.Content[0].Text.Text != "ok" {
		t.Fatalf("user %+v", u)
	}
}
