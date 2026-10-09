package provider

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/cloudwego/eino/schema"

	"github.com/runforyou-ai/einorun/llm"
	"github.com/runforyou-ai/einorun/provider/vendor"
)

// completions is a fake Chat Completions endpoint. reply decides each
// response from the decoded request body.
type completions struct {
	mu       sync.Mutex
	requests []map[string]any
	reply    func(body map[string]any) (int, string)
}

func (c *completions) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	raw, _ := io.ReadAll(r.Body)
	var body map[string]any
	_ = json.Unmarshal(raw, &body)
	c.mu.Lock()
	c.requests = append(c.requests, body)
	c.mu.Unlock()
	if body["stream"] == true {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, chunk := range []string{
			`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"part"}}]}`,
			`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"length"}]}`,
		} {
			_, _ = io.WriteString(w, "data: "+chunk+"\n\n")
		}
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
		return
	}
	status, response := c.reply(body)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = io.WriteString(w, response)
}

func completion(content, finish, refusal string) string {
	message := map[string]any{"role": "assistant", "content": content}
	if refusal != "" {
		message["refusal"] = refusal
	}
	encoded, _ := json.Marshal(map[string]any{
		"id": "c1", "object": "chat.completion", "created": 1, "model": "m",
		"choices": []any{map[string]any{"index": 0, "message": message, "finish_reason": finish}},
		"usage":   map[string]any{"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
	})
	return string(encoded)
}

func ask(t *testing.T, factory llm.ModelFactory, options llm.ModelOptions) (*schema.AgenticMessage, error) {
	t.Helper()
	m, err := factory(context.Background(), options)
	if err != nil {
		t.Fatal(err)
	}
	return m.Generate(context.Background(), []*schema.AgenticMessage{schema.UserAgenticMessage("hi")})
}

func TestStructuredOutputFallsBackWhenRejected(t *testing.T) {
	server := &completions{reply: func(body map[string]any) (int, string) {
		if _, constrained := body["response_format"]; constrained {
			return http.StatusBadRequest, `{"error":{"message":"response_format json_schema is not supported"}}`
		}
		return http.StatusOK, completion(`{"a":1}`, "stop", "")
	}}
	ts := httptest.NewServer(server)
	defer ts.Close()
	factory := Factory(ChatConfig{Brand: vendor.Mistral, BaseURL: ts.URL, APIKey: "k", Model: "m"})
	message, err := ask(t, factory, llm.ModelOptions{Output: llm.SchemaFor[struct {
		A int `json:"a"`
	}](), DisableThinking: true})
	if err != nil {
		t.Fatal(err)
	}
	if llm.Text(message) != `{"a":1}` {
		t.Fatalf("text %q", llm.Text(message))
	}
	if len(server.requests) != 2 {
		t.Fatalf("%d requests", len(server.requests))
	}
	format := server.requests[0]["response_format"].(map[string]any)
	if format["type"] != "json_schema" {
		t.Fatalf("format %v", format)
	}
	if _, ok := server.requests[0]["reasoning"]; ok {
		t.Fatal("thinking fields sent for a brand without them")
	}
}

func TestStopReasonsAndRefusals(t *testing.T) {
	var next string
	server := &completions{reply: func(map[string]any) (int, string) { return http.StatusOK, next }}
	ts := httptest.NewServer(server)
	defer ts.Close()
	factory := Factory(ChatConfig{Brand: vendor.OpenAICompatible, BaseURL: ts.URL, Model: "m"})

	next = completion("partial", "length", "")
	message, err := ask(t, factory, llm.ModelOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if llm.StopReasonOf(message) != llm.StopLength {
		t.Fatalf("stop %q", llm.StopReasonOf(message))
	}
	if llm.UsageOf(message.ResponseMeta).Total != 5 {
		t.Fatalf("usage %+v", llm.UsageOf(message.ResponseMeta))
	}

	next = completion("", "stop", "I can't help with that")
	message, err = ask(t, factory, llm.ModelOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if llm.StopReasonOf(message) != llm.StopRefusal {
		t.Fatalf("refusal stop %q", llm.StopReasonOf(message))
	}
}

func TestRequestFields(t *testing.T) {
	server := &completions{reply: func(map[string]any) (int, string) { return http.StatusOK, completion("ok", "stop", "") }}
	ts := httptest.NewServer(server)
	defer ts.Close()

	factory := Factory(ChatConfig{Brand: vendor.Zhipu, BaseURL: ts.URL, Model: "m", MaxOutputTokens: 100})
	if _, err := ask(t, factory, llm.ModelOptions{DisableThinking: true, MaxOutputTokens: 50}); err != nil {
		t.Fatal(err)
	}
	body := server.requests[0]
	if body["max_tokens"] != float64(50) {
		t.Fatalf("max_tokens %v", body["max_tokens"])
	}
	if thinking, _ := body["thinking"].(map[string]any); thinking["type"] != "disabled" {
		t.Fatalf("thinking %v", body["thinking"])
	}

	factory = Factory(ChatConfig{Brand: vendor.DeepSeek, BaseURL: ts.URL, Model: "m"})
	if _, err := ask(t, factory, llm.ModelOptions{DisableThinking: true, Output: llm.SchemaFor[struct {
		A int `json:"a"`
	}]()}); err != nil {
		t.Fatal(err)
	}
	body = server.requests[1]
	if format, _ := body["response_format"].(map[string]any); format["type"] != "json_object" {
		t.Fatalf("deepseek format %v", body["response_format"])
	}
}

func TestStructuredOverride(t *testing.T) {
	server := &completions{reply: func(map[string]any) (int, string) { return http.StatusOK, completion("{}", "stop", "") }}
	ts := httptest.NewServer(server)
	defer ts.Close()
	factory := Factory(ChatConfig{Brand: vendor.OpenAICompatible, BaseURL: ts.URL, Model: "m", Structured: vendor.StructuredJSONObject})
	if _, err := ask(t, factory, llm.ModelOptions{Output: llm.SchemaFor[struct{}]()}); err != nil {
		t.Fatal(err)
	}
	if format, _ := server.requests[0]["response_format"].(map[string]any); format["type"] != "json_object" {
		t.Fatalf("format %v", server.requests[0]["response_format"])
	}
}

func TestConfigErrors(t *testing.T) {
	if _, err := NewChatModel(context.Background(), ChatConfig{Brand: vendor.Anthropic, BaseURL: "https://example.com", Model: "m"}); err == nil ||
		!strings.Contains(err.Error(), "max output tokens") {
		t.Fatalf("anthropic without max tokens: %v", err)
	}
	for _, c := range []ChatConfig{
		{Brand: vendor.Volcengine, Structured: vendor.StructuredJSONSchema},
		{Brand: vendor.Google, Structured: vendor.StructuredJSONObject},
		{Brand: vendor.DeepSeek, Structured: vendor.StructuredJSONSchema},
		{Brand: vendor.OpenAI, Protocol: "nope"},
	} {
		c.BaseURL, c.Model, c.MaxOutputTokens, c.Output = "https://example.com", "m", 10, llm.SchemaFor[struct{}]()
		if _, err := NewChatModel(context.Background(), c); err == nil {
			t.Fatalf("accepted %s %s %s", c.Brand, c.Protocol, c.Structured)
		}
	}
	if _, err := NewChatModel(context.Background(), ChatConfig{Brand: "nope", BaseURL: "https://example.com"}); err == nil {
		t.Fatal("unknown brand accepted")
	}
	if _, err := NewChatModel(context.Background(), ChatConfig{Brand: vendor.OpenAI, BaseURL: "not a url"}); err == nil {
		t.Fatal("invalid URL accepted")
	}
}

func TestStructuredModelsDoNotStream(t *testing.T) {
	m, err := NewChatModel(context.Background(), ChatConfig{Brand: vendor.OpenAI, BaseURL: "https://example.com", Model: "m", Output: llm.SchemaFor[struct{}]()})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := m.Stream(context.Background(), nil); !errors.Is(err, ErrStructuredStream) {
		t.Fatalf("stream: %v", err)
	}
}

func TestStreamStopReason(t *testing.T) {
	ts := httptest.NewServer(&completions{})
	defer ts.Close()
	m, err := NewChatModel(context.Background(), ChatConfig{Brand: vendor.OpenAICompatible, BaseURL: ts.URL, Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	stream, err := m.Stream(context.Background(), []*schema.AgenticMessage{schema.UserAgenticMessage("hi")})
	if err != nil {
		t.Fatal(err)
	}
	var chunks []*schema.AgenticMessage
	for {
		chunk, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		chunks = append(chunks, chunk)
	}
	message, err := schema.ConcatAgenticMessages(chunks)
	if err != nil {
		t.Fatal(err)
	}
	if llm.StopReasonOf(message) != llm.StopLength || llm.Text(message) != "part" {
		t.Fatalf("stop %q text %q", llm.StopReasonOf(message), llm.Text(message))
	}
}

func TestForcedToolOverCompatibleProtocol(t *testing.T) {
	server := &completions{reply: func(map[string]any) (int, string) {
		encoded, _ := json.Marshal(map[string]any{
			"id": "c1", "object": "chat.completion", "created": 1, "model": "m",
			"choices": []any{map[string]any{"index": 0, "finish_reason": "tool_calls", "message": map[string]any{
				"role": "assistant", "content": "",
				"tool_calls": []any{map[string]any{"id": "t1", "type": "function", "function": map[string]any{"name": "output", "arguments": `{"a":1}`}}},
			}}},
		})
		return http.StatusOK, string(encoded)
	}}
	ts := httptest.NewServer(server)
	defer ts.Close()
	// An Anthropic model reached through an OpenAI-compatible gateway.
	factory := Factory(ChatConfig{Brand: vendor.Anthropic, Protocol: vendor.ProtocolCompatible, BaseURL: ts.URL, Model: "m", Language: llm.Chinese})
	message, err := ask(t, factory, llm.ModelOptions{Output: llm.SchemaFor[struct {
		A int `json:"a"`
	}]()})
	if err != nil {
		t.Fatal(err)
	}
	if llm.Text(message) != `{"a":1}` {
		t.Fatalf("text %q", llm.Text(message))
	}
	body := server.requests[0]
	tools, _ := body["tools"].([]any)
	if len(tools) != 1 || body["tool_choice"] == nil {
		t.Fatalf("tools %v choice %v", body["tools"], body["tool_choice"])
	}
	function := tools[0].(map[string]any)["function"].(map[string]any)
	if function["description"] != "按要求的结构提交输出" {
		t.Fatalf("description %v", function["description"])
	}
}

func TestQwenStructuredWithoutThinking(t *testing.T) {
	server := &completions{reply: func(map[string]any) (int, string) { return http.StatusOK, completion("{}", "stop", "") }}
	ts := httptest.NewServer(server)
	defer ts.Close()
	factory := Factory(ChatConfig{Brand: vendor.Alibaba, BaseURL: ts.URL, Model: "m"})
	if _, err := ask(t, factory, llm.ModelOptions{DisableThinking: true, Output: llm.SchemaFor[struct{}]()}); err != nil {
		t.Fatal(err)
	}
	body := server.requests[0]
	kwargs, _ := body["chat_template_kwargs"].(map[string]any)
	if body["enable_thinking"] != false || kwargs["enable_thinking"] != false || body["response_format"] == nil {
		t.Fatalf("body %v", body)
	}
}
