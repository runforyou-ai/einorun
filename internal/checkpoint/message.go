// Package checkpoint encodes model context into a stable JSON format that does
// not depend on how Eino serializes its types.
package checkpoint

import (
	"encoding/json"
	"fmt"
	"reflect"

	"github.com/cloudwego/eino/schema"
	openaischema "github.com/cloudwego/eino/schema/openai"
)

// Message is one model context message.
type Message struct {
	Role   string           `json:"role"`
	Blocks []Block          `json:"blocks,omitempty"`
	Usage  *Usage           `json:"usage,omitempty"`
	Extra  map[string]Value `json:"extra,omitempty"`
}

// Block is one content block; Type decides the fields in use.
type Block struct {
	Type             string           `json:"type"`
	Text             string           `json:"text,omitempty"`
	Signature        string           `json:"signature,omitempty"`        // reasoning signature that must be sent back unchanged
	ReasoningContent []string         `json:"reasoningContent,omitempty"` // Responses API reasoning item parts
	Media            *Media           `json:"media,omitempty"`
	Call             *ToolCall        `json:"call,omitempty"`
	Result           *ToolResult      `json:"result,omitempty"`
	Extra            map[string]Value `json:"extra,omitempty"`
}

// Media is an image, audio, video or file in user input or a tool result.
type Media struct {
	URL      string `json:"url,omitempty"`
	Data     string `json:"data,omitempty"` // base64
	MIMEType string `json:"mimeType,omitempty"`
	Name     string `json:"name,omitempty"`
	Detail   string `json:"detail,omitempty"`
}

// ToolCall is a function call made by the model.
type ToolCall struct {
	CallID    string `json:"callId,omitempty"`
	Name      string `json:"name"`
	Arguments string `json:"arguments,omitempty"`
}

// ToolResult is the result of a function call.
type ToolResult struct {
	CallID  string       `json:"callId,omitempty"`
	Name    string       `json:"name"`
	Content []ResultPart `json:"content,omitempty"`
}

// ResultPart is text or media in a tool result.
type ResultPart struct {
	Type  string           `json:"type"`
	Text  string           `json:"text,omitempty"`
	Media *Media           `json:"media,omitempty"`
	Extra map[string]Value `json:"extra,omitempty"`
}

// Usage is the usage of a model output.
type Usage struct {
	PromptTokens     int `json:"promptTokens"`
	CachedTokens     int `json:"cachedTokens,omitempty"`
	CompletionTokens int `json:"completionTokens"`
	ReasoningTokens  int `json:"reasoningTokens,omitempty"`
	TotalTokens      int `json:"totalTokens"`
}

// Value is an Extra value with its kind: string, bool, int, float, bytes or
// json.
type Value struct {
	Kind  string          `json:"kind"`
	Value json.RawMessage `json:"value"`
}

const (
	roleSystem    = "system"
	roleUser      = "user"
	roleAssistant = "assistant"

	blockUserText      = "userText"
	blockUserImage     = "userImage"
	blockUserAudio     = "userAudio"
	blockUserVideo     = "userVideo"
	blockUserFile      = "userFile"
	blockAssistantText = "assistantText"
	blockReasoning     = "reasoning"
	blockToolCall      = "toolCall"
	blockToolResult    = "toolResult"

	partText  = "text"
	partImage = "image"
	partAudio = "audio"
	partVideo = "video"
	partFile  = "file"
)

// EncodeMessages encodes model context.
func EncodeMessages(messages []*schema.AgenticMessage) ([]Message, error) {
	encoded := make([]Message, 0, len(messages))
	for i, message := range messages {
		m, err := encodeMessage(message)
		if err != nil {
			return nil, fmt.Errorf("checkpoint: message %d: %w", i, err)
		}
		encoded = append(encoded, m)
	}
	return encoded, nil
}

// DecodeMessages decodes model context.
func DecodeMessages(messages []Message) ([]*schema.AgenticMessage, error) {
	decoded := make([]*schema.AgenticMessage, 0, len(messages))
	for i, record := range messages {
		m, err := decodeMessage(record)
		if err != nil {
			return nil, fmt.Errorf("checkpoint: message %d: %w", i, err)
		}
		decoded = append(decoded, m)
	}
	return decoded, nil
}

// encodeMessage encodes one message.
func encodeMessage(message *schema.AgenticMessage) (Message, error) {
	var record Message
	switch message.Role {
	case schema.AgenticRoleTypeSystem:
		record.Role = roleSystem
	case schema.AgenticRoleTypeUser:
		record.Role = roleUser
	case schema.AgenticRoleTypeAssistant:
		record.Role = roleAssistant
	default:
		return Message{}, fmt.Errorf("unsupported role %q", message.Role)
	}
	extra, err := encodeExtra(message.Extra)
	if err != nil {
		return Message{}, err
	}
	record.Extra = extra
	if meta := message.ResponseMeta; meta != nil && meta.TokenUsage != nil {
		u := meta.TokenUsage
		record.Usage = &Usage{
			PromptTokens: u.PromptTokens, CachedTokens: u.PromptTokenDetails.CachedTokens,
			CompletionTokens: u.CompletionTokens, ReasoningTokens: u.CompletionTokensDetails.ReasoningTokens, TotalTokens: u.TotalTokens,
		}
	}
	for _, block := range message.ContentBlocks {
		if block == nil {
			continue
		}
		b, err := encodeBlock(block)
		if err != nil {
			return Message{}, err
		}
		record.Blocks = append(record.Blocks, b)
	}
	return record, nil
}

// encodeBlock encodes one content block.
func encodeBlock(block *schema.ContentBlock) (Block, error) {
	extra, err := encodeExtra(block.Extra)
	if err != nil {
		return Block{}, err
	}
	record := Block{Extra: extra}
	switch block.Type {
	case schema.ContentBlockTypeUserInputText:
		record.Type, record.Text = blockUserText, block.UserInputText.Text
	case schema.ContentBlockTypeUserInputImage:
		m := block.UserInputImage
		record.Type, record.Media = blockUserImage, &Media{URL: m.URL, Data: m.Base64Data, MIMEType: m.MIMEType, Detail: string(m.Detail)}
	case schema.ContentBlockTypeUserInputAudio:
		m := block.UserInputAudio
		record.Type, record.Media = blockUserAudio, &Media{URL: m.URL, Data: m.Base64Data, MIMEType: m.MIMEType}
	case schema.ContentBlockTypeUserInputVideo:
		m := block.UserInputVideo
		record.Type, record.Media = blockUserVideo, &Media{URL: m.URL, Data: m.Base64Data, MIMEType: m.MIMEType}
	case schema.ContentBlockTypeUserInputFile:
		m := block.UserInputFile
		record.Type, record.Media = blockUserFile, &Media{URL: m.URL, Data: m.Base64Data, MIMEType: m.MIMEType, Name: m.Name}
	case schema.ContentBlockTypeAssistantGenText:
		record.Type, record.Text = blockAssistantText, block.AssistantGenText.Text
	case schema.ContentBlockTypeReasoning:
		r := block.Reasoning
		record.Type, record.Text, record.Signature = blockReasoning, r.Text, r.Signature
		if r.OpenAIExtension != nil {
			for _, content := range r.OpenAIExtension.Content {
				if content != nil {
					record.ReasoningContent = append(record.ReasoningContent, content.Text)
				}
			}
		}
	case schema.ContentBlockTypeFunctionToolCall:
		c := block.FunctionToolCall
		record.Type, record.Call = blockToolCall, &ToolCall{CallID: c.CallID, Name: c.Name, Arguments: c.Arguments}
	case schema.ContentBlockTypeFunctionToolResult:
		r := block.FunctionToolResult
		record.Type, record.Result = blockToolResult, &ToolResult{CallID: r.CallID, Name: r.Name}
		for _, part := range r.Content {
			if part == nil {
				continue
			}
			p, err := encodePart(part)
			if err != nil {
				return Block{}, err
			}
			record.Result.Content = append(record.Result.Content, p)
		}
	default:
		return Block{}, fmt.Errorf("unsupported content block type %q", block.Type)
	}
	return record, nil
}

// encodePart encodes one part of a tool result.
func encodePart(part *schema.FunctionToolResultContentBlock) (ResultPart, error) {
	extra, err := encodeExtra(part.Extra)
	if err != nil {
		return ResultPart{}, err
	}
	record := ResultPart{Extra: extra}
	switch part.Type {
	case schema.FunctionToolResultContentBlockTypeText:
		record.Type, record.Text = partText, part.Text.Text
	case schema.FunctionToolResultContentBlockTypeImage:
		record.Type, record.Media = partImage, &Media{URL: part.Image.URL, Data: part.Image.Base64Data, MIMEType: part.Image.MIMEType, Detail: string(part.Image.Detail)}
	case schema.FunctionToolResultContentBlockTypeAudio:
		record.Type, record.Media = partAudio, &Media{URL: part.Audio.URL, Data: part.Audio.Base64Data, MIMEType: part.Audio.MIMEType}
	case schema.FunctionToolResultContentBlockTypeVideo:
		record.Type, record.Media = partVideo, &Media{URL: part.Video.URL, Data: part.Video.Base64Data, MIMEType: part.Video.MIMEType}
	case schema.FunctionToolResultContentBlockTypeFile:
		record.Type, record.Media = partFile, &Media{URL: part.File.URL, Data: part.File.Base64Data, MIMEType: part.File.MIMEType, Name: part.File.Name}
	default:
		return ResultPart{}, fmt.Errorf("unsupported tool result part %q", part.Type)
	}
	return record, nil
}

// encodeExtra keeps values of plain kinds (strings, booleans, numbers, bytes,
// JSON objects and arrays). Other values are output metadata of model
// components and do not enter the checkpoint.
func encodeExtra(extra map[string]any) (map[string]Value, error) {
	if len(extra) == 0 {
		return nil, nil
	}
	values := make(map[string]Value, len(extra))
	for key, value := range extra {
		var kind string
		var plain any
		switch typed := value.(type) {
		case []byte:
			kind, plain = "bytes", typed
		case map[string]any, []any:
			kind, plain = "json", typed
		default:
			v := reflect.ValueOf(value)
			switch v.Kind() {
			case reflect.String:
				kind, plain = "string", v.String()
			case reflect.Bool:
				kind, plain = "bool", v.Bool()
			case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
				kind, plain = "int", v.Int()
			case reflect.Float32, reflect.Float64:
				kind, plain = "float", v.Float()
			default:
				continue
			}
		}
		data, err := json.Marshal(plain)
		if err != nil {
			return nil, fmt.Errorf("extra %q: %w", key, err)
		}
		values[key] = Value{Kind: kind, Value: data}
	}
	if len(values) == 0 {
		return nil, nil
	}
	return values, nil
}

// decodeMessage decodes one message.
func decodeMessage(record Message) (*schema.AgenticMessage, error) {
	message := &schema.AgenticMessage{}
	switch record.Role {
	case roleSystem:
		message.Role = schema.AgenticRoleTypeSystem
	case roleUser:
		message.Role = schema.AgenticRoleTypeUser
	case roleAssistant:
		message.Role = schema.AgenticRoleTypeAssistant
	default:
		return nil, fmt.Errorf("unsupported role %q", record.Role)
	}
	extra, err := decodeExtra(record.Extra)
	if err != nil {
		return nil, err
	}
	message.Extra = extra
	if u := record.Usage; u != nil {
		message.ResponseMeta = &schema.AgenticResponseMeta{TokenUsage: &schema.TokenUsage{
			PromptTokens: u.PromptTokens, PromptTokenDetails: schema.PromptTokenDetails{CachedTokens: u.CachedTokens},
			CompletionTokens: u.CompletionTokens, CompletionTokensDetails: schema.CompletionTokensDetails{ReasoningTokens: u.ReasoningTokens},
			TotalTokens: u.TotalTokens,
		}}
	}
	for _, b := range record.Blocks {
		block, err := decodeBlock(b)
		if err != nil {
			return nil, err
		}
		message.ContentBlocks = append(message.ContentBlocks, block)
	}
	return message, nil
}

// decodeBlock decodes one content block.
func decodeBlock(record Block) (*schema.ContentBlock, error) {
	m := record.Media
	if m == nil {
		m = &Media{}
	}
	var block *schema.ContentBlock
	switch record.Type {
	case blockUserText:
		block = schema.NewContentBlock(&schema.UserInputText{Text: record.Text})
	case blockUserImage:
		block = schema.NewContentBlock(&schema.UserInputImage{URL: m.URL, Base64Data: m.Data, MIMEType: m.MIMEType, Detail: schema.ImageURLDetail(m.Detail)})
	case blockUserAudio:
		block = schema.NewContentBlock(&schema.UserInputAudio{URL: m.URL, Base64Data: m.Data, MIMEType: m.MIMEType})
	case blockUserVideo:
		block = schema.NewContentBlock(&schema.UserInputVideo{URL: m.URL, Base64Data: m.Data, MIMEType: m.MIMEType})
	case blockUserFile:
		block = schema.NewContentBlock(&schema.UserInputFile{URL: m.URL, Base64Data: m.Data, MIMEType: m.MIMEType, Name: m.Name})
	case blockAssistantText:
		block = schema.NewContentBlock(&schema.AssistantGenText{Text: record.Text})
	case blockReasoning:
		r := &schema.Reasoning{Text: record.Text, Signature: record.Signature}
		if len(record.ReasoningContent) > 0 {
			r.OpenAIExtension = &openaischema.ReasoningExtension{}
			for _, text := range record.ReasoningContent {
				r.OpenAIExtension.Content = append(r.OpenAIExtension.Content, &openaischema.ReasoningContent{Text: text})
			}
		}
		block = schema.NewContentBlock(r)
	case blockToolCall:
		if record.Call == nil {
			return nil, fmt.Errorf("block %q lacks its call", record.Type)
		}
		block = schema.NewContentBlock(&schema.FunctionToolCall{CallID: record.Call.CallID, Name: record.Call.Name, Arguments: record.Call.Arguments})
	case blockToolResult:
		if record.Result == nil {
			return nil, fmt.Errorf("block %q lacks its result", record.Type)
		}
		result := &schema.FunctionToolResult{CallID: record.Result.CallID, Name: record.Result.Name}
		for _, p := range record.Result.Content {
			part, err := decodePart(p)
			if err != nil {
				return nil, err
			}
			result.Content = append(result.Content, part)
		}
		block = schema.NewContentBlock(result)
	default:
		return nil, fmt.Errorf("unsupported content block type %q", record.Type)
	}
	extra, err := decodeExtra(record.Extra)
	if err != nil {
		return nil, err
	}
	block.Extra = extra
	return block, nil
}

// decodePart decodes one part of a tool result.
func decodePart(record ResultPart) (*schema.FunctionToolResultContentBlock, error) {
	m := record.Media
	if m == nil {
		m = &Media{}
	}
	part := &schema.FunctionToolResultContentBlock{}
	switch record.Type {
	case partText:
		part.Type, part.Text = schema.FunctionToolResultContentBlockTypeText, &schema.UserInputText{Text: record.Text}
	case partImage:
		part.Type, part.Image = schema.FunctionToolResultContentBlockTypeImage, &schema.UserInputImage{URL: m.URL, Base64Data: m.Data, MIMEType: m.MIMEType, Detail: schema.ImageURLDetail(m.Detail)}
	case partAudio:
		part.Type, part.Audio = schema.FunctionToolResultContentBlockTypeAudio, &schema.UserInputAudio{URL: m.URL, Base64Data: m.Data, MIMEType: m.MIMEType}
	case partVideo:
		part.Type, part.Video = schema.FunctionToolResultContentBlockTypeVideo, &schema.UserInputVideo{URL: m.URL, Base64Data: m.Data, MIMEType: m.MIMEType}
	case partFile:
		part.Type, part.File = schema.FunctionToolResultContentBlockTypeFile, &schema.UserInputFile{URL: m.URL, Base64Data: m.Data, MIMEType: m.MIMEType, Name: m.Name}
	default:
		return nil, fmt.Errorf("unsupported tool result part %q", record.Type)
	}
	extra, err := decodeExtra(record.Extra)
	if err != nil {
		return nil, err
	}
	part.Extra = extra
	return part, nil
}

// decodeExtra restores Extra values with their kinds: strings as string,
// integers as int, floats as float64, bytes as []byte and JSON as generic
// objects or arrays.
func decodeExtra(values map[string]Value) (map[string]any, error) {
	if len(values) == 0 {
		return nil, nil
	}
	extra := make(map[string]any, len(values))
	for key, value := range values {
		var target any
		switch value.Kind {
		case "string":
			target = new(string)
		case "bool":
			target = new(bool)
		case "int":
			target = new(int)
		case "float":
			target = new(float64)
		case "bytes":
			target = new([]byte)
		case "json":
			target = new(any)
		default:
			return nil, fmt.Errorf("unsupported extra kind %q for %q", value.Kind, key)
		}
		if err := json.Unmarshal(value.Value, target); err != nil {
			return nil, fmt.Errorf("extra %q: %w", key, err)
		}
		extra[key] = reflect.ValueOf(target).Elem().Interface()
	}
	return extra, nil
}
