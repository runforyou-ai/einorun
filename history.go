package einorun

import (
	"context"
	"fmt"
	"log/slog"
	"slices"
	"sync"
	"sync/atomic"

	"github.com/cloudwego/eino/schema"

	"github.com/runforyou-ai/einorun/llm"
)

const (
	// mediaWindowPercent bounds inline media to this share of the window.
	mediaWindowPercent = 20
	// mediaTokens is the token estimate of one image or media item.
	mediaTokens = 1280
	// maxMediaBytes bounds a single inline item.
	maxMediaBytes = 10 << 20
	// maxRunMediaBytes bounds the inline media of a run.
	maxRunMediaBytes = 20 << 20
	// historyWindowPercent bounds claimed history to this share of the window.
	historyWindowPercent = 50
)

// inlineTypes lists the formats that may be sent to a model, by modality.
var inlineTypes = map[string]llm.Modality{
	"image/png":       llm.Image,
	"image/jpeg":      llm.Image,
	"image/webp":      llm.Image,
	"image/gif":       llm.Image,
	"audio/wav":       llm.Audio,
	"audio/wave":      llm.Audio,
	"audio/vnd.wav":   llm.Audio,
	"audio/vnd.wave":  llm.Audio,
	"audio/x-pn-wav":  llm.Audio,
	"video/mp4":       llm.Video,
	"video/webm":      llm.Video,
	"video/quicktime": llm.Video,
	"application/pdf": llm.File,
}

// mediaPolicy is how media reaches the model.
type mediaPolicy struct {
	read         MediaReader
	url          MediaURLFunc
	adapt        MediaAdapter
	modalities   map[llm.Modality]bool
	maxCount     int
	enabled      *atomic.Bool // false once the model rejected native media
	budget       *mediaBudget
	projections  *projectionCache
	text         *Text
}

// projectionCache remembers text projections by media key and digest so a
// resumed run does not call the adapter again.
type projectionCache struct {
	mu    sync.Mutex
	texts map[string]string
}

func newProjectionCache() *projectionCache {
	return &projectionCache{texts: map[string]string{}}
}

func projectionKey(ref MediaRef) string {
	return ref.Key + "\x00" + ref.SHA256
}

func (c *projectionCache) get(ref MediaRef) (string, bool) {
	if c == nil {
		return "", false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	text, ok := c.texts[projectionKey(ref)]
	return text, ok
}

func (c *projectionCache) put(ref MediaRef, text string) {
	if c == nil || text == "" {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.texts[projectionKey(ref)] = text
}

func (c *projectionCache) snapshot() map[string]string {
	if c == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.texts) == 0 {
		return nil
	}
	out := make(map[string]string, len(c.texts))
	for k, v := range c.texts {
		out[k] = v
	}
	return out
}

func (c *projectionCache) restore(texts map[string]string) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.texts = map[string]string{}
	for k, v := range texts {
		c.texts[k] = v
	}
}

// mediaBudget counts the media sent inline in a run, shared by input
// attachments and tool results of every agent.
type mediaBudget struct {
	mu    sync.Mutex
	count int
	bytes int64
}

// reserve takes room for one item of size, and reports whether there was any.
func (b *mediaBudget) reserve(size int64, maxCount int) bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.count >= maxCount || size > maxMediaBytes || b.bytes+size > maxRunMediaBytes {
		return false
	}
	b.count++
	b.bytes += size
	return true
}

// used returns the items and bytes sent.
func (b *mediaBudget) used() (int, int64) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.count, b.bytes
}

// restore sets the items and bytes sent.
func (b *mediaBudget) restore(count int, bytes int64) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.count, b.bytes = count, bytes
}

// mediaMaxCount is the number of inline items a window allows, at least one.
func mediaMaxCount(window int) int {
	return max(1, window*mediaWindowPercent/100/mediaTokens)
}

// inputExtraKey marks the model messages made from claimed input.
const inputExtraKey = "einorun_input"

// IsInput reports whether message was made from claimed conversation input,
// as opposed to messages the runtime adds itself: tool results, media
// carriers, summaries, notices and corrections.
func IsInput(message *schema.AgenticMessage) bool {
	if message == nil {
		return false
	}
	marked, _ := message.Extra[inputExtraKey].(bool)
	return marked
}

// history is the model history of a run: claimed conversation messages and
// complete tool interactions, in order.
type history struct {
	messages []*schema.AgenticMessage
	seen     map[string]bool
}

// appendInput adds the messages not yet in the history, by ID and revision,
// after the previous turn. Supported user attachments are sent inline (URL
// first), newest first, within the media budget; unsupported media is
// projected when an adapter is set, otherwise stays links in the text.
func (h *history) appendInput(ctx context.Context, messages []Message, media mediaPolicy) []*schema.AgenticMessage {
	if h.seen == nil {
		h.seen = map[string]bool{}
	}
	fresh := make([]Message, 0, len(messages))
	for _, m := range messages {
		key := m.ID + "@" + m.Revision
		if !h.seen[key] {
			h.seen[key] = true
			fresh = append(fresh, m)
		}
	}
	inline := map[int]*schema.AgenticMessage{}
	if media.read != nil || media.url != nil || media.adapt != nil {
		for i := len(fresh) - 1; i >= 0; i-- {
			ref := fresh[i].Media
			if ref == nil || fresh[i].Role != RoleUser {
				continue
			}
			message, ok := media.materializeInput(ctx, fresh[i])
			if !ok {
				continue
			}
			inline[i] = message
		}
	}
	for i, m := range fresh {
		var message *schema.AgenticMessage
		switch {
		case m.Role == RoleAssistant:
			message = &schema.AgenticMessage{Role: schema.AgenticRoleTypeAssistant,
				ContentBlocks: []*schema.ContentBlock{schema.NewContentBlock(&schema.AssistantGenText{Text: m.Content})}}
		case inline[i] != nil:
			message = inline[i]
		default:
			message = schema.UserAgenticMessage(m.Content)
		}
		message.Extra = map[string]any{inputExtraKey: true}
		h.messages = append(h.messages, message)
	}
	return slices.Clone(h.messages)
}

// appendOutput keeps model outputs with text or tool calls and all tool
// results.
func (h *history) appendOutput(messages []*schema.AgenticMessage) {
	for _, m := range messages {
		if m.Role != schema.AgenticRoleTypeAssistant || hasToolCalls(m) || llm.Text(m) != "" {
			h.messages = append(h.messages, m)
		}
	}
}

// fillResults adds results that only the model context has (patched on
// recovery) after their calls in the history.
func (h *history) fillResults(context []*schema.AgenticMessage) {
	present := map[string]bool{}
	for _, m := range h.messages {
		for _, id := range resultCallIDs(m) {
			present[id] = true
		}
	}
	patched := map[string]*schema.AgenticMessage{}
	for _, m := range context {
		for _, id := range resultCallIDs(m) {
			if !present[id] {
				patched[id] = m
			}
		}
	}
	if len(patched) == 0 {
		return
	}
	filled := make([]*schema.AgenticMessage, 0, len(h.messages)+len(patched))
	for i := 0; i < len(h.messages); i++ {
		m := h.messages[i]
		filled = append(filled, m)
		if m.Role != schema.AgenticRoleTypeAssistant || !hasToolCalls(m) {
			continue
		}
		for i+1 < len(h.messages) && len(resultCallIDs(h.messages[i+1])) > 0 {
			i++
			filled = append(filled, h.messages[i])
		}
		added := map[*schema.AgenticMessage]bool{}
		for _, call := range toolCalls(m) {
			if result, ok := patched[call.CallID]; ok && !added[result] {
				filled = append(filled, result)
				added[result] = true
			}
		}
	}
	h.messages = filled
}

// resultCallIDs returns the call identifiers of the tool results in m.
func resultCallIDs(m *schema.AgenticMessage) []string {
	var ids []string
	for _, b := range m.ContentBlocks {
		if b != nil && b.Type == schema.ContentBlockTypeFunctionToolResult && b.FunctionToolResult != nil {
			ids = append(ids, b.FunctionToolResult.CallID)
		}
	}
	return ids
}

// trimHistory keeps the newest claimed messages within the history budget;
// the newest message is always kept.
func trimHistory(ctx context.Context, messages []Message, window int) []Message {
	budget := window * historyWindowPercent / 100
	total := 0
	for i := len(messages) - 1; i >= 0; i-- {
		total += llm.EstimateTokens(messages[i].Content)
		if total <= budget || i == len(messages)-1 {
			continue
		}
		slog.WarnContext(ctx, "einorun: claimed history exceeds its budget, keeping the newest messages",
			"budget_tokens", budget, "messages", len(messages), "kept", len(messages)-i-1)
		return messages[i+1:]
	}
	return messages
}

// carriesMedia reports whether input has inline media or non-text tool
// results.
func carriesMedia(input []*schema.AgenticMessage) bool {
	for _, m := range input {
		for _, b := range m.ContentBlocks {
			if b == nil {
				continue
			}
			switch b.Type {
			case schema.ContentBlockTypeUserInputImage, schema.ContentBlockTypeUserInputAudio, schema.ContentBlockTypeUserInputVideo, schema.ContentBlockTypeUserInputFile:
				return true
			case schema.ContentBlockTypeFunctionToolResult:
				if b.FunctionToolResult == nil {
					continue
				}
				for _, c := range b.FunctionToolResult.Content {
					if c != nil && c.Type != schema.FunctionToolResultContentBlockTypeText {
						return true
					}
				}
			}
		}
	}
	return false
}

// withoutMedia returns input without media: inline attachments are dropped
// (their messages keep the text) and non-text tool results become notes.
func withoutMedia(input []*schema.AgenticMessage, text *Text) []*schema.AgenticMessage {
	output := make([]*schema.AgenticMessage, len(input))
	for i, m := range input {
		if !carriesMedia([]*schema.AgenticMessage{m}) {
			output[i] = m
			continue
		}
		stripped := *m
		stripped.ContentBlocks = make([]*schema.ContentBlock, 0, len(m.ContentBlocks))
		for _, b := range m.ContentBlocks {
			switch {
			case b.Type == schema.ContentBlockTypeUserInputImage, b.Type == schema.ContentBlockTypeUserInputAudio, b.Type == schema.ContentBlockTypeUserInputVideo, b.Type == schema.ContentBlockTypeUserInputFile:
			case b.Type == schema.ContentBlockTypeFunctionToolResult && b.FunctionToolResult != nil:
				result := *b.FunctionToolResult
				result.Content = make([]*schema.FunctionToolResultContentBlock, 0, len(b.FunctionToolResult.Content))
				for _, c := range b.FunctionToolResult.Content {
					if c != nil && c.Type != schema.FunctionToolResultContentBlockTypeText {
						c = &schema.FunctionToolResultContentBlock{Type: schema.FunctionToolResultContentBlockTypeText,
							Text: &schema.UserInputText{Text: fmt.Sprintf(text.MediaUnavailable, c.Type)}}
					}
					result.Content = append(result.Content, c)
				}
				copied := *b
				copied.FunctionToolResult = &result
				stripped.ContentBlocks = append(stripped.ContentBlocks, &copied)
			default:
				stripped.ContentBlocks = append(stripped.ContentBlocks, b)
			}
		}
		output[i] = &stripped
	}
	return output
}

// withoutSystem returns messages without the system instruction, which the
// agent adds on every run.
func withoutSystem(messages []*schema.AgenticMessage) []*schema.AgenticMessage {
	return slices.DeleteFunc(slices.Clone(messages), func(m *schema.AgenticMessage) bool {
		return m.Role == schema.AgenticRoleTypeSystem
	})
}
