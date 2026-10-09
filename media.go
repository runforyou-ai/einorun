package einorun

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"slices"
	"strings"
	"sync"

	"github.com/cloudwego/eino/adk"
	"github.com/cloudwego/eino/schema"

	"github.com/runforyou-ai/einorun/internal/prompt"
	"github.com/runforyou-ai/einorun/llm"
)

// mediaExtraKey marks the user message that carries the media of a tool
// result, by the provider call ID.
const mediaExtraKey = "einorun_media_for"

// WithMedia is returned by tools whose result includes media the host
// stores: text is the result and refs the media, in order. The refs are
// saved with the call; the runtime reads the media back before the next model
// call and passes it to the model within the run's media budget.
func WithMedia(text string, refs ...MediaRef) error {
	return &control{media: &mediaResult{text: text, refs: slices.Clone(refs)}}
}

// mediaResult is a result with media.
type mediaResult struct {
	text string
	refs []MediaRef
}

// readMedia reads media and checks its digest when the reference has one.
func readMedia(ctx context.Context, read MediaReader, ref MediaRef) ([]byte, error) {
	if read == nil {
		return nil, errors.New("einorun: the request has no media reader")
	}
	data, err := read(ctx, ref)
	if err != nil {
		return nil, err
	}
	if ref.SHA256 != "" {
		sum := sha256.Sum256(data)
		if !strings.EqualFold(hex.EncodeToString(sum[:]), ref.SHA256) {
			return nil, fmt.Errorf("einorun: media %s changed since it was referenced", ref.Key)
		}
	}
	return data, nil
}

// inputBlock returns a user input block with media.
func inputBlock(modality llm.Modality, data []byte, mime string) *schema.ContentBlock {
	encoded := base64.StdEncoding.EncodeToString(data)
	switch modality {
	case llm.Audio:
		return schema.NewContentBlock(&schema.UserInputAudio{Base64Data: encoded, MIMEType: mime})
	case llm.Video:
		return schema.NewContentBlock(&schema.UserInputVideo{Base64Data: encoded, MIMEType: mime})
	default:
		return schema.NewContentBlock(&schema.UserInputImage{Base64Data: encoded, MIMEType: mime})
	}
}

// resultPart returns a tool result part with media.
func resultPart(modality llm.Modality, data []byte, mime string) *schema.FunctionToolResultContentBlock {
	encoded := base64.StdEncoding.EncodeToString(data)
	switch modality {
	case llm.Audio:
		return &schema.FunctionToolResultContentBlock{Type: schema.FunctionToolResultContentBlockTypeAudio, Audio: &schema.UserInputAudio{Base64Data: encoded, MIMEType: mime}}
	case llm.Video:
		return &schema.FunctionToolResultContentBlock{Type: schema.FunctionToolResultContentBlockTypeVideo, Video: &schema.UserInputVideo{Base64Data: encoded, MIMEType: mime}}
	default:
		return &schema.FunctionToolResultContentBlock{Type: schema.FunctionToolResultContentBlockTypeImage, Image: &schema.UserInputImage{Base64Data: encoded, MIMEType: mime}}
	}
}

// mediaInjector passes the media of tool results to the model before the
// next model call: inside the tool result when the model accepts media there,
// otherwise as a user message right after it. Media the model cannot view,
// cannot be read or exceeds the budget becomes a note.
type mediaInjector struct {
	adk.TypedBaseChatModelAgentMiddleware[*schema.AgenticMessage]
	policy   mediaPolicy
	inResult bool
	text     *prompt.Runtime

	mu       sync.Mutex
	pending  map[string][]MediaRef             // provider call ID -> media not yet passed
	injected map[string]*schema.AgenticMessage // provider call ID -> message carrying its media
	results  map[string]*schema.AgenticMessage // provider call ID -> tool result message with media inside
}

// add registers the media of a call.
func (m *mediaInjector) add(callID string, refs []MediaRef) {
	if len(refs) == 0 {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.pending[callID] = slices.Clone(refs)
}

// pendingRefs returns the media not yet passed, for the checkpoint.
func (m *mediaInjector) pendingRefs() map[string][]MediaRef {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.pending) == 0 {
		return nil
	}
	return maps.Clone(m.pending)
}

// restore restores media not yet passed.
func (m *mediaInjector) restore(pending map[string][]MediaRef) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.pending = maps.Clone(pending)
	if m.pending == nil {
		m.pending = map[string][]MediaRef{}
	}
}

// BeforeModelRewriteState passes pending media whose tool results are in the
// context.
func (m *mediaInjector) BeforeModelRewriteState(ctx context.Context, state *adk.TypedChatModelAgentState[*schema.AgenticMessage], _ *adk.TypedModelContext[*schema.AgenticMessage]) (context.Context, *adk.TypedChatModelAgentState[*schema.AgenticMessage], error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.pending) == 0 {
		return ctx, state, nil
	}
	messages := make([]*schema.AgenticMessage, 0, len(state.Messages)+len(m.pending))
	for _, message := range state.Messages {
		var ready []string
		for _, id := range resultCallIDs(message) {
			if _, ok := m.pending[id]; ok {
				ready = append(ready, id)
			}
		}
		if len(ready) == 0 {
			messages = append(messages, message)
			continue
		}
		if m.inResult {
			message = m.intoResults(ctx, message, ready)
			messages = append(messages, message)
			continue
		}
		messages = append(messages, message)
		for _, id := range ready {
			carrier := m.carrier(ctx, id, m.pending[id])
			m.injected[id] = carrier
			delete(m.pending, id)
			messages = append(messages, carrier)
		}
	}
	state.Messages = messages
	return ctx, state, nil
}

// intoResults adds the media of the ready calls to their result blocks.
func (m *mediaInjector) intoResults(ctx context.Context, message *schema.AgenticMessage, ready []string) *schema.AgenticMessage {
	copied := *message
	copied.ContentBlocks = slices.Clone(message.ContentBlocks)
	for i, b := range copied.ContentBlocks {
		if b == nil || b.Type != schema.ContentBlockTypeFunctionToolResult || !slices.Contains(ready, b.FunctionToolResult.CallID) {
			continue
		}
		id := b.FunctionToolResult.CallID
		result := *b.FunctionToolResult
		result.Content = slices.Clone(result.Content)
		for _, ref := range m.pending[id] {
			if part, note := m.load(ctx, ref); part != nil {
				result.Content = append(result.Content, resultPart(part.modality, part.data, ref.MIME))
			} else {
				result.Content = append(result.Content, &schema.FunctionToolResultContentBlock{Type: schema.FunctionToolResultContentBlockTypeText, Text: &schema.UserInputText{Text: note}})
			}
		}
		block := *b
		block.FunctionToolResult = &result
		copied.ContentBlocks[i] = &block
		delete(m.pending, id)
	}
	for _, id := range ready {
		if m.results == nil {
			m.results = map[string]*schema.AgenticMessage{}
		}
		m.results[id] = &copied
	}
	return &copied
}

// carrier returns the user message carrying the media of a call.
func (m *mediaInjector) carrier(ctx context.Context, callID string, refs []MediaRef) *schema.AgenticMessage {
	message := &schema.AgenticMessage{Role: schema.AgenticRoleTypeUser, Extra: map[string]any{mediaExtraKey: callID}}
	for _, ref := range refs {
		if part, note := m.load(ctx, ref); part != nil {
			message.ContentBlocks = append(message.ContentBlocks, inputBlock(part.modality, part.data, ref.MIME))
		} else {
			message.ContentBlocks = append(message.ContentBlocks, schema.NewContentBlock(&schema.UserInputText{Text: note}))
		}
	}
	return message
}

// loaded is media read for the model.
type loaded struct {
	modality llm.Modality
	data     []byte
}

// load reads media the model can view within the budget, or returns the note
// that replaces it.
func (m *mediaInjector) load(ctx context.Context, ref MediaRef) (*loaded, string) {
	modality, ok := inlineTypes[ref.MIME]
	note := fmt.Sprintf(m.text.MediaUnavailable, cmpOr(ref.MIME, "media"))
	if !ok || !m.policy.modalities[modality] || !m.policy.enabled.Load() || m.policy.read == nil {
		return nil, note
	}
	if !m.policy.budget.reserve(ref.Size, m.policy.maxCount) {
		return nil, note
	}
	data, err := readMedia(ctx, m.policy.read, ref)
	if err != nil {
		m.policy.budget.release(ref.Size)
		slog.WarnContext(ctx, "einorun: reading tool result media failed", "key", ref.Key, "error", err)
		return nil, note
	}
	return &loaded{modality: modality, data: data}, ""
}

// apply puts the passed media into the turn history the same way: carrier
// messages after their results, or result messages with media inside.
func (m *mediaInjector) apply(messages []*schema.AgenticMessage) []*schema.AgenticMessage {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.injected) == 0 && len(m.results) == 0 {
		return messages
	}
	present := map[string]bool{}
	for _, message := range messages {
		if id, ok := message.Extra[mediaExtraKey].(string); ok {
			present[id] = true
		}
	}
	out := make([]*schema.AgenticMessage, 0, len(messages)+len(m.injected))
	for _, message := range messages {
		ids := resultCallIDs(message)
		replaced := message
		for _, id := range ids {
			if withMedia, ok := m.results[id]; ok {
				replaced = withMedia
			}
		}
		out = append(out, replaced)
		for _, id := range ids {
			if carrier, ok := m.injected[id]; ok && !present[id] {
				out = append(out, carrier)
				present[id] = true
			}
		}
	}
	return out
}

// settle forgets passed media once the history holds it.
func (m *mediaInjector) settle() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.injected = map[string]*schema.AgenticMessage{}
	m.results = nil
}

// cmpOr returns a when it is not empty, otherwise b.
func cmpOr(a, b string) string {
	if a != "" {
		return a
	}
	return b
}
