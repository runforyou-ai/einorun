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

	"github.com/runforyou-ai/einorun/llm"
)

// mediaExtraKey marks the user message that carries the media of a tool
// result, by the provider call ID.
const mediaExtraKey = "einorun_media_for"

// errMediaChanged reports media whose digest no longer matches its reference.
var errMediaChanged = errors.New("einorun: media changed since it was referenced")

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
			return nil, fmt.Errorf("%w: %s", errMediaChanged, ref.Key)
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
// otherwise as a user message right after it. Newer media is chosen first
// within the run's budget; media that cannot be passed becomes a note saying
// why.
type mediaInjector struct {
	adk.TypedBaseChatModelAgentMiddleware[*schema.AgenticMessage]
	policy   mediaPolicy
	inResult bool
	text     *Text

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

// item is one piece of media of a ready call and what becomes of it.
type item struct {
	ref      MediaRef
	modality llm.Modality
	data     []byte
	note     string
}

// BeforeModelRewriteState passes pending media whose tool results are in the
// context. Media is read outside the lock.
func (m *mediaInjector) BeforeModelRewriteState(ctx context.Context, state *adk.TypedChatModelAgentState[*schema.AgenticMessage], _ *adk.TypedModelContext[*schema.AgenticMessage]) (context.Context, *adk.TypedChatModelAgentState[*schema.AgenticMessage], error) {
	var order []string
	taken := map[string][]MediaRef{}
	m.mu.Lock()
	for _, message := range state.Messages {
		for _, id := range resultCallIDs(message) {
			if refs, ok := m.pending[id]; ok {
				taken[id] = refs
				order = append(order, id)
				delete(m.pending, id)
			}
		}
	}
	m.mu.Unlock()
	if len(order) == 0 {
		return ctx, state, nil
	}
	// Read newest first, so the budget keeps the latest media.
	items := map[string][]*item{}
	for i := len(order) - 1; i >= 0; i-- {
		refs := taken[order[i]]
		loaded := make([]*item, len(refs))
		for j := len(refs) - 1; j >= 0; j-- {
			loaded[j] = m.load(ctx, refs[j])
		}
		items[order[i]] = loaded
	}
	messages := make([]*schema.AgenticMessage, 0, len(state.Messages)+len(order))
	injected := map[string]*schema.AgenticMessage{}
	results := map[string]*schema.AgenticMessage{}
	// Carriers follow the whole run of tool results of a batch, so that the
	// results stay together.
	var carriers []*schema.AgenticMessage
	flush := func() {
		messages = append(messages, carriers...)
		carriers = nil
	}
	for _, message := range state.Messages {
		ids := resultCallIDs(message)
		if len(ids) == 0 {
			flush()
		}
		var ready []string
		for _, id := range ids {
			if _, ok := items[id]; ok {
				ready = append(ready, id)
			}
		}
		if len(ready) == 0 {
			messages = append(messages, message)
			continue
		}
		if m.inResult {
			message = intoResults(message, ready, items)
			for _, id := range ready {
				results[id] = message
			}
			messages = append(messages, message)
			continue
		}
		messages = append(messages, message)
		for _, id := range ready {
			carrier := &schema.AgenticMessage{Role: schema.AgenticRoleTypeUser, Extra: map[string]any{mediaExtraKey: id}}
			for _, it := range items[id] {
				if it.data != nil {
					carrier.ContentBlocks = append(carrier.ContentBlocks, inputBlock(it.modality, it.data, it.ref.MIME))
				} else {
					carrier.ContentBlocks = append(carrier.ContentBlocks, schema.NewContentBlock(&schema.UserInputText{Text: it.note}))
				}
			}
			injected[id] = carrier
			carriers = append(carriers, carrier)
		}
	}
	flush()
	m.mu.Lock()
	maps.Copy(m.injected, injected)
	if len(results) > 0 {
		if m.results == nil {
			m.results = map[string]*schema.AgenticMessage{}
		}
		maps.Copy(m.results, results)
	}
	m.mu.Unlock()
	state.Messages = messages
	return ctx, state, nil
}

// intoResults adds the media of the ready calls to their result blocks.
func intoResults(message *schema.AgenticMessage, ready []string, items map[string][]*item) *schema.AgenticMessage {
	copied := *message
	copied.ContentBlocks = slices.Clone(message.ContentBlocks)
	for i, b := range copied.ContentBlocks {
		if b == nil || b.Type != schema.ContentBlockTypeFunctionToolResult || !slices.Contains(ready, b.FunctionToolResult.CallID) {
			continue
		}
		result := *b.FunctionToolResult
		result.Content = slices.Clone(result.Content)
		for _, it := range items[result.CallID] {
			if it.data != nil {
				result.Content = append(result.Content, resultPart(it.modality, it.data, it.ref.MIME))
			} else {
				result.Content = append(result.Content, &schema.FunctionToolResultContentBlock{Type: schema.FunctionToolResultContentBlockTypeText, Text: &schema.UserInputText{Text: it.note}})
			}
		}
		block := *b
		block.FunctionToolResult = &result
		copied.ContentBlocks[i] = &block
	}
	return &copied
}

// load reads media the model can view within the budget, charged by the bytes
// actually read, or returns the note that replaces it.
func (m *mediaInjector) load(ctx context.Context, ref MediaRef) *item {
	it := &item{ref: ref}
	modality, ok := inlineTypes[ref.MIME]
	if !ok || !m.policy.modalities[modality] || !m.policy.enabled.Load() || m.policy.read == nil {
		it.note = fmt.Sprintf(m.text.MediaUnavailable, cmpOr(ref.MIME, "media"))
		return it
	}
	data, err := readMedia(ctx, m.policy.read, ref)
	switch {
	case errors.Is(err, errMediaChanged):
		it.note = fmt.Sprintf(m.text.MediaChanged, ref.Key)
		return it
	case err != nil:
		slog.WarnContext(ctx, "einorun: reading tool result media failed", "key", ref.Key, "error", err)
		it.note = fmt.Sprintf(m.text.MediaUnavailable, ref.MIME)
		return it
	}
	if !m.policy.budget.reserve(int64(len(data)), m.policy.maxCount) {
		it.note = fmt.Sprintf(m.text.MediaOverBudget, ref.Key)
		return it
	}
	it.modality, it.data = modality, data
	return it
}

// apply puts the passed media into the turn history the same way: carrier
// messages after their results, or result messages with media inside. With
// media turned off, the history keeps notes only.
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
	var carriers []*schema.AgenticMessage
	for _, message := range messages {
		ids := resultCallIDs(message)
		if len(ids) == 0 {
			out = append(out, carriers...)
			carriers = nil
		}
		replaced := message
		for _, id := range ids {
			if withMedia, ok := m.results[id]; ok {
				replaced = withMedia
			}
		}
		out = append(out, replaced)
		for _, id := range ids {
			if carrier, ok := m.injected[id]; ok && !present[id] {
				carriers = append(carriers, carrier)
				present[id] = true
			}
		}
	}
	out = append(out, carriers...)
	if !m.policy.enabled.Load() {
		return withoutMedia(out, m.text)
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
