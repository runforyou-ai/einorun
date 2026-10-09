package einorun

import (
	"cmp"
	"context"
	"fmt"
	"log/slog"
	"maps"
	"reflect"
	"slices"
	"strings"
	"sync"

	"github.com/cloudwego/eino/adk"
	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"

	"github.com/runforyou-ai/einorun/llm"
	"github.com/runforyou-ai/einorun/stream"
)

// toolEntry is what the runtime knows about a registered tool, by its
// model-visible name.
type toolEntry struct {
	spec ToolSpec
	name string
	// describe returns the description shown for a call, by its arguments.
	describe func(arguments string) string
	// builtin marks a tool the runtime or a built-in extension added.
	builtin bool
}

// recorder keeps the process of one execution attempt: blocks in model
// order, sub-agent calls, the candidate reply and the plan. It publishes
// display changes and hands changes to the journal at safe points.
type recorder struct {
	adk.TypedBaseChatModelAgentMiddleware[*schema.AgenticMessage]
	mu             sync.Mutex
	process        []Block
	children       []ToolCall
	childPositions map[string]int // parent record ID + "/" + provider call ID
	candidate      string
	toolPositions  map[string]int // provider call ID of main-agent calls
	tools          map[string]*toolEntry
	activity       map[string]string // record ID of a delegation call -> tool a sub-agent is calling
	plan           []PlanTask
	changes        changeSet
	call           *modelCall
	pub            *publisher
	journal        Journal
	usage          llm.Usage
	runID          string
	// onStep saves a safe point after a model output is finalized.
	onStep func(ctx context.Context, messages []*schema.AgenticMessage) error
}

// modelCall maps the stream chunk indices of an unfinished model call to
// block positions.
type modelCall struct {
	id             string
	start          int
	indices        []int
	positions      map[int]int
	nextOrdinal    int
	candidateParts []candidatePart
	hasTools       bool
}

// candidatePart is candidate text from one chunk index.
type candidatePart struct {
	index int
	text  string
}

// newRecorder returns a recorder for one execution attempt.
func newRecorder(request Request, streamID string) *recorder {
	return &recorder{
		toolPositions: map[string]int{}, childPositions: map[string]int{}, tools: map[string]*toolEntry{}, activity: map[string]string{},
		changes: newChangeSet(), pub: &publisher{stream: streamID, sink: request.Stream}, journal: request.Journal, runID: request.RunID,
	}
}

// entry returns the registered tool with the model-visible name.
func (r *recorder) entry(name string) (*toolEntry, bool) {
	entry, ok := r.tools[name]
	return entry, ok
}

// nameCall fills in the record name, traits and notes of a call. Unknown
// tools are treated as not replayable with side effects.
func (r *recorder) nameCall(call *ToolCall, name string) {
	call.Name = name
	entry, ok := r.entry(name)
	if !ok {
		call.Replayable, call.SideEffects = false, true
		return
	}
	call.Replayable, call.SideEffects = entry.spec.Replayable, entry.spec.SideEffects
	if entry.spec.RecordName != nil {
		call.Name = cmp.Or(entry.spec.RecordName(call.Arguments), name)
	}
	if len(entry.spec.Notes) > 0 {
		if call.Notes == nil {
			call.Notes = map[string]string{}
		}
		maps.Copy(call.Notes, entry.spec.Notes)
	}
}

// touchLocked records a change of call: its revision increases and it is
// registered for the next save.
func (r *recorder) touchLocked(call *ToolCall) {
	call.Rev++
	r.changes.call(call.ID)
}

// streamingModel hands stream chunks to the recorder as they are read.
type streamingModel struct {
	model.BaseModel[*schema.AgenticMessage]
	recorder *recorder
}

// WrapModel records every model call.
func (r *recorder) WrapModel(_ context.Context, m model.BaseModel[*schema.AgenticMessage], _ *adk.TypedModelContext[*schema.AgenticMessage]) (model.BaseModel[*schema.AgenticMessage], error) {
	return &streamingModel{BaseModel: m, recorder: r}, nil
}

// Stream assigns the model call ID and records chunks as they pass.
func (m *streamingModel) Stream(ctx context.Context, input []*schema.AgenticMessage, opts ...model.Option) (*schema.StreamReader[*schema.AgenticMessage], error) {
	id := llm.NewModelCallID()
	reader, err := m.BaseModel.Stream(llm.WithModelCallID(ctx, id), input, opts...)
	if err != nil {
		return nil, err
	}
	m.recorder.mu.Lock()
	m.recorder.beginCallLocked(id)
	m.recorder.mu.Unlock()
	return schema.StreamReaderWithConvert(reader, func(chunk *schema.AgenticMessage) (*schema.AgenticMessage, error) {
		if chunk != nil {
			m.recorder.receive(chunk)
		}
		return chunk, nil
	}), nil
}

// Generate assigns the model call ID.
func (m *streamingModel) Generate(ctx context.Context, input []*schema.AgenticMessage, opts ...model.Option) (*schema.AgenticMessage, error) {
	id := llm.NewModelCallID()
	m.recorder.mu.Lock()
	m.recorder.beginCallLocked(id)
	m.recorder.mu.Unlock()
	return m.BaseModel.Generate(llm.WithModelCallID(ctx, id), input, opts...)
}

// beginCallLocked starts a model call; blocks of a previous call that was
// retried before it was finalized are removed, and the candidate is cleared.
func (r *recorder) beginCallLocked(id string) *modelCall {
	if r.call != nil && len(r.process) > r.call.start {
		removed := make([]string, 0, len(r.process)-r.call.start)
		for _, block := range r.process[r.call.start:] {
			removed = append(removed, block.ID)
			r.changes.remove(block)
		}
		r.process = r.process[:r.call.start]
		r.pub.add(stream.Operation{Kind: stream.OpRemoveBlocks, BlockIDs: removed})
	}
	r.call = &modelCall{id: cmp.Or(id, llm.NewModelCallID()), start: len(r.process), positions: map[int]int{}}
	if r.candidate != "" {
		r.candidate = ""
		r.pub.add(stream.Operation{Kind: stream.OpClearCandidate})
	}
	return r.call
}

// receive accumulates reasoning, text and tool calls by chunk index; text
// before the first tool call is the candidate reply.
func (r *recorder) receive(chunk *schema.AgenticMessage) {
	r.mu.Lock()
	defer r.mu.Unlock()
	call := r.call
	if call == nil {
		return
	}
	for _, block := range chunk.ContentBlocks {
		if block == nil {
			continue
		}
		index := call.nextOrdinal
		if block.StreamingMeta != nil {
			index = block.StreamingMeta.Index
		} else {
			call.nextOrdinal++
		}
		if !slices.Contains(call.indices, index) {
			call.indices = append(call.indices, index)
		}
		position, exists := call.positions[index]
		switch block.Type {
		case schema.ContentBlockTypeReasoning:
			if block.Reasoning == nil || block.Reasoning.Text == "" {
				continue
			}
			if exists {
				r.appendTextLocked(position, block.Reasoning.Text)
			} else {
				r.addBlockLocked(index, Block{Kind: KindThinking, Text: block.Reasoning.Text})
			}
		case schema.ContentBlockTypeAssistantGenText:
			if block.AssistantGenText == nil {
				continue
			}
			text := block.AssistantGenText.Text
			switch {
			case text == "":
			case !call.hasTools:
				if last := len(call.candidateParts) - 1; last >= 0 && call.candidateParts[last].index == index {
					call.candidateParts[last].text += text
				} else {
					call.candidateParts = append(call.candidateParts, candidatePart{index: index, text: text})
				}
				r.candidate += text
				r.pub.add(stream.Operation{Kind: stream.OpAppendCandidate, Text: text})
			case exists:
				r.appendTextLocked(position, text)
			default:
				r.addBlockLocked(index, Block{Kind: KindContent, Text: text})
			}
		case schema.ContentBlockTypeFunctionToolCall:
			// Candidate text before the first tool call becomes explanation blocks.
			if !call.hasTools {
				call.hasTools = true
				if r.candidate != "" {
					r.candidate = ""
					r.pub.add(stream.Operation{Kind: stream.OpClearCandidate})
					for _, part := range call.candidateParts {
						r.addBlockLocked(part.index, Block{Kind: KindContent, Text: part.text})
					}
				}
			}
			chunkCall := block.FunctionToolCall
			if !exists {
				r.addBlockLocked(index, Block{Kind: KindToolCall, Call: r.queuedCall(chunkCall, call.id)})
				continue
			}
			recorded := r.process[position].Call
			recorded.Arguments += chunkCall.Arguments
			r.touchLocked(recorded)
			if (recorded.CallID == "" && chunkCall.CallID != "") || (recorded.Name == "" && chunkCall.Name != "") {
				recorded.CallID = cmp.Or(recorded.CallID, chunkCall.CallID)
				if recorded.Name == "" {
					r.nameCall(recorded, chunkCall.Name)
				}
				r.pub.add(stream.Operation{Kind: stream.OpUpsertBlock, Block: r.view(r.process[position])})
			}
		}
	}
}

// queuedCall returns a queued call record for a call the model made.
func (r *recorder) queuedCall(call *schema.FunctionToolCall, modelCallID string) *ToolCall {
	recorded := &ToolCall{ID: llm.NewModelCallID(), ModelCallID: modelCallID, CallID: call.CallID, Arguments: call.Arguments, Status: StatusQueued}
	r.nameCall(recorded, call.Name)
	return recorded
}

// addBlockLocked appends a block of the current model call.
func (r *recorder) addBlockLocked(index int, block Block) {
	position := len(r.process)
	block.ID, block.Position, block.ModelCallID = llm.NewModelCallID(), int64(position+1), r.call.id
	r.process = append(r.process, block)
	r.call.positions[index] = position
	r.changes.block(block.ID)
	if block.Call != nil {
		r.touchLocked(block.Call)
	}
	r.pub.add(stream.Operation{Kind: stream.OpUpsertBlock, Block: r.view(block)})
}

// appendTextLocked appends text to a block.
func (r *recorder) appendTextLocked(position int, text string) {
	r.process[position].Text += text
	r.changes.block(r.process[position].ID)
	r.pub.add(stream.Operation{Kind: stream.OpAppendText, BlockID: r.process[position].ID, Text: text})
}

// AfterModelRewriteState finalizes the blocks of the model call from the
// complete output, keeping the IDs of streamed blocks, then saves a safe point.
func (r *recorder) AfterModelRewriteState(ctx context.Context, state *adk.TypedChatModelAgentState[*schema.AgenticMessage], _ *adk.TypedModelContext[*schema.AgenticMessage]) (context.Context, *adk.TypedChatModelAgentState[*schema.AgenticMessage], error) {
	r.finalize(ctx, state.Messages[len(state.Messages)-1])
	if r.onStep != nil {
		if err := r.onStep(ctx, state.Messages); err != nil {
			return ctx, state, err
		}
	}
	return ctx, state, nil
}

// finalize replaces the streamed blocks of the current model call with the
// blocks of the complete output and adds its usage.
func (r *recorder) finalize(ctx context.Context, message *schema.AgenticMessage) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.usage.Add(llm.UsageOf(message.ResponseMeta))
	call := r.call
	if call == nil {
		call = r.beginCallLocked("")
	}
	r.call = nil
	// The i-th block of the complete output is made of the i-th smallest chunk
	// index; streamed IDs are kept only when the counts match.
	indices := slices.Sorted(slices.Values(call.indices))
	keyed := len(indices) == len(message.ContentBlocks)
	withCalls := hasToolCalls(message)
	var final []Block
	for i, block := range message.ContentBlocks {
		var b Block
		switch {
		case block.Type == schema.ContentBlockTypeReasoning && block.Reasoning != nil && block.Reasoning.Text != "":
			b = Block{Kind: KindThinking, Text: block.Reasoning.Text}
		case block.Type == schema.ContentBlockTypeAssistantGenText && withCalls && block.AssistantGenText != nil && block.AssistantGenText.Text != "":
			b = Block{Kind: KindContent, Text: block.AssistantGenText.Text}
		case block.Type == schema.ContentBlockTypeFunctionToolCall:
			b = Block{Kind: KindToolCall, Call: r.queuedCall(block.FunctionToolCall, call.id)}
		default:
			continue
		}
		b.ID, b.ModelCallID = llm.NewModelCallID(), call.id
		if keyed {
			if position, ok := call.positions[indices[i]]; ok && r.process[position].Kind == b.Kind {
				b.ID = r.process[position].ID
				if streamed := r.process[position].Call; streamed != nil {
					b.Call.ID, b.Call.Rev = streamed.ID, streamed.Rev
				}
			}
		}
		final = append(final, b)
	}
	var ops []stream.Operation
	streamed := slices.Clone(r.process[call.start:])
	sameOrder := slices.EqualFunc(streamed, final, func(a, b Block) bool { return a.ID == b.ID })
	if !sameOrder && len(streamed) > 0 {
		slog.WarnContext(ctx, "einorun: finalized blocks differ from the streamed ones, replacing them",
			"run_id", r.runID, "streamed_blocks", len(streamed), "final_blocks", len(final))
		ids := make([]string, 0, len(streamed))
		for _, block := range streamed {
			ids = append(ids, block.ID)
		}
		ops = append(ops, stream.Operation{Kind: stream.OpRemoveBlocks, BlockIDs: ids})
	}
	for _, block := range streamed {
		if !slices.ContainsFunc(final, func(kept Block) bool { return kept.ID == block.ID }) {
			r.changes.remove(block)
		}
	}
	r.process = r.process[:call.start]
	for i, block := range final {
		block.Position = int64(call.start + i + 1)
		r.changes.block(block.ID)
		if block.Call != nil {
			r.toolPositions[block.Call.CallID] = call.start + i
			r.touchLocked(block.Call)
		}
		r.process = append(r.process, block)
		if view := r.view(block); !sameOrder || !reflect.DeepEqual(r.view(streamed[i]), view) {
			ops = append(ops, stream.Operation{Kind: stream.OpUpsertBlock, Block: view})
		}
	}
	candidate := ""
	if !withCalls {
		candidate = llm.Text(message)
	}
	if candidate != r.candidate {
		r.candidate = candidate
		ops = append(ops, stream.Operation{Kind: stream.OpClearCandidate})
		if candidate != "" {
			ops = append(ops, stream.Operation{Kind: stream.OpAppendCandidate, Text: candidate})
		}
	}
	r.pub.add(ops...)
}

// updateCall applies update to the main-agent call with the provider call ID,
// publishes it and writes it to the journal.
func (r *recorder) updateCall(ctx context.Context, providerCallID string, update func(*ToolCall)) (ToolCall, error) {
	saved, err := r.changeCall(providerCallID, update)
	if err != nil {
		return ToolCall{}, err
	}
	return saved, r.journal.SaveToolCall(ctx, saved)
}

// changeCall applies update to the main-agent call with the provider call ID
// and publishes it; the next step writes it.
func (r *recorder) changeCall(providerCallID string, update func(*ToolCall)) (ToolCall, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	position, ok := r.toolPositions[providerCallID]
	if !ok {
		return ToolCall{}, fmt.Errorf("einorun: call %q has no model output", providerCallID)
	}
	call := r.process[position].Call
	update(call)
	r.touchLocked(call)
	if call.Status.Settled() {
		delete(r.activity, call.ID)
	}
	r.pub.add(stream.Operation{Kind: stream.OpUpsertBlock, Block: r.view(r.process[position])})
	return call.Clone(), nil
}

// mainCall returns a copy of the main-agent call with the provider call ID.
func (r *recorder) mainCall(providerCallID string) (ToolCall, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	position, ok := r.toolPositions[providerCallID]
	if !ok {
		return ToolCall{}, false
	}
	return r.process[position].Call.Clone(), true
}

// childStarted records a call a sub-agent starts under the delegation call
// with the provider call ID parentCallID, and makes it the delegation's
// current activity. It returns the record and false when the delegation call
// is unknown.
func (r *recorder) childStarted(parentCallID, modelCallID, name, providerCallID, arguments string) (*ToolCall, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	position, ok := r.toolPositions[parentCallID]
	if !ok {
		return nil, false
	}
	parent := r.process[position].Call
	call := ToolCall{ID: llm.NewModelCallID(), ParentID: parent.ID, ModelCallID: cmp.Or(modelCallID, parent.ModelCallID), CallID: providerCallID,
		Arguments: arguments, Status: StatusQueued}
	r.nameCall(&call, name)
	r.childPositions[parent.ID+"/"+providerCallID] = len(r.children)
	r.children = append(r.children, call)
	if parent.Status == StatusRunning && r.activity[parent.ID] != call.Name {
		r.activity[parent.ID] = call.Name
		r.pub.add(stream.Operation{Kind: stream.OpUpsertBlock, Block: r.view(r.process[position])})
	}
	clone := call.Clone()
	return &clone, true
}

// childPosition returns the index of a sub-agent call.
func (r *recorder) childPosition(parentID, providerCallID string) (int, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	index, ok := r.childPositions[parentID+"/"+providerCallID]
	return index, ok
}

// updateChild applies update to a sub-agent call and writes it.
func (r *recorder) updateChild(ctx context.Context, parentID, providerCallID string, update func(*ToolCall)) (ToolCall, error) {
	saved, err := r.changeChild(parentID, providerCallID, update)
	if err != nil {
		return ToolCall{}, err
	}
	return saved, r.journal.SaveToolCall(ctx, saved)
}

// changeChild applies update to a sub-agent call; the next step writes it.
func (r *recorder) changeChild(parentID, providerCallID string, update func(*ToolCall)) (ToolCall, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	index, ok := r.childPositions[parentID+"/"+providerCallID]
	if !ok {
		return ToolCall{}, fmt.Errorf("einorun: sub-agent call %q is unknown", providerCallID)
	}
	call := &r.children[index]
	update(call)
	r.touchLocked(call)
	return call.Clone(), nil
}

// childCalls returns copies of the sub-agent calls.
func (r *recorder) childCalls() []ToolCall {
	r.mu.Lock()
	defer r.mu.Unlock()
	calls := make([]ToolCall, len(r.children))
	for i, c := range r.children {
		calls[i] = c.Clone()
	}
	return calls
}

// awaiting reports whether a main-agent call waits for an external result or
// for a decision.
func (r *recorder) awaiting() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.ContainsFunc(r.process, func(b Block) bool {
		return b.Call != nil && ((b.Call.Handover == HandoverAwait && !b.Call.Status.Settled()) || (paused(b.Call) && b.Call.Decision == nil))
	})
}

// currentPlan returns a copy of the plan.
func (r *recorder) currentPlan() []PlanTask {
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Clone(r.plan)
}

// setPlan replaces the plan and publishes it; it is saved with the next step.
func (r *recorder) setPlan(plan []PlanTask) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.plan = plan
	r.pub.add(stream.Operation{Kind: stream.OpSetPlan, Plan: slices.Clone(plan)})
}

// annotate writes notes to calls by record ID and returns the changed records.
func (r *recorder) annotate(notes map[string]map[string]string) []ToolCall {
	r.mu.Lock()
	defer r.mu.Unlock()
	var changed []ToolCall
	apply := func(call *ToolCall) {
		add, ok := notes[call.ID]
		if !ok || len(add) == 0 {
			return
		}
		if call.Notes == nil {
			call.Notes = map[string]string{}
		}
		maps.Copy(call.Notes, add)
		r.touchLocked(call)
		changed = append(changed, call.Clone())
	}
	for i := range r.process {
		if r.process[i].Call != nil {
			apply(r.process[i].Call)
		}
	}
	for i := range r.children {
		apply(&r.children[i])
	}
	return changed
}

// discardPending drops the candidate, the blocks of an unfinished model call
// and trailing calls that never started, when new input arrives at a safe
// point.
func (r *recorder) discardPending() {
	r.mu.Lock()
	defer r.mu.Unlock()
	var removed []string
	if r.call != nil {
		for _, block := range r.process[r.call.start:] {
			removed = append(removed, block.ID)
			r.changes.remove(block)
		}
		r.process, r.call = r.process[:r.call.start], nil
	}
	for len(r.process) > 0 {
		last := r.process[len(r.process)-1]
		if last.Call == nil || last.Call.Status != StatusQueued {
			break
		}
		delete(r.toolPositions, last.Call.CallID)
		removed = append(removed, last.ID)
		r.changes.remove(last)
		r.process = r.process[:len(r.process)-1]
	}
	if len(removed) > 0 {
		r.pub.add(stream.Operation{Kind: stream.OpRemoveBlocks, BlockIDs: removed})
	}
	if r.candidate != "" {
		r.candidate = ""
		r.pub.add(stream.Operation{Kind: stream.OpClearCandidate})
	}
}

// blocks returns copies of the blocks.
func (r *recorder) blocks() []Block {
	r.mu.Lock()
	defer r.mu.Unlock()
	return cloneBlocks(r.process)
}

// partialBlocks returns the blocks of an interrupted run; an unfinished
// candidate reply becomes a content block at the end.
func (r *recorder) partialBlocks() []Block {
	r.mu.Lock()
	defer r.mu.Unlock()
	blocks := cloneBlocks(r.process)
	if strings.TrimSpace(r.candidate) == "" {
		return blocks
	}
	modelCallID := ""
	if r.call != nil {
		modelCallID = r.call.id
	} else if len(blocks) > 0 {
		modelCallID = blocks[len(blocks)-1].ModelCallID
	}
	return append(blocks, Block{ID: llm.NewModelCallID(), Position: int64(len(blocks) + 1),
		ModelCallID: cmp.Or(modelCallID, llm.NewModelCallID()), Kind: KindContent, Text: r.candidate})
}

// cloneBlocks copies blocks and their calls.
func cloneBlocks(blocks []Block) []Block {
	cloned := slices.Clone(blocks)
	for i := range cloned {
		if call := cloned[i].Call; call != nil {
			c := call.Clone()
			cloned[i].Call = &c
		}
	}
	return cloned
}

// restore starts from saved blocks, sub-agent calls and plan and publishes
// them whole.
func (r *recorder) restore(blocks []Block, children []ToolCall, plan []PlanTask) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.process = cloneBlocks(blocks)
	for position, block := range r.process {
		if block.Call != nil {
			r.toolPositions[block.Call.CallID] = position
		}
	}
	r.children = make([]ToolCall, len(children))
	for i, c := range children {
		r.children[i] = c.Clone()
		r.childPositions[c.ParentID+"/"+c.CallID] = i
	}
	r.plan = slices.Clone(plan)
	ops := make([]stream.Operation, 0, len(r.process)+1)
	for _, block := range r.process {
		ops = append(ops, stream.Operation{Kind: stream.OpUpsertBlock, Block: r.view(block)})
	}
	if len(plan) > 0 {
		ops = append(ops, stream.Operation{Kind: stream.OpSetPlan, Plan: slices.Clone(plan)})
	}
	r.pub.add(ops...)
}

// modelUsage returns the usage of finalized main-agent outputs.
func (r *recorder) modelUsage() llm.Usage {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.usage
}

// pendingChanges returns the changes since the last save and their marks.
func (r *recorder) pendingChanges() (Changes, changeMarks) {
	r.mu.Lock()
	defer r.mu.Unlock()
	marks := r.changes.marks()
	var changes Changes
	for _, block := range r.process {
		if _, ok := marks.blocks[block.ID]; ok {
			changes.Blocks = append(changes.Blocks, cloneBlocks([]Block{block})...)
		}
		if block.Call != nil {
			if _, ok := marks.calls[block.Call.ID]; ok {
				changes.Calls = append(changes.Calls, block.Call.Clone())
			}
		}
	}
	for _, call := range r.children {
		if _, ok := marks.calls[call.ID]; ok {
			changes.Calls = append(changes.Calls, call.Clone())
		}
	}
	changes.RemovedBlocks = sortedKeys(marks.removedBlocks)
	changes.RemovedCalls = sortedKeys(marks.removedCalls)
	return changes, marks
}

// saveStep writes the changes since the last save with the step; on success
// the saved entries are cleared, on failure they stay for the next save.
func (r *recorder) saveStep(ctx context.Context, step Step) error {
	changes, marks := r.pendingChanges()
	step.Changes = changes
	if err := r.journal.SaveStep(ctx, step); err != nil {
		return err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.changes.commit(marks)
	return nil
}

// markChanged registers calls and removed blocks changed during recovery.
func (r *recorder) markChanged(removed []Block, changed []ToolCall) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, block := range removed {
		r.changes.remove(block)
	}
	for _, call := range changed {
		r.changes.call(call.ID)
	}
}

// hasToolCalls reports whether message contains tool calls.
func hasToolCalls(message *schema.AgenticMessage) bool {
	return slices.ContainsFunc(message.ContentBlocks, func(b *schema.ContentBlock) bool {
		return b != nil && b.Type == schema.ContentBlockTypeFunctionToolCall
	})
}

// toolCalls returns the tool calls of message.
func toolCalls(message *schema.AgenticMessage) []*schema.FunctionToolCall {
	var calls []*schema.FunctionToolCall
	for _, b := range message.ContentBlocks {
		if b != nil && b.Type == schema.ContentBlockTypeFunctionToolCall && b.FunctionToolCall != nil {
			calls = append(calls, b.FunctionToolCall)
		}
	}
	return calls
}
