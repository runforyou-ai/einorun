package prompt

// Runtime is the model-facing text the runtime writes in one language.
type Runtime struct {
	// Cancelled is the result of a call that never got one because other
	// input arrived first. %[1]s is the tool name, %[2]s the call identifier.
	Cancelled string
	// InterruptedReplayable is the result of an interrupted call that may be
	// repeated.
	InterruptedReplayable string
	// Interrupted is the result of an interrupted call without side effects.
	Interrupted string
	// NeedsReview is the result of an interrupted call whose side effects are
	// unknown.
	NeedsReview string
	// CannotAwait fails a call that tried to wait for an external result where
	// the run cannot suspend.
	CannotAwait string
	// CannotDetach fails a call a sub-agent tried to hand over.
	CannotDetach string
	// CannotComplete fails a completion result from a tool that is not a
	// completion tool, or from a sub-agent.
	CannotComplete string
	// AwaitingResult is the placeholder result of a call waiting for an
	// external result.
	AwaitingResult string
	// BatchTooMany rejects a batch with several completion tools. %s lists
	// the completion tools.
	BatchTooMany string
	// BatchMixed rejects a batch mixing a completion tool with other tools.
	// %s lists the completion tools.
	BatchMixed string
	// FinalNotice tells the model the iteration budget is spent.
	FinalNotice string
	// FinalNoticeCompletion does the same when completion tools remain. %s
	// lists them.
	FinalNoticeCompletion string
	// MediaUnavailable replaces media the model cannot view. %s is the media
	// type.
	MediaUnavailable string
	// ListSeparator joins tool names.
	ListSeparator string
}

var runtimes = map[string]*Runtime{
	"zh": {
		Cancelled:             "工具调用 %[1]s（ID 为 %[2]s）已被取消——在其完成之前收到了另一条消息。",
		InterruptedReplayable: "执行被中断，没有返回结果，需要时可以重新调用。",
		Interrupted:           "执行被中断，结果未知。",
		NeedsReview:           "执行被中断，外部操作的实际结果未知，已交由人工核对，不要重复执行。",
		CannotAwait:           "这个调用无法等待外部结果，没有执行。",
		CannotDetach:          "这个调用无法交给外部继续执行，没有执行。",
		CannotComplete:        "这个工具不能结束本次运行。",
		AwaitingResult:        `{"status":"awaiting_external_result"}`,
		BatchTooMany:          "%s 一次只能调用其中一个。",
		BatchMixed:            "%s 必须单独调用，不能与其他工具同时调用。",
		FinalNotice:           "工具调用次数已达本轮上限，请基于已获得的信息给出最终回答。",
		FinalNoticeCompletion: "工具调用次数已达本轮上限，不能再调用其他工具。请直接给出最终回答，或单独调用 %s 结束。",
		MediaUnavailable:      "[%s：当前模型无法查看此内容]",
		ListSeparator:         "、",
	},
	"en": {
		Cancelled:             "Tool call %[1]s (ID %[2]s) was cancelled: another message arrived before it finished.",
		InterruptedReplayable: "The call was interrupted and returned no result; call it again if needed.",
		Interrupted:           "The call was interrupted; its outcome is unknown.",
		NeedsReview:           "The call was interrupted and the outcome of its external effects is unknown. It has been passed on for human review; do not repeat it.",
		CannotAwait:           "This call cannot wait for an external result and was not executed.",
		CannotDetach:          "This call cannot be handed over and was not executed.",
		CannotComplete:        "This tool cannot end the run.",
		AwaitingResult:        `{"status":"awaiting_external_result"}`,
		BatchTooMany:          "Call only one of %s at a time.",
		BatchMixed:            "%s must be called on its own, not together with other tools.",
		FinalNotice:           "The tool call limit for this turn is reached. Give your final answer based on what you have found.",
		FinalNoticeCompletion: "The tool call limit for this turn is reached; no other tools can be called. Give your final answer, or call %s on its own to finish.",
		MediaUnavailable:      "[%s: the current model cannot view this content]",
		ListSeparator:         ", ",
	},
}

// RuntimeFor returns the runtime text for language, falling back to English.
func RuntimeFor(language string) *Runtime {
	if c, ok := runtimes[language]; ok {
		return c
	}
	return runtimes["en"]
}
