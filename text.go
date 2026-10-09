package einorun

import (
	"reflect"

	"github.com/runforyou-ai/einorun/llm"
)

// Text is the model-facing text the runtime writes. Config.Text overrides
// the default text of the run's language field by field: empty fields keep
// the default. Format verbs are documented per field and must be kept.
type Text struct {
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
	// MediaChanged replaces media whose content changed since the tool read
	// it. %s is the media key.
	MediaChanged string
	// MediaOverBudget replaces media beyond the run's media budget. %s is the
	// media key.
	MediaOverBudget string
	// ListSeparator joins tool names.
	ListSeparator string
	// SummaryPreamble introduces a summary of earlier context.
	SummaryPreamble string
	// OffloadRead describes the tool that reads offloaded results.
	OffloadRead string
	// SubagentDescription describes the general sub-agent.
	SubagentDescription string
	// SubagentTool describes the delegation tool.
	SubagentTool string
	// SubagentGuide is added to the main agent's instruction when it can
	// delegate. %s is the delegation tool name.
	SubagentGuide string
	// SubagentTypes introduces the list of sub-agents.
	SubagentTypes string
	// SubagentNoResult is the result of a delegation whose sub-agent gave no
	// answer.
	SubagentNoResult string
	// SkillForkResult wraps the answer of a skill run by a sub-agent. %[1]s
	// is the skill name, %[2]s the answer.
	SkillForkResult string
}

// texts are the default text by language.
var texts = map[llm.Language]Text{
	llm.Chinese: {
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
		MediaChanged:          "[%s：内容在读取后已变化，没有附上；需要时重新读取]",
		MediaOverBudget:       "[%s：本次运行可附给模型的媒体已达上限，没有附上]",
		ListSeparator:         "、",
		SummaryPreamble:       "【较早对话摘要】此前的对话已压缩为以下摘要，摘要之后的消息保持原样。",
		OffloadRead:           "读取本次运行中因结果过大而转存的工具输出，file_path 使用转存提示中给出的路径。",
		SubagentDescription:   "通用子 Agent：使用与你相同的工具（不含委派与任务清单工具），适合调研资料、查阅大量内容、独立完成一段工作等可以单独交付结果的任务。",
		SubagentTool:          "把一个可以独立完成的子任务交给子 Agent。子 Agent 看不到本次对话，prompt 中写清目标、已知信息和期望的结果形式；description 用一句短语概括任务。子 Agent 的最终回答作为结果返回给你，不会展示给用户。可以在一次回复中发起多个调用并行执行。",
		SubagentGuide:         "可以用 %s 工具把独立的子任务交给子 Agent，只取回结论，适合需要大量查阅或可以并行的工作；简单任务直接自己完成。",
		SubagentTypes:         "可用的子 Agent 类型：",
		SubagentNoResult:      "子 Agent 没有给出结果。",
		SkillForkResult:       "技能 %[1]s 已由子 Agent 执行完成，结果：\n%[2]s",
	},
	llm.English: {
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
		MediaChanged:          "[%s: the content changed after it was read and was not attached; read it again if needed]",
		MediaOverBudget:       "[%s: the media limit of this run is reached; not attached]",
		ListSeparator:         ", ",
		SummaryPreamble:       "[Summary of earlier conversation] The conversation so far is summarized below; the messages after the summary are unchanged.",
		OffloadRead:           "Read a tool output of this run that was offloaded because it was too large; use the file_path given in the offload note.",
		SubagentDescription:   "General sub-agent: uses the same tools as you (without delegation and the task list), for tasks that can be delivered on their own, such as research, reading a lot of material or completing a separate piece of work.",
		SubagentTool:          "Hand a self-contained task to a sub-agent. The sub-agent cannot see this conversation: state the goal, what is known and the expected form of the result in prompt; summarize the task in a short phrase in description. The sub-agent's final answer is returned to you and is not shown to the user. Several calls in one response run in parallel.",
		SubagentGuide:         "Use the %s tool to hand self-contained tasks to sub-agents and get back only their conclusions, for work that needs a lot of reading or can run in parallel; do simple tasks yourself.",
		SubagentTypes:         "Available sub-agent types:",
		SubagentNoResult:      "The sub-agent gave no result.",
		SkillForkResult:       "Skill %[1]s was run by a sub-agent. Result:\n%[2]s",
	},
}

// DefaultText returns the default text for language, English for languages
// without their own.
func DefaultText(language llm.Language) Text {
	if t, ok := texts[language]; ok {
		return t
	}
	return texts[llm.English]
}

// withOverrides returns t with the non-empty fields of o.
func (t Text) withOverrides(o Text) Text {
	base, over := reflect.ValueOf(&t).Elem(), reflect.ValueOf(o)
	for i := range base.NumField() {
		if s := over.Field(i).String(); s != "" {
			base.Field(i).SetString(s)
		}
	}
	return t
}
