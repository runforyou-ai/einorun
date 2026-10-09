# einorun

[![CI](https://github.com/runforyou-ai/einorun/actions/workflows/ci.yml/badge.svg)](https://github.com/runforyou-ai/einorun/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/runforyou-ai/einorun.svg)](https://pkg.go.dev/github.com/runforyou-ai/einorun)

[中文](README.zh-CN.md)

einorun runs agents built on [Eino](https://github.com/cloudwego/eino) as durable, resumable runs.
Eino's ADK gives you the agent loop; einorun adds what a production host needs around it.

- **Durable input.** A run claims input from a `Feed` and picks up messages that arrive while it works.
- **Checkpoints and recovery.** The process is recorded in a `Journal` at safe points; a run resumes after a crash, or after waiting for an external result.
- **Tool calls handed over.** Tools can hand a call to another system and wait (`Await`), carry on with a receipt (`Detached`) or submit it for a human decision.
- **Structured completion and output guards.** End a run with a structured decision, check final text before it is delivered, share one correction budget.
- **Context management.** Large results are offloaded, old tool calls cleared, long histories summarized.
- **Built-in extensions:** a task list (`Planning`), delegation to sub-agents (`Subagent`) and skills (`Skills`).
- **A live stream** of the run's process, and **provider adapters** for the major model vendors.

einorun knows nothing about your business. Your instructions, tools, approvals and executors plug in through tool specs, guards and extensions. Persistence is yours too: implement two small interfaces against your own schema, and check them with `journaltest`.

Requires Go 1.27+.

> einorun is under active development. The design is in [docs/design.md](docs/design.md).

## Packages

| Package | |
|---|---|
| [`einorun`](https://pkg.go.dev/github.com/runforyou-ai/einorun) | Run records and the `Feed` and `Journal` contracts |
| [`llm`](https://pkg.go.dev/github.com/runforyou-ai/einorun/llm) | Model contract, single calls and structured output |
| [`provider`](https://pkg.go.dev/github.com/runforyou-ai/einorun/provider) | Chat models, embeddings, rerank, model discovery and probes for major vendors |
| [`stream`](https://pkg.go.dev/github.com/runforyou-ai/einorun/stream) | Live display stream of a run |
| [`memory`](https://pkg.go.dev/github.com/runforyou-ai/einorun/memory) | Long-term memory: recall extension and extraction |
| [`toolname`](https://pkg.go.dev/github.com/runforyou-ai/einorun/toolname) | Valid, deterministic model-visible names for external tools |
| [`tools/web`](https://pkg.go.dev/github.com/runforyou-ai/einorun/tools/web) | Web search and page reading tools over your own services |
| [`inmem`](https://pkg.go.dev/github.com/runforyou-ai/einorun/inmem) | In-memory `Feed` and `Journal` |
| [`journaltest`](https://pkg.go.dev/github.com/runforyou-ai/einorun/journaltest) | Contract suites for your `Feed` and `Journal` |

## Example

[`examples/chat`](examples/chat) is a command-line agent in about 150 lines: an in-memory conversation, a provider model, a tool, the task list and sub-agents, with the reply streamed to the terminal.

```sh
EINORUN_BRAND=openai EINORUN_BASE_URL=https://api.openai.com/v1 \
  EINORUN_MODEL=gpt-4.1 EINORUN_API_KEY=... go run ./examples/chat
```

## Structured output without a run

```go
type Title struct {
	Title string `json:"title" jsonschema_description:"A short title"`
}

title, usage, err := llm.GenerateObject[Title](ctx, factory, llm.GenerateRequest{
	Instruction: "Give the conversation a title.",
	Input:       transcript,
	Language:    llm.English,
})
```

## License

MIT
