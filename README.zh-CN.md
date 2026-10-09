# einorun

[English](README.md)

einorun 把基于 [Eino](https://github.com/cloudwego/eino) 构建的 Agent 作为可持久、可恢复的运行来执行。
Eino 的 ADK 提供 Agent 循环，einorun 补上生产环境宿主需要的其余部分。

- **持久输入**：运行从 `Feed` 认领输入，执行期间到达的新消息也会被接住。
- **检查点与恢复**：过程在安全点记入 `Journal`；崩溃之后、或等待外部结果之后，运行都能接着执行。
- **交出工具调用**：工具可以把调用交给其他系统并等待（`Await`）、凭回执继续（`Detached`）、提交人工裁决，或暂停运行等待确认（批准、拒绝或改参数），确认后在同一轮内执行。
- **结构化完成与输出守卫**：以结构化决定结束运行，在正文交付前检查，纠正次数统一计算。
- **上下文治理**：大结果转存、较早的工具调用清理、长历史摘要。
- **内置扩展**：任务清单（`Planning`）、委派子 Agent（`Subagent`）与技能（`Skills`）。
- **实时运行流**，以及主要模型厂商的**供应商适配**。

einorun 不了解你的业务。指令、工具、审批与执行器通过工具规格、守卫和扩展接入。持久化也由你掌握：按自己的表结构实现两个小接口，再用 `journaltest` 校验。

需要 Go 1.27+。

> einorun 仍在开发中，设计见 [docs/design.md](docs/design.md)。

## 示例

[`examples/chat`](examples/chat) 是约 150 行的命令行 Agent：内存对话、供应商模型、一个工具、任务清单与子 Agent，回复实时输出到终端。

```sh
EINORUN_BRAND=openai EINORUN_BASE_URL=https://api.openai.com/v1 \
  EINORUN_MODEL=gpt-4.1 EINORUN_API_KEY=... go run ./examples/chat
```

## 许可

MIT
