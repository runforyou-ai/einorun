// Command chat is a minimal command-line agent built with einorun: an
// in-memory conversation, a model from the provider package, a tool, the task
// list and sub-agents, with the reply streamed to the terminal.
//
//	EINORUN_BRAND=openai EINORUN_BASE_URL=https://api.openai.com/v1 \
//	EINORUN_MODEL=gpt-4.1 EINORUN_API_KEY=... go run ./examples/chat
//
// EINORUN_BASE_URL is the vendor endpoint; a local OpenAI-compatible server
// works with EINORUN_BRAND=openai_compatible. EINORUN_MAX_OUTPUT_TOKENS bounds
// each answer (4096 by default) and EINORUN_LANGUAGE (en or zh) selects the
// language of the text written for the model.
//
// Each line is answered by a new run on the same conversation, with the
// default in-memory journal: only the messages carry over to the next run,
// not the tool calls or the task list. Lines typed while a run works are read
// after it ends. A real host buffers the stream instead of writing to the
// terminal in the callback, which must not block.
package main

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"strconv"
	"time"

	"github.com/cloudwego/eino/adk"
	"github.com/cloudwego/eino/components/tool"
	toolutils "github.com/cloudwego/eino/components/tool/utils"

	"github.com/runforyou-ai/einorun"
	"github.com/runforyou-ai/einorun/inmem"
	"github.com/runforyou-ai/einorun/llm"
	"github.com/runforyou-ai/einorun/provider"
	"github.com/runforyou-ai/einorun/provider/vendor"
	"github.com/runforyou-ai/einorun/stream"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	err := chat(ctx)
	stop()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// chat reads lines from the terminal and answers each with a run. Each line
// starts a new run on the same conversation: only the messages carry over
// from one run to the next.
func chat(ctx context.Context) error {
	language := llm.Language(env("EINORUN_LANGUAGE", "en"))
	// Eino's own tool text, such as the task list's, follows the process-wide
	// ADK language.
	adkLanguage := adk.LanguageEnglish
	if language == llm.Chinese {
		adkLanguage = adk.LanguageChinese
	}
	if err := adk.SetLanguage(adkLanguage); err != nil {
		return err
	}
	maxOutput, err := strconv.Atoi(env("EINORUN_MAX_OUTPUT_TOKENS", "4096"))
	if err != nil || maxOutput <= 0 {
		return errors.New("EINORUN_MAX_OUTPUT_TOKENS must be a positive number")
	}
	config := provider.ChatConfig{
		Brand:    vendor.Brand(env("EINORUN_BRAND", "openai")),
		BaseURL:  os.Getenv("EINORUN_BASE_URL"),
		APIKey:   os.Getenv("EINORUN_API_KEY"),
		Model:    os.Getenv("EINORUN_MODEL"),
		Language: language,
	}
	preset, known := vendor.Of(config.Brand)
	switch {
	case !known:
		return fmt.Errorf("EINORUN_BRAND: unknown brand %q", config.Brand)
	case config.BaseURL == "" || config.Model == "":
		return errors.New("set EINORUN_BASE_URL and EINORUN_MODEL")
	case config.APIKey == "" && !preset.CredentialsOptional:
		return errors.New("set EINORUN_API_KEY")
	}
	factory := provider.Factory(config)
	clock, err := clockTool()
	if err != nil {
		return err
	}
	runtime := einorun.New(einorun.Config{Language: language})
	feed := inmem.NewFeed()
	var readErr error
	lines := readLines(&readErr)
	for turn := 1; ; turn++ {
		fmt.Print("> ")
		var line string
		select {
		case <-ctx.Done():
			fmt.Println()
			return nil
		case read, ok := <-lines:
			if !ok {
				return readErr
			}
			line = read
		}
		if line == "" {
			continue
		}
		feed.Append(einorun.Message{ID: "u" + strconv.Itoa(turn), Revision: "1", Role: einorun.RoleUser, Content: line})
		printer := &printer{}
		result, err := runtime.Run(ctx, einorun.Request{
			RunID:       "turn-" + strconv.Itoa(turn),
			Instruction: "You are a helpful assistant. Use the task list for work with several steps.",
			Model:       einorun.Model{New: factory, MaxOutputTokens: maxOutput},
			Feed:        feed,
			Tools:       []einorun.ToolSpec{{Tool: clock, Replayable: true}},
			Extensions: []einorun.Extension{einorun.Planning(), einorun.Subagent(einorun.SubagentSpec{
				Instruction: "Complete the delegated task and report the result briefly.",
			})},
			Stream: printer.apply,
		})
		if errors.Is(err, context.Canceled) {
			fmt.Println()
			return nil
		}
		// A failed turn leaves the input unanswered; the next line is
		// answered together with it.
		if err != nil {
			printer.endLine()
			fmt.Fprintln(os.Stderr, "error:", err)
			continue
		}
		printer.endLine()
		fmt.Printf("[%d tokens]\n", result.Usage.Total)
		// The reply joins the conversation and the input it answers is consumed.
		feed.Append(einorun.Message{ID: "a" + strconv.Itoa(turn), Revision: "1", Role: einorun.RoleAssistant, Content: result.Text})
		feed.Consume(result.EndSeq)
	}
}

// readLines reads the terminal in the background, so that waiting for input
// can be interrupted. When the channel closes, *err holds the read error, nil
// at the end of the input.
func readLines(err *error) <-chan string {
	lines := make(chan string)
	go func() {
		defer close(lines)
		input := bufio.NewScanner(os.Stdin)
		input.Buffer(make([]byte, 64<<10), 1<<20)
		for input.Scan() {
			lines <- input.Text()
		}
		*err = input.Err()
	}()
	return lines
}

// printer shows the run's process from its stream: tool calls when they
// start, task list changes and the reply as it is written. Text written before
// tool calls stays on screen as the model's explanation.
type printer struct {
	started map[string]bool
	midLine bool
}

// apply prints the operations of a delta in order.
func (p *printer) apply(d stream.Delta) {
	if p.started == nil {
		p.started = map[string]bool{}
	}
	for _, op := range d.Operations {
		switch op.Kind {
		case stream.OpAppendCandidate:
			fmt.Print(op.Text)
			p.midLine = true
		case stream.OpClearCandidate:
			p.endLine()
		case stream.OpUpsertBlock:
			// A call is shown once it leaves the queue, when its arguments are
			// complete; merged deltas may skip its running state.
			call := op.Block.Call
			if call == nil || call.Status == "" || call.Status == string(einorun.StatusQueued) || p.started[op.Block.ID] {
				continue
			}
			p.started[op.Block.ID] = true
			p.endLine()
			fmt.Printf("· %s %s\n", call.Name, call.Description)
		case stream.OpSetPlan:
			p.endLine()
			for _, task := range op.Plan {
				fmt.Printf("  [%s] %s\n", task.Status, task.Subject)
			}
		}
	}
}

// endLine ends a line of reply text.
func (p *printer) endLine() {
	if p.midLine {
		fmt.Println()
		p.midLine = false
	}
}

// clockInput is the input of the clock tool.
type clockInput struct {
	Zone string `json:"zone" jsonschema:"required" jsonschema_description:"IANA time zone, such as Asia/Shanghai"`
}

// clockTool returns a tool that tells the time in a time zone.
func clockTool() (tool.InvokableTool, error) {
	return toolutils.InferTool("clock", "Tell the current time in a time zone.",
		func(_ context.Context, in clockInput) (string, error) {
			location, err := time.LoadLocation(in.Zone)
			if err != nil {
				return "", err
			}
			return time.Now().In(location).Format(time.RFC1123), nil
		})
}

// env returns an environment variable or a default.
func env(name, fallback string) string {
	if value := os.Getenv(name); value != "" {
		return value
	}
	return fallback
}
