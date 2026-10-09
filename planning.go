package einorun

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"

	"github.com/cloudwego/eino/adk"
	"github.com/cloudwego/eino/adk/filesystem"
	"github.com/cloudwego/eino/adk/middlewares/plantask"
	"github.com/cloudwego/eino/schema"

	"github.com/runforyou-ai/einorun/stream"
)

const (
	// planningExtensionName is the name of the Planning extension.
	planningExtensionName = "einorun.planning"
	// planDir is the directory of task files in the run's memory.
	planDir = "/plan"
)

// PlanTools are the names of the task list tools, in registration order.
var PlanTools = []string{plantask.TaskCreateToolName, plantask.TaskGetToolName, plantask.TaskUpdateToolName, plantask.TaskListToolName}

// Planning returns the extension that gives the main agent a task list. The
// list is kept in the run's checkpoint and published as the run's plan
// (Step.Plan, Result.Plan and the stream). Tasks the framework removes once
// all of them are completed stay in the plan as completed.
func Planning() Extension { return planningExtension{} }

// planningExtension creates the Planning instance of a run.
type planningExtension struct{}

// Name returns the extension name.
func (planningExtension) Name() string { return planningExtensionName }

// Instance returns the run's instance.
func (planningExtension) Instance(context.Context, RunScope) (Extension, error) {
	return &planningInstance{files: map[string]string{}}, nil
}

// planningInstance is the Planning extension of one run: the backend of the
// task list tools.
type planningInstance struct {
	middleware adk.TypedChatModelAgentMiddleware[*schema.AgenticMessage]
	publish    func([]PlanTask)

	mu    sync.Mutex
	files map[string]string
	shown []PlanTask // the plan, by task number
}

// Name returns the extension name.
func (p *planningInstance) Name() string { return planningExtensionName }

// bind declares the task list tools and creates their middleware.
func (p *planningInstance) bind(ctx context.Context, e *execution) error {
	for _, name := range PlanTools {
		if err := e.declareTool(name, ToolSpec{Replayable: true, Retain: RetainKeep}, nil); err != nil {
			return err
		}
	}
	middleware, err := plantask.NewTyped[*schema.AgenticMessage](ctx, &plantask.Config{Backend: p, BaseDir: planDir})
	if err != nil {
		return fmt.Errorf("einorun: create the task list middleware: %w", err)
	}
	p.middleware, p.publish = middleware, e.recorder.setPlan
	return nil
}

// ModelMiddlewares adds the task list tools to the main agent.
func (p *planningInstance) ModelMiddlewares(scope AgentScope) []adk.TypedChatModelAgentMiddleware[*schema.AgenticMessage] {
	if !scope.Main || p.middleware == nil {
		return nil
	}
	return []adk.TypedChatModelAgentMiddleware[*schema.AgenticMessage]{p.middleware}
}

// planState is the checkpoint state of the task list.
type planState struct {
	Files map[string]string `json:"files,omitempty"`
	Shown []PlanTask        `json:"shown,omitempty"`
}

// Save returns the task files and the plan.
func (p *planningInstance) Save() (json.RawMessage, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return json.Marshal(planState{Files: p.files, Shown: p.shown})
}

// Restore restores the task files and the plan.
func (p *planningInstance) Restore(data json.RawMessage) error {
	var s planState
	if err := json.Unmarshal(data, &s); err != nil {
		return err
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.files = maps.Clone(s.Files)
	if p.files == nil {
		p.files = map[string]string{}
	}
	p.shown = s.Shown
	return nil
}

// LsInfo lists the files of a directory.
func (p *planningInstance) LsInfo(_ context.Context, req *plantask.LsInfoRequest) ([]plantask.FileInfo, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	dir := filepath.Clean(req.Path)
	var infos []plantask.FileInfo
	for path, content := range p.files {
		if filepath.Dir(path) == dir {
			infos = append(infos, plantask.FileInfo{Path: path, Size: int64(len(content))})
		}
	}
	slices.SortFunc(infos, func(a, b plantask.FileInfo) int { return strings.Compare(a.Path, b.Path) })
	return infos, nil
}

// Read returns a file.
func (p *planningInstance) Read(_ context.Context, req *plantask.ReadRequest) (*filesystem.FileContent, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	content, ok := p.files[filepath.Clean(req.FilePath)]
	if !ok {
		return nil, fmt.Errorf("%s: %w", req.FilePath, os.ErrNotExist)
	}
	return &filesystem.FileContent{Content: content}, nil
}

// Write writes a file and updates the plan when it is a task.
func (p *planningInstance) Write(_ context.Context, req *plantask.WriteRequest) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	path := filepath.Clean(req.FilePath)
	p.files[path] = req.Content
	task, ok := planTask(path, req.Content)
	if !ok {
		return nil
	}
	if i := slices.IndexFunc(p.shown, func(t PlanTask) bool { return t.ID == task.ID }); i >= 0 {
		p.shown[i] = task
	} else {
		p.shown = append(p.shown, task)
		slices.SortFunc(p.shown, func(a, b PlanTask) int {
			x, _ := strconv.Atoi(a.ID)
			y, _ := strconv.Atoi(b.ID)
			return x - y
		})
	}
	p.publishLocked()
	return nil
}

// Delete deletes a file. A deleted task leaves the plan, unless every task
// left is completed: the framework then clears the list and the plan keeps
// the tasks as completed.
func (p *planningInstance) Delete(_ context.Context, req *plantask.DeleteRequest) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	path := filepath.Clean(req.FilePath)
	content, ok := p.files[path]
	if !ok {
		return nil
	}
	delete(p.files, path)
	task, ok := planTask(path, content)
	if !ok {
		return nil
	}
	cleanup := task.Status == stream.PlanCompleted
	for other, otherContent := range p.files {
		if remaining, ok := planTask(other, otherContent); ok && remaining.Status != stream.PlanCompleted {
			cleanup = false
		}
	}
	if cleanup {
		return nil
	}
	p.shown = slices.DeleteFunc(p.shown, func(t PlanTask) bool { return t.ID == task.ID })
	p.publishLocked()
	return nil
}

// publishLocked hands a copy of the plan to the recorder.
func (p *planningInstance) publishLocked() {
	if p.publish != nil {
		p.publish(slices.Clone(p.shown))
	}
}

// planTask parses a task file; false when path is not a task file or the
// content cannot be parsed.
func planTask(path, content string) (PlanTask, bool) {
	id, found := strings.CutSuffix(filepath.Base(path), ".json")
	if !found {
		return PlanTask{}, false
	}
	if _, err := strconv.Atoi(id); err != nil {
		return PlanTask{}, false
	}
	var task PlanTask
	if err := json.Unmarshal([]byte(content), &task); err != nil {
		return PlanTask{}, false
	}
	return task, true
}
