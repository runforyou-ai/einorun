package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"

	"github.com/cloudwego/eino/schema"
	"github.com/eino-contrib/jsonschema"

	"github.com/runforyou-ai/einorun/internal/prompt"
)

var (
	// ErrRefused reports that the model refused to produce the requested output.
	ErrRefused = errors.New("llm: model refused to produce the output")
	// ErrTruncated reports that the output was cut off at the output limit.
	ErrTruncated = errors.New("llm: model output was truncated")
)

// OutputSchema describes the JSON object a model is asked to produce.
type OutputSchema struct {
	// Name names the structure; providers that constrain output with a forced
	// tool call use it as the tool name.
	Name string
	// Schema is the object schema: every property required, no additional
	// properties, nullable properties expressed as anyOf with null.
	Schema *jsonschema.Schema
}

// SchemaFor derives an OutputSchema from the JSON fields of T. Every field is
// required, including omitempty fields; fields tagged jsonschema:"nullable" may
// be null; descriptions come from jsonschema_description tags.
func SchemaFor[T any]() *OutputSchema {
	t := reflect.TypeFor[T]()
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	// Expanding the root needs a type name; anonymous structs are reflected
	// inline, which DoNotReference makes equivalent.
	reflector := &jsonschema.Reflector{Anonymous: true, DoNotReference: true, ExpandedStruct: t.Name() != ""}
	s := reflector.ReflectFromType(t)
	s.Version = ""
	normalizeSchema(s)
	name := t.Name()
	if name == "" {
		name = "output"
	}
	return &OutputSchema{Name: name, Schema: s}
}

// normalizeSchema rewrites oneOf as anyOf, which every provider's structured
// output accepts, and makes every property of every object required, as
// strict structured output demands (omitempty does not make a field
// optional).
func normalizeSchema(s *jsonschema.Schema) {
	if s == nil {
		return
	}
	if len(s.OneOf) > 0 && len(s.AnyOf) == 0 {
		s.AnyOf, s.OneOf = s.OneOf, nil
	}
	for _, child := range s.AnyOf {
		normalizeSchema(child)
	}
	normalizeSchema(s.Items)
	if s.Properties != nil {
		s.Required = s.Required[:0]
		for pair := s.Properties.Oldest(); pair != nil; pair = pair.Next() {
			s.Required = append(s.Required, pair.Key)
			normalizeSchema(pair.Value)
		}
	}
}

// GenerateRequest is a single model call without tools and with thinking off.
type GenerateRequest struct {
	Instruction     string
	Input           string
	MaxOutputTokens int
	// Output, when set, asks for a JSON object matching the schema.
	Output *OutputSchema
	// Language selects the text of the retry request GenerateObject sends.
	Language Language
}

// GenerateResult is the text, usage and stop reason of a single call.
type GenerateResult struct {
	Text  string
	Usage Usage
	Stop  StopReason
}

// Generate calls the model once with a system instruction and one user
// message. The call's context carries a new ModelCallID.
func Generate(ctx context.Context, factory ModelFactory, request GenerateRequest) (GenerateResult, error) {
	return generate(ctx, factory, request, nil)
}

// generate calls the model once; previous, when set, appends the model's last
// output and a retry request.
func generate(ctx context.Context, factory ModelFactory, request GenerateRequest, previous *retry) (GenerateResult, error) {
	m, err := factory(ctx, ModelOptions{MaxOutputTokens: request.MaxOutputTokens, DisableThinking: true, Output: request.Output})
	if err != nil {
		return GenerateResult{}, err
	}
	messages := []*schema.AgenticMessage{schema.SystemAgenticMessage(request.Instruction), schema.UserAgenticMessage(request.Input)}
	if previous != nil {
		messages = append(messages,
			&schema.AgenticMessage{Role: schema.AgenticRoleTypeAssistant, ContentBlocks: []*schema.ContentBlock{
				schema.NewContentBlock(&schema.AssistantGenText{Text: previous.text}),
			}},
			schema.UserAgenticMessage(fmt.Sprintf(prompt.For(string(request.Language)).StructuredRetry, previous.problem)))
	}
	message, err := m.Generate(WithModelCallID(ctx, NewModelCallID()), messages)
	if err != nil {
		return GenerateResult{}, err
	}
	return GenerateResult{Text: Text(message), Usage: UsageOf(message.ResponseMeta), Stop: StopReasonOf(message)}, nil
}

// retry is the output of the previous attempt and why it was not accepted.
type retry struct {
	text    string
	problem string
}

// GenerateObject asks the model for a JSON object described by T and decodes
// it. When the provider supports structured output the model is constrained;
// otherwise the instruction carries the format and the first JSON object in
// the text is decoded. Text that does not decode is retried once with the
// decode error. Every attempt gets its own ModelCallID. Validating field values is up to the caller. The usage of all
// attempts is returned, also with an error.
func GenerateObject[T any](ctx context.Context, factory ModelFactory, request GenerateRequest) (T, Usage, error) {
	var zero T
	var usage Usage
	request.Output = SchemaFor[T]()
	var previous *retry
	for {
		result, err := generate(ctx, factory, request, previous)
		usage.Add(result.Usage)
		if err != nil {
			return zero, usage, err
		}
		switch result.Stop {
		case StopRefusal:
			return zero, usage, ErrRefused
		case StopLength:
			return zero, usage, ErrTruncated
		}
		var value T
		err = DecodeObject(result.Text, &value)
		if err == nil {
			return value, usage, nil
		}
		if previous != nil || ctx.Err() != nil {
			return zero, usage, fmt.Errorf("llm: decode structured output: %w", err)
		}
		previous = &retry{text: result.Text, problem: err.Error()}
	}
}

// DecodeObject decodes the first JSON object of text, which starts at the first
// {, into target, tolerating code fences and surrounding prose.
func DecodeObject(text string, target any) error {
	start := strings.Index(text, "{")
	if start < 0 {
		return errors.New("llm: no JSON object in model output")
	}
	return json.NewDecoder(strings.NewReader(text[start:])).Decode(target)
}
