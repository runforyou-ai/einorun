// Package probe checks that a model vendor endpoint is reachable and accepts
// the credentials, using a read-only request of each vendor.
package probe

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/runforyou-ai/einorun/provider/apierr"
	"github.com/runforyou-ai/einorun/provider/internal/httpx"
	"github.com/runforyou-ai/einorun/provider/vendor"
)

// Endpoint is a vendor endpoint to check.
type Endpoint struct {
	Brand   vendor.Brand
	BaseURL string
	APIKey  string
}

// DefaultTimeout bounds a probe made with the default client.
const DefaultTimeout = 30 * time.Second

// Run sends the brand's read-only request and validates the minimal shape of
// the response. Failures are classified with apierr. A nil client means one
// that never follows redirects and times out after DefaultTimeout.
func Run(ctx context.Context, client httpx.Doer, endpoint Endpoint) error {
	if client == nil {
		client = httpx.NewClient(DefaultTimeout)
	}
	request := httpx.Request{APIKey: endpoint.APIKey}
	var validate func([]byte) error
	var err error
	switch endpoint.Brand {
	case vendor.Ollama:
		request.URL, err = OllamaURL(endpoint.BaseURL, "api/tags")
		validate = arrayField("models")
	case vendor.Alibaba:
		request.URL, err = alibabaModelsURL(endpoint.BaseURL)
		validate = alibabaModels
	case vendor.Anthropic:
		request.URL, err = vendor.AppendPath(endpoint.BaseURL, "v1/models")
		request.APIKey = ""
		request.Header = http.Header{"X-Api-Key": {endpoint.APIKey}, "Anthropic-Version": {"2023-06-01"}}
		validate = arrayField("data")
	case vendor.Google:
		request.URL, err = vendor.AppendPath(endpoint.BaseURL, "v1beta/models")
		request.APIKey = ""
		request.Header = http.Header{"X-Goog-Api-Key": {endpoint.APIKey}}
		validate = arrayField("models")
	case vendor.OpenRouter:
		// The model list needs no credentials; the key endpoint checks them.
		request.URL, err = vendor.AppendPath(endpoint.BaseURL, "key")
		validate = objectField("data")
	case vendor.TypeSafe:
		request.URL, err = vendor.AppendPath(endpoint.BaseURL, "models")
		validate = arrayField("models")
	default:
		if _, known := vendor.Of(endpoint.Brand); !known {
			return apierr.New(apierr.StageCapability, apierr.InvalidConfig, fmt.Errorf("unknown brand %q", endpoint.Brand))
		}
		request.URL, err = vendor.AppendPath(endpoint.BaseURL, "models")
		validate = arrayField("data")
	}
	if err != nil {
		return apierr.InvalidConfigError(err)
	}
	body, err := httpx.Do(ctx, client, request)
	if err != nil {
		return err
	}
	if err := validate(body); err != nil {
		return apierr.New(apierr.StageCapability, apierr.Protocol, err)
	}
	return nil
}

// OllamaURL returns the URL of an Ollama native API path for a base URL that
// may end in the OpenAI-compatible /v1.
func OllamaURL(baseURL, path string) (string, error) {
	parsed, err := vendor.ParseURL(baseURL)
	if err != nil {
		return "", err
	}
	parsed.Path = strings.TrimSuffix(strings.TrimSuffix(parsed.Path, "/"), "/v1")
	parsed.RawPath = ""
	return vendor.AppendPath(parsed.String(), path)
}

// alibabaModelsURL returns the DashScope model list URL for a compatible or
// native base URL.
func alibabaModelsURL(baseURL string) (string, error) {
	parsed, err := vendor.ParseURL(baseURL)
	if err != nil {
		return "", err
	}
	path := strings.TrimSuffix(parsed.Path, "/")
	switch {
	case strings.HasSuffix(path, "/compatible-mode/v1"):
		path = strings.TrimSuffix(path, "/compatible-mode/v1") + "/api/v1/models"
	case strings.HasSuffix(path, "/api/v1"):
		path += "/models"
	default:
		path += "/api/v1/models"
	}
	parsed.Path, parsed.RawPath = path, ""
	return parsed.String(), nil
}

// arrayField validates that the response has an array under field.
func arrayField(field string) func([]byte) error {
	return func(body []byte) error {
		var payload map[string]json.RawMessage
		if err := json.Unmarshal(body, &payload); err != nil {
			return err
		}
		if value := payload[field]; len(value) == 0 || value[0] != '[' {
			return fmt.Errorf("response has no %s array", field)
		}
		return nil
	}
}

// objectField validates that the response has an object under field.
func objectField(field string) func([]byte) error {
	return func(body []byte) error {
		var payload map[string]json.RawMessage
		if err := json.Unmarshal(body, &payload); err != nil {
			return err
		}
		if value := payload[field]; len(value) == 0 || value[0] != '{' {
			return fmt.Errorf("response has no %s object", field)
		}
		return nil
	}
}

// alibabaModels validates DashScope's model list.
func alibabaModels(body []byte) error {
	var payload struct {
		Success bool `json:"success"`
		Output  struct {
			Models json.RawMessage `json:"models"`
		} `json:"output"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return err
	}
	if !payload.Success || len(payload.Output.Models) == 0 || payload.Output.Models[0] != '[' {
		return errors.New("response has no successful models array")
	}
	return nil
}
