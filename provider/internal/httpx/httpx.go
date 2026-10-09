// Package httpx sends JSON requests to vendor APIs and classifies failures with
// apierr.
package httpx

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/runforyou-ai/einorun/provider/apierr"
)

// maxBodyExcerpt is the length of the response excerpt kept in status errors.
const maxBodyExcerpt = 512

// Doer sends HTTP requests.
type Doer interface {
	Do(*http.Request) (*http.Response, error)
}

// Request is one call.
type Request struct {
	Method   string
	URL      string
	APIKey   string // sent as a bearer token when set
	Header   http.Header
	JSON     any // request body
	MaxBytes int64
}

// Call sends request and decodes a 2xx JSON response into output (when not
// nil). Failures are classified with apierr; context cancellation is returned
// unchanged.
func Call(ctx context.Context, client Doer, request Request, output any) error {
	body, err := Do(ctx, client, request)
	if err != nil {
		return err
	}
	if output == nil {
		return nil
	}
	if err := json.Unmarshal(body, output); err != nil {
		return apierr.New(apierr.StageCapability, apierr.Protocol, err)
	}
	return nil
}

// Do sends request and returns the body of a 2xx response.
func Do(ctx context.Context, client Doer, request Request) ([]byte, error) {
	var reader io.Reader
	if request.JSON != nil {
		var encoded bytes.Buffer
		encoder := json.NewEncoder(&encoded)
		encoder.SetEscapeHTML(false)
		if err := encoder.Encode(request.JSON); err != nil {
			return nil, err
		}
		reader = &encoded
	}
	method := request.Method
	if method == "" {
		method = http.MethodGet
	}
	req, err := http.NewRequestWithContext(ctx, method, request.URL, reader)
	if err != nil {
		return nil, apierr.InvalidConfigError(err)
	}
	for key, values := range request.Header {
		req.Header[key] = append([]string(nil), values...)
	}
	req.Header.Set("Accept", "application/json")
	if request.JSON != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if key := strings.TrimSpace(request.APIKey); key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	response, err := client.Do(req)
	if err != nil {
		return nil, transportError(ctx, err)
	}
	defer func() { _ = response.Body.Close() }()
	limit := request.MaxBytes
	if limit <= 0 {
		limit = 16 << 20
	}
	body, err := io.ReadAll(io.LimitReader(response.Body, limit+1))
	if err != nil {
		return nil, transportError(ctx, err)
	}
	if int64(len(body)) > limit {
		return nil, apierr.New(apierr.StageCapability, apierr.Protocol, errors.New("response too large"))
	}
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		return nil, apierr.FromStatus(response.StatusCode, excerpt(body))
	}
	return body, nil
}

// transportError returns cancellation by the caller unchanged and classifies
// everything else, including the caller's deadline, with apierr.
func transportError(ctx context.Context, err error) error {
	if errors.Is(ctx.Err(), context.Canceled) {
		return ctx.Err()
	}
	if errors.Is(ctx.Err(), context.DeadlineExceeded) {
		return apierr.New(apierr.StageConnect, apierr.Timeout, context.DeadlineExceeded)
	}
	return apierr.FromTransport(err)
}

// NewClient returns an HTTP client that never follows redirects, so that
// credentials are only ever sent to the configured endpoint, with the given
// timeout per request.
func NewClient(timeout time.Duration) *http.Client {
	return &http.Client{
		Timeout:       timeout,
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	}
}

// excerpt returns the start of body as valid UTF-8 text.
func excerpt(body []byte) string {
	text := strings.TrimSpace(string(body))
	if len(text) <= maxBodyExcerpt {
		return text
	}
	text = text[:maxBodyExcerpt]
	for !utf8.ValidString(text) {
		text = text[:len(text)-1]
	}
	return text + "…"
}
