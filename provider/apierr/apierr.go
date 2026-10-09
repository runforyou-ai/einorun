// Package apierr classifies failures of calls to model vendor APIs in a way
// that does not depend on the vendor.
package apierr

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
)

// Stage is where a call failed.
type Stage string

// Stages of a call.
const (
	StageConnect      Stage = "connect"
	StageAuthenticate Stage = "authenticate"
	StageAuthorize    Stage = "authorize"
	StageCapability   Stage = "capability"
)

// Kind is what went wrong.
type Kind string

// Kinds of failure.
const (
	InvalidConfig Kind = "invalid_config"
	Unauthorized  Kind = "unauthorized"
	Forbidden     Kind = "forbidden"
	NotFound      Kind = "not_found"
	RateLimited   Kind = "rate_limited"
	Timeout       Kind = "timeout"
	Network       Kind = "network"
	TLS           Kind = "tls"
	Protocol      Kind = "protocol"
	Unavailable   Kind = "unavailable"
)

// Error is a classified failure.
type Error struct {
	Stage Stage
	Kind  Kind
	// Status is the HTTP status of the response, or 0 when there was none.
	Status int
	Err    error
}

// Error describes the failure. It never includes credentials.
func (e *Error) Error() string {
	if e.Err == nil {
		return fmt.Sprintf("model API failed at %s: %s", e.Stage, e.Kind)
	}
	return fmt.Sprintf("model API failed at %s: %s: %v", e.Stage, e.Kind, e.Err)
}

// Unwrap returns the underlying error.
func (e *Error) Unwrap() error { return e.Err }

// New returns a classified error.
func New(stage Stage, kind Kind, err error) *Error {
	return &Error{Stage: stage, Kind: kind, Err: err}
}

// InvalidConfigError reports a configuration that cannot work, such as a
// malformed endpoint.
func InvalidConfigError(err error) *Error { return New(StageConnect, InvalidConfig, err) }

// As returns the classified error in err's chain.
func As(err error) (*Error, bool) {
	return errors.AsType[*Error](err)
}

// FromStatus classifies a non-2xx response. body is a short excerpt of the
// response body.
func FromStatus(status int, body string) *Error {
	cause := fmt.Errorf("unexpected HTTP status %d", status)
	if body != "" {
		cause = fmt.Errorf("unexpected HTTP status %d: %s", status, body)
	}
	e := &Error{Stage: StageCapability, Kind: Protocol, Status: status, Err: cause}
	switch {
	case status == http.StatusUnauthorized:
		e.Stage, e.Kind = StageAuthenticate, Unauthorized
	case status == http.StatusForbidden:
		e.Stage, e.Kind = StageAuthorize, Forbidden
	case status == http.StatusNotFound:
		e.Kind = NotFound
	case status == http.StatusRequestTimeout, status == http.StatusGatewayTimeout:
		e.Stage, e.Kind = StageConnect, Timeout
	case status == http.StatusTooManyRequests:
		e.Kind = RateLimited
	case status >= http.StatusInternalServerError:
		e.Stage, e.Kind = StageConnect, Unavailable
	}
	return e
}

// redactURL removes user information and the query from address, where
// credentials may be written.
func redactURL(address string) string {
	parsed, err := url.Parse(address)
	if err != nil {
		return "<invalid URL>"
	}
	parsed.User, parsed.RawQuery, parsed.ForceQuery = nil, "", false
	return parsed.String()
}

// FromTransport classifies an error returned before a response arrived. It
// returns context cancellation unchanged and keeps errors that are already
// classified.
func FromTransport(err error) error {
	if err == nil || errors.Is(err, context.Canceled) {
		return err
	}
	if _, ok := As(err); ok {
		return err
	}
	// Remove credentials from the URL before the error is wrapped anywhere.
	if urlError, ok := errors.AsType[*url.Error](err); ok {
		redacted := *urlError
		redacted.URL = redactURL(urlError.URL)
		err = &redacted
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return New(StageConnect, Timeout, err)
	}
	var certificateInvalid x509.CertificateInvalidError
	var hostname x509.HostnameError
	var authority x509.UnknownAuthorityError
	var record tls.RecordHeaderError
	if errors.As(err, &certificateInvalid) || errors.As(err, &hostname) || errors.As(err, &authority) || errors.As(err, &record) {
		return New(StageConnect, TLS, err)
	}
	if urlError, ok := errors.AsType[*url.Error](err); ok && urlError.Timeout() {
		return New(StageConnect, Timeout, err)
	}
	if netError, ok := errors.AsType[net.Error](err); ok {
		if netError.Timeout() {
			return New(StageConnect, Timeout, err)
		}
		return New(StageConnect, Network, err)
	}
	return New(StageConnect, Unavailable, err)
}
