package discovery

import (
	"context"
	"errors"
	"testing"

	"github.com/runforyou-ai/einorun/provider/vendor"
)

func TestDiscoverUnsupportedManual(t *testing.T) {
	_, err := Discover(context.Background(), nil, Endpoint{Brand: vendor.Manual, BaseURL: "https://example.com/v1"})
	if !errors.Is(err, ErrUnsupported) {
		t.Fatalf("got %v, want ErrUnsupported", err)
	}
}
