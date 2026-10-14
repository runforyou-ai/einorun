package einorun

import (
	"context"

	"github.com/runforyou-ai/einorun/llm"
)

// MediaAdapter projects media the model cannot take natively into text or
// other accepted modalities. Hosts supply OCR, transcription and document
// conversion; nil on the request means no projection.
type MediaAdapter interface {
	Adapt(ctx context.Context, ref MediaRef, policy AdaptPolicy) (Adaptation, error)
}

// AdaptPolicy tells the adapter which modalities the model still accepts and
// the largest item the run will take.
type AdaptPolicy struct {
	Accepted []llm.Modality
	MaxBytes int64
}

// Adaptation is what the model should see instead of, or after failing, a
// native media block.
type Adaptation struct {
	Parts []AdaptPart
}

// AdaptPart is one piece of an adaptation: text (Modality nil) or a native
// block built from URL or bytes.
type AdaptPart struct {
	// Modality is the native modality, or nil for text.
	Modality *llm.Modality
	MIME     string
	URL      string
	Data     []byte
	Text     string
	// Kind labels the projection for notes and metering (ocr, transcript,
	// document); it does not change runtime branching.
	Kind string
}
