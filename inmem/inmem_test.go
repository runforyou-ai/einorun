package inmem_test

import (
	"context"
	"testing"

	"github.com/runforyou-ai/einorun"
	"github.com/runforyou-ai/einorun/inmem"
	"github.com/runforyou-ai/einorun/journaltest"
)

func TestJournal(t *testing.T) {
	journaltest.RunJournal(t, func(*testing.T) journaltest.JournalHarness {
		j := inmem.NewJournal()
		return journaltest.JournalHarness{
			Journal:  j,
			Load:     func(context.Context) (einorun.Resume, error) { return j.Resume(), nil },
			Usage:    func(context.Context) (einorun.Usage, error) { return j.Usage(), nil },
			External: j.External,
		}
	})
}

func TestFeed(t *testing.T) {
	journaltest.RunFeed(t, func(*testing.T) journaltest.FeedHarness {
		f := inmem.NewFeed()
		return journaltest.FeedHarness{
			Feed:    f,
			Append:  func(_ context.Context, m einorun.Message) (int64, error) { return f.Append(m), nil },
			Consume: func(_ context.Context, seq int64) error { f.Consume(seq); return nil },
		}
	})
}
