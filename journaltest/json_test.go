package journaltest

import "testing"

func TestSameJSON(t *testing.T) {
	for _, c := range []struct {
		a, b string
		same bool
	}{
		{`{"a":1,"b":2}`, `{"b": 2, "a": 1}`, true},
		{`{"n":9007199254740993}`, `{"n":9007199254740992}`, false},
		{`{"a":1}`, `{"a":"1"}`, false},
		{`{"n":1}`, `{"n":1.0}`, true},
		{`{"a":1} {}`, `{"a":1}`, false},
		{``, ``, true},
		{``, `{}`, false},
	} {
		if got := sameJSON([]byte(c.a), []byte(c.b)); got != c.same {
			t.Fatalf("sameJSON(%s, %s) = %v", c.a, c.b, got)
		}
	}
}
