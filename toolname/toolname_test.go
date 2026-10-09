package toolname

import (
	"strings"
	"testing"
)

func TestName(t *testing.T) {
	got := Name("mcp", "local:files", "My Files", "read-file")
	if !strings.HasPrefix(got, "mcp__My_Files__read_file_") || len(got) != len("mcp__My_Files__read_file_")+digestLength || !Valid(got) {
		t.Fatalf("got %q", got)
	}
	if again := Name("mcp", "local:files", "My Files", "read-file"); again != got {
		t.Fatalf("not stable: %q %q", got, again)
	}
}

func TestNameTellsApartCollisions(t *testing.T) {
	a := Name("mcp", "server-a", "files", "read file")
	b := Name("mcp", "server-b", "files", "read file")
	c := Name("mcp", "server-a", "files", "read_file")
	if a == b || a == c {
		t.Fatalf("collision: %q %q %q", a, b, c)
	}
}

func TestNameFallbacks(t *testing.T) {
	got := Name("", "id", "文件", "读取")
	if !strings.HasPrefix(got, "server__tool_") || !Valid(got) {
		t.Fatalf("got %q", got)
	}
}

func TestNameTransliterates(t *testing.T) {
	pinyin := map[rune]string{'文': "wen", '件': "jian", '读': "du", '取': "qu"}
	n := Namer{Transliterate: func(r rune) string { return pinyin[r] }}
	got := n.Name("mcp", "id", "文件", "读取 v2")
	if !strings.HasPrefix(got, "mcp__wen_jian__du_qu_v2_") || !Valid(got) {
		t.Fatalf("got %q", got)
	}
}

func TestNameSanitizesTransliteration(t *testing.T) {
	n := Namer{Transliterate: func(rune) string { return "ä-b c" }}
	if got := n.Slug("x字y"); got != "x_b_c_y" {
		t.Fatalf("got %q", got)
	}
}

func TestNameIsBounded(t *testing.T) {
	long := Name("mcp", "id", strings.Repeat("server", 20), strings.Repeat("tool", 20))
	if len(long) != MaxLength || !Valid(long) || long != Name("mcp", "id", strings.Repeat("server", 20), strings.Repeat("tool", 20)) {
		t.Fatalf("got %q (%d)", long, len(long))
	}
	// The cut falls on the separator; underscores before the digest are dropped.
	cut := Name("a", "id", strings.Repeat("b", 51), "c")
	if cut[:len(cut)-digestLength-1] != "a__"+strings.Repeat("b", 51) {
		t.Fatalf("got %q", cut)
	}
}

func TestValid(t *testing.T) {
	for name, want := range map[string]bool{"web_search": true, "a-b": true, "": false, "a b": false, "工具": false, strings.Repeat("a", 65): false} {
		if Valid(name) != want {
			t.Errorf("Valid(%q) = %v", name, !want)
		}
	}
}
