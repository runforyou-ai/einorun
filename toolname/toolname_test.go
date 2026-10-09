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

func TestNameDigestSeparatesFields(t *testing.T) {
	if Name("mcp", "x\x00", "server", "y") == Name("mcp", "x", "server", "\x00y") {
		t.Fatal("fields with separators collide")
	}
	long := strings.Repeat("n", 40)
	if Name(long+"a", "id", "s", "t") == Name(long+"b", "id", "s", "t") {
		t.Fatal("namespaces cut to the same words collide")
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
	if len(long) > MaxLength || !Valid(long) {
		t.Fatalf("got %q (%d)", long, len(long))
	}
	// A long server name leaves the tool a readable part.
	shared := Name("mcp", "id", strings.Repeat("s", 80), "delete_record")
	if !strings.Contains(shared, "__delete_record_") || len(shared) > MaxLength {
		t.Fatalf("got %q", shared)
	}
	// Words cut at a separator leave no trailing underscores.
	words := Name("", "id", strings.Repeat("ab_", 30), strings.Repeat("cd_", 30))
	if strings.Contains(words, "___") || !Valid(words) {
		t.Fatalf("got %q", words)
	}
}

func TestNameStartsWithLetterOrUnderscore(t *testing.T) {
	got := Name("", "id", "123 Files", "read")
	if !strings.HasPrefix(got, "_123_Files__read_") || !Valid(got) {
		t.Fatalf("got %q", got)
	}
	for _, name := range []string{Name("1mcp", "id", "server", "tool"), Name("", "id", strings.Repeat("9", 80), strings.Repeat("8", 80))} {
		if !Valid(name) || len(name) > MaxLength {
			t.Fatalf("got %q (%d)", name, len(name))
		}
	}
}

func TestNameDependsOnEveryArgument(t *testing.T) {
	base := Name("mcp", "local:files", "My Files", "read-file")
	if renamed := Name("mcp", "local:files", "Renamed", "read-file"); renamed == base || renamed[len(renamed)-digestLength:] != base[len(base)-digestLength:] {
		t.Fatalf("base %q renamed %q", base, renamed)
	}
}

func TestValid(t *testing.T) {
	for name, want := range map[string]bool{"web_search": true, "a-b": true, "_x": true, "1x": false, "": false, "a b": false, "工具": false, strings.Repeat("a", 65): false} {
		if Valid(name) != want {
			t.Errorf("Valid(%q) = %v", name, !want)
		}
	}
}
