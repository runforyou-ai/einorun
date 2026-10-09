package vendor

import "testing"

func TestCompatibleURL(t *testing.T) {
	cases := []struct {
		brand Brand
		in    string
		want  string
	}{
		{OpenAI, "https://api.openai.com/v1/", "https://api.openai.com/v1"},
		{Alibaba, "https://dashscope.aliyuncs.com", "https://dashscope.aliyuncs.com/compatible-mode/v1"},
		{Alibaba, "https://dashscope.aliyuncs.com/api/v1", "https://dashscope.aliyuncs.com/compatible-mode/v1"},
		{Alibaba, "https://dashscope.aliyuncs.com/compatible-mode/v1/", "https://dashscope.aliyuncs.com/compatible-mode/v1"},
		{Ollama, "http://localhost:11434", "http://localhost:11434/v1"},
		{Ollama, "http://localhost:11434/v1", "http://localhost:11434/v1"},
	}
	for _, c := range cases {
		preset, ok := Of(c.brand)
		if !ok {
			t.Fatalf("%s unknown", c.brand)
		}
		got, err := preset.CompatibleURL(c.in)
		if err != nil || got != c.want {
			t.Fatalf("%s %s: %q %v, want %q", c.brand, c.in, got, err, c.want)
		}
	}
	preset, _ := Of(OpenAI)
	if _, err := preset.CompatibleURL("localhost:8080"); err == nil {
		t.Fatal("accepted an endpoint without scheme")
	}
}

func TestPresets(t *testing.T) {
	for _, brand := range Brands {
		preset, ok := Of(brand)
		if !ok || preset.Structured == "" || preset.Rerank == "" {
			t.Fatalf("%s: %+v", brand, preset)
		}
	}
	if _, ok := Of("unknown"); ok {
		t.Fatal("unknown brand reported as known")
	}
	if p, _ := Of(Anthropic); !p.RequiresMaxOutputTokens || p.Structured != StructuredForcedTool {
		t.Fatalf("anthropic %+v", p)
	}
}

func TestAppendPath(t *testing.T) {
	got, err := AppendPath("https://example.com/base/", "v1/models")
	if err != nil || got != "https://example.com/base/v1/models" {
		t.Fatalf("%q %v", got, err)
	}
}
