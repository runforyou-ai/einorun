// Package prompt holds the model-facing text that einorun writes itself, in
// Chinese and English.
package prompt

// Catalog is the library's model-facing text in one language.
type Catalog struct {
	// StructuredRetry asks the model to repeat a structured answer that could
	// not be decoded. %s is the decode error.
	StructuredRetry string
}

var catalogs = map[string]*Catalog{
	"zh": {
		StructuredRetry: "上面的输出无法解析为要求的 JSON 对象（%s）。按原要求重新输出，只输出一个 JSON 对象。",
	},
	"en": {
		StructuredRetry: "The output above could not be parsed as the requested JSON object (%s). Answer the original request again and output a single JSON object only.",
	},
}

// For returns the catalog for language, falling back to English.
func For(language string) *Catalog {
	if c, ok := catalogs[language]; ok {
		return c
	}
	return catalogs["en"]
}
