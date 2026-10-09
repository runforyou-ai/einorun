// Package toolname builds valid model-visible names for tools that come from
// other systems, such as MCP servers.
//
// Model APIs accept tool names of ASCII letters, digits, underscores and
// hyphens, up to 64 characters. External servers and tools can be named in
// any script and can collide once reduced to that alphabet, so a name is
// built from readable words and ends with a digest of the server's stable
// identity and the original tool name:
//
//	<namespace>__<server>__<tool>_<digest>
package toolname

import (
	"crypto/sha256"
	"encoding/hex"
	"regexp"
	"strings"
)

const (
	// MaxLength is the longest tool name model APIs accept.
	MaxLength = 64
	// digestLength is the number of hexadecimal digits of the digest.
	digestLength = 8
)

// valid matches the names model APIs accept.
var valid = regexp.MustCompile(`^[A-Za-z0-9_-]{1,64}$`)

// Valid reports whether name is a valid model-visible tool name.
func Valid(name string) bool {
	return valid.MatchString(name)
}

// Namer builds tool names. The zero value drops characters other than ASCII
// letters and digits.
type Namer struct {
	// Transliterate returns ASCII words for a rune that is not an ASCII letter
	// or digit, such as the pinyin of a Chinese character; an empty result
	// treats the rune as a separator. Characters of the result other than
	// ASCII letters and digits separate words.
	Transliterate func(r rune) string
}

// Name returns the model-visible name of tool on a server. namespace groups
// the names of one kind of source, such as "mcp"; identity is the server's
// stable identity, which tells apart servers with the same display name; name
// is the server's display name. The result is valid, at most MaxLength
// characters and always ends with the digest of identity and tool, so it
// stays the same while the identity and tool name do.
func (n Namer) Name(namespace, identity, server, tool string) string {
	sum := sha256.Sum256([]byte(identity + "\x00" + tool))
	suffix := "_" + hex.EncodeToString(sum[:])[:digestLength]
	parts := make([]string, 0, 3)
	if prefix := n.Slug(namespace); prefix != "" {
		parts = append(parts, prefix)
	}
	parts = append(parts, or(n.Slug(server), "server"), or(n.Slug(tool), "tool"))
	body := strings.Join(parts, "__")
	body = strings.TrimRight(body[:min(len(body), MaxLength-len(suffix))], "_")
	return body + suffix
}

// Slug turns s into ASCII words joined by underscores: runs of ASCII
// letters and digits form words, other runes are transliterated or separate
// words.
func (n Namer) Slug(s string) string {
	var words []string
	var word strings.Builder
	flush := func() {
		if word.Len() > 0 {
			words = append(words, word.String())
			word.Reset()
		}
	}
	for _, r := range s {
		if isAlnum(r) {
			word.WriteRune(r)
			continue
		}
		flush()
		if n.Transliterate == nil {
			continue
		}
		// Transliterated text is split into words by the same rule.
		for _, t := range n.Transliterate(r) {
			if isAlnum(t) {
				word.WriteRune(t)
			} else {
				flush()
			}
		}
		flush()
	}
	flush()
	return strings.Join(words, "_")
}

// Name returns the model-visible name of tool with the zero Namer.
func Name(namespace, identity, server, tool string) string {
	return Namer{}.Name(namespace, identity, server, tool)
}

// isAlnum reports whether r is an ASCII letter or digit.
func isAlnum(r rune) bool {
	return r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9'
}

// or returns a when it is not empty, otherwise b.
func or(a, b string) string {
	if a != "" {
		return a
	}
	return b
}
