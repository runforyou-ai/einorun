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
//
// Some APIs also require the first character to be a letter or an
// underscore; names start with an underscore when the words would start
// with a digit.
package toolname

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"regexp"
	"strings"
)

const (
	// MaxLength is the longest tool name model APIs accept.
	MaxLength = 64
	// digestLength is the number of hexadecimal digits of the digest.
	digestLength = 12
	// maxNamespace bounds the namespace words.
	maxNamespace = 16
)

// valid matches the names every supported model API accepts.
var valid = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_-]{0,63}$`)

// Valid reports whether name is a valid model-visible tool name: ASCII
// letters, digits, underscores and hyphens, starting with a letter or an
// underscore, at most MaxLength characters.
func Valid(name string) bool {
	return valid.MatchString(name)
}

// Namer builds tool names. The zero value drops characters other than ASCII
// letters and digits, so a server named "文件" becomes the fallback word
// "server" and only the digest tells such servers apart.
type Namer struct {
	// Transliterate returns ASCII words for a rune that is not an ASCII letter
	// or digit, such as the pinyin of a Chinese character; an empty result
	// treats the rune as a separator. Each transliterated rune forms its own
	// words ("文件" becomes "wen_jian"); characters of the result other than
	// ASCII letters and digits separate words.
	Transliterate func(r rune) string
}

// Name returns the model-visible name of tool on a server. namespace groups
// the names of one kind of source, such as "mcp"; identity is the server's
// stable identity, which tells apart servers with the same display name;
// server is the server's display name. The result is valid and at most
// MaxLength characters. It depends on every argument: the same arguments
// always give the same name, and a renamed server gets new names, so hosts
// that keep names across runs pass a server label that does not change, and
// map names back to tools themselves. The digest of namespace, identity and
// tool, which ends every name, keeps the names of different servers and
// tools apart even when their words are the same. When the words are too
// long, the server and the tool share the room left, so the tool keeps a
// readable part.
func (n Namer) Name(namespace, identity, server, tool string) string {
	sum := digest(namespace, identity, tool)
	suffix := "_" + hex.EncodeToString(sum[:])[:digestLength]
	prefix := cut(n.Slug(namespace), maxNamespace)
	serverWords, toolWords := or(n.Slug(server), "server"), or(n.Slug(tool), "tool")
	room := MaxLength - len(suffix) - len("____") - 1 // separators and a leading underscore
	if prefix != "" {
		room -= len(prefix) + len("__")
	}
	if len(serverWords)+len(toolWords) > room {
		toolWords = cut(toolWords, max(room/2, room-len(serverWords)))
		serverWords = cut(serverWords, room-len(toolWords))
	}
	parts := []string{serverWords, toolWords}
	if prefix != "" {
		parts = append([]string{prefix}, parts...)
	}
	name := strings.Join(parts, "__") + suffix
	if name[0] >= '0' && name[0] <= '9' {
		name = "_" + name
	}
	return name
}

// digest hashes the fields with their lengths, so no two field lists share
// an encoding.
func digest(fields ...string) [sha256.Size]byte {
	var encoded []byte
	for _, field := range fields {
		encoded = binary.AppendUvarint(encoded, uint64(len(field)))
		encoded = append(encoded, field...)
	}
	return sha256.Sum256(encoded)
}

// cut returns s at most limit bytes long without trailing underscores.
func cut(s string, limit int) string {
	return strings.TrimRight(s[:min(len(s), limit)], "_")
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
