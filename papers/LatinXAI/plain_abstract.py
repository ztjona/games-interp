"""Emit the abstract as plain unicode text for the OpenReview submission field.

The LXAI author guidelines require the abstract as plain unicode, max 2000
characters, so every LaTeX construct is resolved to its unicode equivalent
rather than stripped: \\times -> U+00D7, ``...'' -> U+201C/U+201D, -- -> U+2013.
Run this after any edit to the abstract in main.tex so the two cannot drift.

Usage:
    plain_abstract.py <main.tex> [--out=PATH] [--ascii]
"""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

MAX_CHARS = 2000


def extract(tex: str) -> str:
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    if not m:
        raise SystemExit("no abstract block found")
    return m.group(1)


def to_unicode(s: str) -> str:
    # inline markup: keep the content, drop the styling
    for cmd in ("emph", "textbf", "textit", "texttt", "textsc"):
        # repeat: handles the (rare) nested case
        while True:
            new = re.sub(r"\\" + cmd + r"\{([^{}]*)\}", r"\1", s)
            if new == s:
                break
            s = new
    s = s.replace(r"\times", "\u00d7")
    s = s.replace(r"\%", "%")
    s = s.replace(r"\&", "&")
    s = s.replace("\\$", "$")
    s = re.sub(r"\$([^$]*)\$", r"\1", s)          # unwrap inline math
    s = s.replace("``", "\u201c").replace("''", "\u201d")
    s = s.replace("---", "\u2014")                 # em dash
    s = re.sub(r"(?<=\d)--(?=\d)", "\u2013", s)    # numeric range -> en dash
    s = s.replace("--", "\u2013")
    s = s.replace("{,}", ",").replace("~", "\u00a0")
    s = re.sub(r"\\[a-zA-Z]+", "", s)              # any stragglers
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    tex = Path(args[0]).read_text(encoding="utf-8")
    out = to_unicode(extract(tex))

    if "--ascii" in flags:
        out = (out.replace("\u00d7", "x").replace("\u2013", "-")
                  .replace("\u2014", "--").replace("\u201c", '"')
                  .replace("\u201d", '"').replace("\u00a0", " "))

    dest = next((f.split("=", 1)[1] for f in flags if f.startswith("--out=")), None)
    if dest:
        Path(dest).write_text(out + "\n", encoding="utf-8")

    n = len(out)
    print(out)
    print()
    print(f"characters: {n} / {MAX_CHARS}"
          + (f"  ({MAX_CHARS - n} to spare)" if n <= MAX_CHARS
             else f"  OVER BY {n - MAX_CHARS}"))
    print(f"words: {len(out.split())}")
    non_ascii = sorted({c for c in out if ord(c) > 127})
    print("non-ASCII characters used: "
          + (", ".join(f"{c!r} U+{ord(c):04X} {unicodedata.name(c, '?')}"
                       for c in non_ascii) or "none"))
    leftovers = [t for t in ("\\", "{", "}", "$", "``", "''") if t in out]
    print("leftover LaTeX markup: " + (", ".join(map(repr, leftovers)) or "none"))
    if dest:
        print(f"wrote -> {dest}")


if __name__ == "__main__":
    main()
