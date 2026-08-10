"""Terminal rendering helpers for the REPL.

Extracted from repl.py so the REPL module itself owns only
PT-level concerns (keybindings, event loop, prompt layout). Anything
Rich-based that transforms structured input into terminal output
lives here:

- ``render_diff_detail`` - colored diff with syntax highlighting
  and word-level change emphasis.
- ``find_stable_boundary`` - split a streaming markdown string at a
  paragraph boundary that preserves fence parity, so the safe prefix
  can render now and the uncommitted tail keeps buffering.

Surface scope: terminal (Rich + pygments). A Slack or web surface
would have its own analogous module with the relevant formatting
library; this file is intentionally not generic across surfaces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import difflib
import re

from markdown_it import MarkdownIt
from pygments import highlight
from pygments.formatters import Terminal256Formatter
from pygments.lexers import (
    TextLexer,
    get_lexer_by_name,
    get_lexer_for_filename,
)
from pygments.style import Style
from pygments.token import (
    Comment,
    Keyword,
    Name,
    Number,
    Operator,
    Punctuation,
    String,
    Token,
)
from pygments.util import ClassNotFound
from rich.text import Text


if TYPE_CHECKING:
    from pygments.lexer import Lexer
    from rich.console import Console


class _MonokaiStyle(Style):
    """Monokai Extended color palette (syntect-derived hex values)."""

    styles = {  # noqa: RUF012 -- Pygments Style requires mutable class-level dict
        Token: "#f8f8f2",
        Keyword: "#f92672",  # rgb(249,38,114) - pink
        Keyword.Type: "#66d9ef",  # storage/type - cyan
        Name.Builtin: "#a6e22e",  # rgb(166,226,46) - green
        Name.Class: "#a6e22e",
        Name.Function: "#a6e22e",
        Name.Attribute: "#a6e22e",
        Name.Decorator: "#a6e22e",
        Name.Variable: "#ffffff",
        Number: "#ae84ff",  # rgb(190,132,255) - purple
        String: "#e6db74",  # rgb(230,219,116) - yellow
        String.Escape: "#ae84ff",
        Comment: "#75715e",  # rgb(117,113,94) - gray
        Operator: "#f92672",  # pink
        Punctuation: "#f8f8f2",  # near-white
        Name.Namespace: "#f92672",
        Name.Tag: "#f92672",
    }


_DIFF_CONTEXT_LINES = 3  # config-globals: ignore -- diff context lines, display pref
_DIFF_FORMATTER: Terminal256Formatter[str] = Terminal256Formatter(style=_MonokaiStyle)

# Diff background colors (dark mode).
_DIFF_ADDED_STYLE: Final = "on rgb(34,92,43)"  # Dark green.
_DIFF_REMOVED_STYLE: Final = "on rgb(122,41,54)"  # Dark red.
_DIFF_ADDED_WORD_STYLE: Final = "on rgb(56,166,96)"  # Brighter green.
_DIFF_REMOVED_WORD_STYLE: Final = "on rgb(179,89,107)"  # Brighter red.

# Muted gray for diff gutter line numbers.
_GUTTER_FG: Final = "rgb(160,160,160)"

# Word-diff falls back to line-level highlighting once the changed
# characters exceed this fraction of the total compared characters.
# 40% is the empirical sweet spot: below this, the word diff is still
# legible (a couple of changed tokens per line); above, the highlight
# noise overwhelms the surrounding context and the line diff is
# clearer. Adjust together with the diff fixtures.
_WORD_DIFF_THRESHOLD = (
    0.4  # config-globals: ignore -- word-diff fallback threshold, display pref
)

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")
# Unified-diff file-header lines (``--- a/foo`` / ``+++ b/foo``) start
# with ``-``/``+`` but are not hunk content. Without this filter
# ``_pair_word_diffs`` would pair the headers against the first hunk's
# remove/add and emit a nonsense word diff against ``a/foo`` vs ``b/foo``.
_FILE_HEADER_RE = re.compile(r"^(?:---|\+\+\+)(?:\s|$)")
_WORD_RE = re.compile(r"(\s+|\w+|[^\s\w]+)")

_md = MarkdownIt()


def render_diff_detail(console: Console, diff: str, file_path: str = "") -> None:
    """Render a unified diff string as a colored diff.

    Line numbers are read directly from @@ headers (already absolute).

    Args:
      console: Rich console to print to.
      diff: Unified diff text (lines starting with ``+``/``-``/`` ``).
      file_path: Filename hint for syntax highlighting.

    """
    lexer = _get_lexer(file_path)
    lines = [ln for ln in diff.splitlines() if not _FILE_HEADER_RE.match(ln)]

    added = sum(1 for ln in lines if ln.startswith("+"))
    removed = sum(1 for ln in lines if ln.startswith("-"))
    console.print(
        Text(f"  ⎿  Added {added} lines, removed {removed} lines", style="dim"),
    )

    body_lines = [ln for ln in lines if not _HUNK_RE.match(ln)]
    word_pairs = _pair_word_diffs(body_lines)
    width = console.width

    body_idx = 0
    old_ln = new_ln = 0
    for line in lines:
        m = _HUNK_RE.match(line)
        if m:
            old_ln = int(m.group(1))
            new_ln = int(m.group(2))
            continue

        if line.startswith("-"):
            if body_idx in word_pairs:
                _, parts = word_pairs[body_idx]
                _render_word_diff_line(
                    console,
                    old_ln,
                    "-",
                    parts=parts,
                    is_add=False,
                    width=width,
                    lexer=lexer,
                )
            else:
                content_text = _highlight(line[1:], lexer)
                _render_diff_line(
                    console,
                    old_ln,
                    "-",
                    content_text=content_text,
                    bg_style=_DIFF_REMOVED_STYLE,
                    width=width,
                )
            old_ln += 1
        elif line.startswith("+"):
            if body_idx in word_pairs:
                _, parts = word_pairs[body_idx]
                _render_word_diff_line(
                    console,
                    new_ln,
                    "+",
                    parts=parts,
                    is_add=True,
                    width=width,
                    lexer=lexer,
                )
            else:
                content_text = _highlight(line[1:], lexer)
                _render_diff_line(
                    console,
                    new_ln,
                    "+",
                    content_text=content_text,
                    bg_style=_DIFF_ADDED_STYLE,
                    width=width,
                )
            new_ln += 1
        else:
            content_text = _highlight(line[1:], lexer)
            _render_diff_line(
                console,
                new_ln,
                " ",
                content_text=content_text,
                bg_style=None,
                width=width,
            )
            old_ln += 1
            new_ln += 1
        body_idx += 1


def find_stable_boundary(text: str) -> int:
    """Find the character offset of the last complete block boundary.

    Tokenises ``text`` with markdown-it and returns the character offset
    at the start of the last block-level token.  Everything before that
    offset is stable - complete blocks that cannot change as more text
    arrives.  The last block is still potentially growing.

    Returns 0 when fewer than two block tokens are present (nothing is
    stable yet).

    Args:
      text: Streaming markdown string to split.

    Returns:
      offset: Character offset of the stable/unstable boundary.

    """
    tokens = _md.parse(text)
    block_lines = [t.map[0] for t in tokens if t.map is not None and t.level == 0]
    if len(block_lines) < 2:
        return 0
    boundary_line = block_lines[-1]
    lines = text.split("\n")
    return sum(len(lines[i]) + 1 for i in range(boundary_line))


def highlight_source(code: str, lang: str) -> Text:
    """Syntax-highlight ``code`` for a named language.

    Args:
      code: Source text, one line or many.
      lang: Pygments lexer name, e.g. ``"bash"``. An unknown name falls
          back to plain text rather than raising -- highlighting is
          decoration, never a reason to lose the content.

    Returns:
      text: Rich ``Text`` carrying the highlighted source.

    """
    if not code:
        return Text("")
    try:
        lexer = get_lexer_by_name(lang, stripnl=False, ensurenl=False)
    except ClassNotFound:
        return Text(code)
    return _highlight(code, lexer)


def _get_lexer(filepath: str) -> Lexer:
    """Pygments lexer for the file's extension, or TextLexer."""
    try:
        return get_lexer_for_filename(filepath, stripnl=False, ensurenl=False)
    except ClassNotFound:
        return TextLexer(stripnl=False, ensurenl=False)


def _highlight(code: str, lexer: Lexer) -> Text:
    """Apply pygments syntax highlighting, return Rich Text."""
    if not code:
        return Text("")
    return Text.from_ansi(highlight(code, lexer, _DIFF_FORMATTER).rstrip("\n"))


def _word_diff_pair(
    removed: str,
    added: str,
) -> list[tuple[str, str]] | None:
    """Return word-level diff parts, or None if change is too large.

    Each tuple is ``(kind, text)`` where kind is ``"="`` (unchanged),
    ``"-"`` (in removed), or ``"+"`` (in added). Returns None if the
    ratio of changed chars exceeds the threshold.
    """
    r_words = _WORD_RE.findall(removed)
    a_words = _WORD_RE.findall(added)
    matcher = difflib.SequenceMatcher(a=r_words, b=a_words, autojunk=False)
    parts: list[tuple[str, str]] = []
    changed = 0
    total = len(removed) + len(added)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            parts.append(("=", "".join(r_words[i1:i2])))
        elif tag == "delete":
            chunk = "".join(r_words[i1:i2])
            parts.append(("-", chunk))
            changed += len(chunk)
        elif tag == "insert":
            chunk = "".join(a_words[j1:j2])
            parts.append(("+", chunk))
            changed += len(chunk)
        elif tag == "replace":
            r_chunk = "".join(r_words[i1:i2])
            a_chunk = "".join(a_words[j1:j2])
            parts.append(("-", r_chunk))
            parts.append(("+", a_chunk))
            changed += len(r_chunk) + len(a_chunk)
    if total > 0 and changed / total > _WORD_DIFF_THRESHOLD:
        return None
    return parts


def _align_blocks(
    removed: list[str],
    added: list[str],
) -> list[tuple[int, int]]:
    """Return (removed_idx, added_idx) pairs aligned by similarity.

    Uses :class:`difflib.SequenceMatcher` on the line sequences so a
    deletion in the middle doesn't shift all subsequent pairings.
    Falls back to positional pairing inside each ``replace`` region
    - once ``difflib`` has localized a block as "no shared lines",
    word diff is our last shot at showing structure.
    """
    sm = difflib.SequenceMatcher(a=removed, b=added, autojunk=False)
    alignments: list[tuple[int, int]] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            alignments.extend(
                zip(range(i1, i2), range(j1, j2), strict=True),
            )
        elif tag == "replace":
            pair_count = min(i2 - i1, j2 - j1)
            alignments.extend(
                zip(
                    range(i1, i1 + pair_count),
                    range(j1, j1 + pair_count),
                    strict=True,
                ),
            )
        # ``delete`` / ``insert`` have no counterpart to pair with.
    return alignments


def _pair_word_diffs(
    diff_lines: list[str],
) -> dict[int, tuple[int, list[tuple[str, str]]]]:
    """Pair adjacent -/+ blocks for word-level highlighting.

    Returns a map: diff index → (partner diff index, word parts).
    Only indices where word diff is viable (below threshold).
    """
    pairs: dict[int, tuple[int, list[tuple[str, str]]]] = {}
    i = 0
    while i < len(diff_lines):
        if not diff_lines[i].startswith("-"):
            i += 1
            continue
        # Collect consecutive -'s.
        r_start = i
        while i < len(diff_lines) and diff_lines[i].startswith("-"):
            i += 1
        r_end = i
        # Collect consecutive +'s.
        a_start = i
        while i < len(diff_lines) and diff_lines[i].startswith("+"):
            i += 1
        a_end = i
        if a_start == a_end:
            continue  # No add block to pair with.
        removed = [diff_lines[j][1:] for j in range(r_start, r_end)]
        added = [diff_lines[j][1:] for j in range(a_start, a_end)]
        for r_off, a_off in _align_blocks(removed, added):
            r_idx = r_start + r_off
            a_idx = a_start + a_off
            parts = _word_diff_pair(removed[r_off], added[a_off])
            if parts is not None:
                pairs[r_idx] = (a_idx, parts)
                pairs[a_idx] = (r_idx, parts)
    return pairs


def _render_diff_line(
    console: Console,
    lineno: int,
    sigil: str,
    *,
    content_text: Text,
    bg_style: str | None,
    width: int,
) -> None:
    """Render one diff line with gutter, content, and full-width bg."""
    gutter_str = f"  {lineno:>4} {sigil} "
    style = f"{_GUTTER_FG} {bg_style}" if bg_style else "dim"
    gutter = Text(gutter_str, style=style)
    if bg_style:
        # Copy before stylize: callers pass fresh ``_highlight()``
        # results today, but mutating the input is a footgun.
        content_text = content_text.copy()
        content_text.stylize(bg_style)
    used = len(gutter_str) + content_text.cell_len
    pad = max(0, width - used)
    if pad > 0:
        padding = Text(" " * pad, style=style) if bg_style else Text(" " * pad)
        console.print(Text.assemble(gutter, content_text, padding))
    else:
        console.print(Text.assemble(gutter, content_text))


def _render_word_diff_line(
    console: Console,
    lineno: int,
    sigil: str,
    *,
    parts: list[tuple[str, str]],
    is_add: bool,
    width: int,
    lexer: Lexer,
) -> None:
    """Render a line with word-level highlighting on changed spans."""
    line_bg = _DIFF_ADDED_STYLE if is_add else _DIFF_REMOVED_STYLE
    word_bg = _DIFF_ADDED_WORD_STYLE if is_add else _DIFF_REMOVED_WORD_STYLE
    keep = "+" if is_add else "-"
    filtered = [(kind, text) for kind, text in parts if kind in ("=", keep)]
    content_str = "".join(text for _, text in filtered)
    content_text = _highlight(content_str, lexer)
    pos = 0
    for kind, text in filtered:
        end = pos + len(text)
        content_text.stylize(line_bg if kind == "=" else word_bg, pos, end)
        pos = end
    gutter_str = f"  {lineno:>4} {sigil} "
    gutter = Text(gutter_str, style=f"{_GUTTER_FG} {line_bg}")
    pad = max(0, width - len(gutter_str) - content_text.cell_len)
    if pad > 0:
        console.print(
            Text.assemble(gutter, content_text, Text(" " * pad, style=line_bg)),
        )
    else:
        console.print(Text.assemble(gutter, content_text))
