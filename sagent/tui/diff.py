"""Diffs in the scrollback: highlighted, with changed words marked.

Typical usage example:

  render_diff(console, splice_diff(result), path=result.path or "")
  console.print(highlight_source("ls -la", "bash"))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import dataclasses
import difflib
import re

from pygments import highlight
from pygments.formatters import Terminal256Formatter
from pygments.lexers import TextLexer, get_lexer_by_name, get_lexer_for_filename
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
from rich.cells import cell_len
from rich.text import Text

from sagent.tui.types import FILLS, Sticky


if TYPE_CHECKING:
    from pygments.lexer import Lexer
    from rich.console import Console

    from sagent.session.types import FileEditResult, FileWriteResult


__all__ = [
    "DiffStyle",
    "highlight_source",
    "render_diff",
    "splice_diff",
    "write_diff",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class DiffStyle:
    """The colors of a diff, for a dark terminal."""

    added: str = "on rgb(34,92,43)"
    removed: str = "on rgb(122,41,54)"
    added_word: str = "on rgb(56,166,96)"
    removed_word: str = "on rgb(179,89,107)"
    gutter: str = "rgb(160,160,160)"

    word_threshold: float = 0.4
    """Share of changed characters past which a line is marked whole, not by word."""


def render_diff(
    console: Console,
    diff: str,
    *,
    path: str = "",
    style: DiffStyle | None = None,
    lines: tuple[int, int] | None = None,
    sticky: Sticky | None = None,
) -> None:
    """Print a unified diff, its line numbers read from the ``@@`` headers.

    Args:
      console: Where to print.
      diff: The diff; ``---``/``+++`` file headers are skipped.
      path: The file, whose name picks the highlighting.
      style: The colors; dark-terminal ones when omitted.
      lines: Body lines kept from the head and the tail, the rest counted in
        a ``⋯ N lines ⋯`` row; ``None`` keeps all.
      sticky: This turn's width, shared with its other blocks.

    """
    s = style if style is not None else DiffStyle()
    lexer = _lexer_for(path)
    hunk = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")
    body = [
        ln for ln in diff.splitlines() if not re.match(r"^(?:---|\+\+\+)(?:\s|$)", ln)
    ]
    added = sum(1 for ln in body if ln.startswith("+"))
    removed = sum(1 for ln in body if ln.startswith("-"))
    console.print(
        Text(f"  ⎿  Added {added} lines, removed {removed} lines", style="dim"),
    )
    changes = [ln for ln in body if not hunk.match(ln)]
    pairs = _word_pairs(changes, threshold=s.word_threshold)
    digits = len(str(_last_number(body, hunk)))
    longest = max((cell_len(ln) for ln in changes), default=0)
    # The margin ``  N ``: the call's indent, the number, a space.
    fill = dataclasses.replace(FILLS["diff"], gutter=digits + 3)
    sticky = sticky if sticky is not None else Sticky()
    width = sticky.cells(fill, longest - 1, width=console.width)
    hidden = (
        range(lines[0], len(changes) - lines[1])
        if lines is not None and len(changes) > sum(lines)
        else range(0)
    )
    index = old = new = 0
    for line in body:
        if m := hunk.match(line):
            old, new = int(m.group(1)), int(m.group(2))
            continue
        sigil = line[:1] if line[:1] in "+-" else " "
        number = old if sigil == "-" else new
        if index in hidden:
            if index == hidden.start:
                indent = " " * (digits + 3)
                console.print(Text(f"{indent}⋯ {len(hidden)} lines ⋯", style="dim"))
        elif index in pairs:
            _print_words(
                console,
                f"{number:>{digits}} ",
                sigil,
                pairs[index],
                s=s,
                width=width,
                lexer=lexer,
            )
        else:
            background = {"+": s.added, "-": s.removed}.get(sigil, "")
            _print_line(
                console,
                f"{number:>{digits}} ",
                content=_highlight(line[1:], lexer),
                background=background,
                gutter=s.gutter,
                width=width,
            )
        old += sigil != "+"
        new += sigil != "-"
        index += 1


def splice_diff(result: FileEditResult) -> str:
    """Return an edit's replacements as a unified diff.

    Args:
      result: What Edit did.

    Returns:
      diff: One hunk per replacement, numbered for the file after the edit.

    """
    hunks: list[str] = []
    shift = 0
    for splice in result.edits:
        before = (splice.before or "").splitlines()
        after = (splice.after or "").splitlines()
        start = splice.start if splice.start is not None else 1
        hunks.append(f"@@ -{start},{len(before)} +{start + shift},{len(after)} @@")
        hunks.extend(f"-{ln}" for ln in before)
        hunks.extend(f"+{ln}" for ln in after)
        shift += len(after) - len(before)
    return "\n".join(hunks)


def write_diff(result: FileWriteResult) -> str:
    """Return a write's content as a unified diff adding every line.

    Args:
      result: What Write did.

    Returns:
      diff: One hunk adding the file's lines from line 1.

    """
    lines = (result.content or "").splitlines()
    return "\n".join([f"@@ -0,0 +1,{len(lines)} @@", *(f"+{ln}" for ln in lines)])


def highlight_source(code: str, language: str) -> Text:
    """Return ``code`` highlighted as ``language``; plain when the name is unknown.

    Args:
      code: The source.
      language: A Pygments lexer name, such as ``"bash"``.

    Returns:
      text: The highlighted source.

    """
    if not code:
        return Text("")
    try:
        lexer = get_lexer_by_name(language, stripnl=False, ensurenl=False)
    except ClassNotFound:
        return Text(code)
    return _highlight(code, lexer)


def _lexer_for(path: str) -> Lexer:
    """Return the lexer ``path``'s name picks, or plain text."""
    try:
        return get_lexer_for_filename(path, stripnl=False, ensurenl=False)
    except ClassNotFound:
        return TextLexer(stripnl=False, ensurenl=False)


def _highlight(code: str, lexer: Lexer) -> Text:
    """Return ``code`` highlighted by ``lexer``."""
    if not code:
        return Text("")
    formatter = Terminal256Formatter[str](style=_Monokai)
    return Text.from_ansi(highlight(code, lexer, formatter).rstrip("\n"))


def _words(
    removed: str,
    added: str,
    *,
    threshold: float,
) -> list[tuple[str, str]] | None:
    """Return a line pair's words as ``=``, ``-`` or ``+``; ``None`` past ``threshold``."""
    token = re.compile(r"(\s+|\w+|[^\s\w]+)")
    a, b = token.findall(removed), token.findall(added)
    parts: list[tuple[str, str]] = []
    changed = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        a=a,
        b=b,
        autojunk=False,
    ).get_opcodes():
        if tag == "equal":
            parts.append(("=", "".join(a[i1:i2])))
            continue
        if tag in ("delete", "replace"):
            parts.append(("-", "".join(a[i1:i2])))
            changed += len(parts[-1][1])
        if tag in ("insert", "replace"):
            parts.append(("+", "".join(b[j1:j2])))
            changed += len(parts[-1][1])
    total = len(removed) + len(added)
    return None if total and changed / total > threshold else parts


# A run of ``-`` lines pairs with the ``+`` run after it, aligned by ``difflib`` so a
# deleted line does not shift every pairing after it.
def _word_pairs(
    lines: list[str],
    *,
    threshold: float,
) -> dict[int, list[tuple[str, str]]]:
    """Return the word parts of each body line that pairs with a changed partner."""
    pairs: dict[int, list[tuple[str, str]]] = {}
    i = 0
    while i < len(lines):
        if not lines[i].startswith("-"):
            i += 1
            continue
        r_start = i
        while i < len(lines) and lines[i].startswith("-"):
            i += 1
        a_start = i
        while i < len(lines) and lines[i].startswith("+"):
            i += 1
        removed = [ln[1:] for ln in lines[r_start:a_start]]
        added = [ln[1:] for ln in lines[a_start:i]]
        matcher = difflib.SequenceMatcher(a=removed, b=added, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag not in ("equal", "replace"):
                continue
            for r, a in zip(range(i1, i2), range(j1, j2), strict=False):
                parts = _words(removed[r], added[a], threshold=threshold)
                if parts is not None:
                    pairs[r_start + r] = parts
                    pairs[a_start + a] = parts
    return pairs


def _last_number(body: list[str], hunk: re.Pattern[str]) -> int:
    """Return the highest line number the diff's margin shows."""
    old = new = last = 0
    for line in body:
        if m := hunk.match(line):
            old, new = int(m.group(1)), int(m.group(2))
            continue
        sigil = line[:1]
        last = max(last, old if sigil == "-" else new)
        old += sigil != "+"
        new += sigil != "-"
    return last


def _print_line(
    console: Console,
    margin: str,
    *,
    content: Text,
    background: str,
    gutter: str,
    width: int,
) -> None:
    """Print one diff line: margin, content, and a background to the edge."""
    style = f"{gutter} {background}" if background else "dim"
    if background:
        content = content.copy()
        content.stylize(background)
    _print_rows(
        console,
        margin,
        content,
        margin_style=style,
        fill_style=style if background else "",
        width=width,
    )


def _print_words(
    console: Console,
    margin: str,
    sigil: str,
    parts: list[tuple[str, str]],
    *,
    s: DiffStyle,
    width: int,
    lexer: Lexer,
) -> None:
    """Print one changed line with its changed words marked."""
    line_bg = s.added if sigil == "+" else s.removed
    word_bg = s.added_word if sigil == "+" else s.removed_word
    kept = [(kind, text) for kind, text in parts if kind in ("=", sigil)]
    content = _highlight("".join(text for _, text in kept), lexer)
    at = 0
    for kind, text in kept:
        content.stylize(line_bg if kind == "=" else word_bg, at, at + len(text))
        at += len(text)
    _print_rows(
        console,
        margin,
        content,
        margin_style=f"{s.gutter} {line_bg}",
        fill_style=line_bg,
        width=width,
    )


def _print_rows(
    console: Console,
    margin: str,
    content: Text,
    *,
    margin_style: str,
    fill_style: str,
    width: int,
) -> None:
    """Print ``content`` wrapped inside ``width``, later rows indented past ``margin``."""
    # Under the ``⎿``, at the indent of the call above; layout, so no style.
    indent = "  "
    rows = content.wrap(
        console,
        max(1, width - len(indent) - len(margin)),
        overflow="fold",
    )
    for i, row in enumerate(rows or [Text("")]):
        lead = margin if i == 0 else " " * len(margin)
        pad = max(0, width - len(indent) - len(margin) - row.cell_len)
        console.print(
            Text.assemble(
                indent,
                Text(lead, style=margin_style),
                row,
                Text(" " * pad, style=fill_style),
            ),
            no_wrap=True,
        )


class _Monokai(Style):
    """Monokai Extended, in syntect's hex values."""

    styles = {  # noqa: RUF012 -- Pygments reads a class-level mapping.
        Token: "#f8f8f2",
        Keyword: "#f92672",
        Keyword.Type: "#66d9ef",
        Name.Builtin: "#a6e22e",
        Name.Class: "#a6e22e",
        Name.Function: "#a6e22e",
        Name.Attribute: "#a6e22e",
        Name.Decorator: "#a6e22e",
        Name.Variable: "#ffffff",
        Number: "#ae84ff",
        String: "#e6db74",
        String.Escape: "#ae84ff",
        Comment: "#75715e",
        Operator: "#f92672",
        Punctuation: "#f8f8f2",
        Name.Namespace: "#f92672",
        Name.Tag: "#f92672",
    }
