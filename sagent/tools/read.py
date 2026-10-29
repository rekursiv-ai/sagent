"""Read: a file's text, line-numbered, or its pictures.

Text comes back numbered, with a note where to resume if cut. An image comes
back as itself, a PDF as a JPEG per page, a notebook as cells and outputs.

Typical usage example:

  result = await Read().call(context, {"file_path": "a.py", "offset": 10})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from typing import TYPE_CHECKING, Annotated, cast

import asyncio
import io
import json
import mimetypes
import re

from treekle import PlainTree, ReadError, from_plain, json_loads

from sagent.model.image import PDFIUM
from sagent.session.types import (
    Attachment,
    FileReadResult,
    UncategorizedToolResult,
)
from sagent.tools.command_analyzer import (
    Invocation,
    bounding_sink,
    replaceable,
    resolve,
    sed_mutates,
)
from sagent.tools.display import ToolDisplay
from sagent.tools.types import Arg, parse, schema_of


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    import pypdfium2 as pdfium

    from sagent.tools.types import ToolContext
else:
    from wrapt import lazy_import

    pdfium = lazy_import("pypdfium2")


__all__ = [
    "Read",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class Read:
    """Read a file's text, line-numbered, or its pictures."""

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Arguments:
        """What the model passes."""

        file_path: Annotated[str, "Absolute path to the file."]
        offset: Annotated[
            int,
            "Line number to start from (1-based, text files only).",
            Arg(minimum=1),
        ] = 1
        limit: Annotated[
            int | None,
            "Max lines to return (text files only). Omit to read until the"
            " result is full; the reply names the offset to resume from.",
            Arg(minimum=1),
        ] = None
        last_lines: Annotated[
            int | None,
            "Read from the final N lines of a text file, such as a log or a"
            " stack trace. Composes with ``limit``: ``last_lines=100,"
            " limit=10`` returns the first 10 of the final 100 lines."
            " ``offset`` is ignored.",
            Arg(minimum=1),
        ] = None
        pages: Annotated[str, 'Page range for PDFs, e.g. "1-5" or "3".'] = ""

    name = "Read"
    description = (files(__package__) / "assets" / "read.md").read_text(
        encoding="utf-8",
    )
    schema = schema_of(Arguments)

    display: ToolDisplay = field(default=ToolDisplay())

    max_pdf_bytes: int = 50 << 20
    """Largest PDF opened at all."""

    max_pdf_pages: int = 10
    """Most pages read when no ``pages`` range is given."""

    max_rendered_bytes: int = 12 << 20
    """Most JPEG bytes one read renders, under the smallest vendor request limit."""

    max_image_bytes: int = 50 << 20
    """Largest image read; the provider shrinks what it sends."""

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        """Return ``Read <path>``, with the lines asked for.

        Args:
          arguments: What the model passed to Read.

        Returns:
          summary: Such as ``Read pkg/a.py:10-19`` or ``Read log:last-50``.

        """
        name = str(arguments.get("file_path", ""))
        offset, limit = arguments.get("offset"), arguments.get("limit")
        last_lines = arguments.get("last_lines")
        if isinstance(last_lines, int):
            return f"Read {name}:last-{last_lines}"
        if isinstance(offset, int) and isinstance(limit, int):
            return f"Read {name}:{offset}-{offset + limit - 1}"
        if isinstance(offset, int):
            return f"Read {name}:{offset}+"
        if isinstance(limit, int):
            return f"Read {name}:1-{limit}"
        return f"Read {name}"

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        """Return the file, so a read after an edit of it sees the edit."""
        return str(context.path(str(arguments.get("file_path", ""))).resolve())

    def from_shell(
        self,
        invocation: Invocation,
    ) -> Mapping[str, PlainTree] | str | None:
        """Return the Read call for ``cat``, ``head``, ``tail``, or ``sed -n 'M,Np'``.

        Args:
          invocation: One simple command, with its pipeline.

        Returns:
          arguments: The Read call that shows the same lines; a message when
            the read cannot be translated, such as ``cat`` of many files; or
            ``None`` when the command is not a plain read.

        """
        deny = frozenset(
            {
                *("-c", "-f", "-F", "--bytes", "--follow", "--retry", "-A", "-v"),
                *("-e", "-E", "-t", "-T", "--show-all", "--show-nonprinting"),
            },
        )
        counts = frozenset({"-n", "-c", "--lines", "--bytes"})
        takes_value = {
            "head": counts,
            "tail": counts,
            "sed": frozenset({"-e", "-f", "--expression", "--file"}),
        }.get(invocation.exe, frozenset[str]())
        if not replaceable(
            invocation,
            exes=frozenset({"cat", "head", "tail", "sed"}),
            deny=deny,
            takes_value=takes_value,
        ):
            return None
        if invocation.exe == "sed" and not _sed_prints(invocation.args):
            return None
        refusal = (
            "Read each file with the Read tool, all in one block: one round trip,"
            " and lines numbered to cite."
        )
        window = _window(invocation)
        if window is None:
            return refusal
        path, lines = window
        target = resolve(invocation.cwd, path)
        if target is None or any(c in target for c in "*?[$`"):
            return refusal
        return {"file_path": target, **lines}

    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> FileReadResult | UncategorizedToolResult:
        """Read ``file_path``.

        Args:
          context: The calling agent's state.
          arguments: What the model passed, as ``Read.Arguments``.

        Returns:
          result: What the file holds, or why it could not be read.

        """
        a = parse(arguments, Read.Arguments)
        try:
            return await asyncio.to_thread(self._read, context, a)
        except OSError as e:
            return _error(f"Error reading {a.file_path}: {e}")

    def _read(
        self,
        context: ToolContext,
        a: Read.Arguments,
    ) -> FileReadResult | UncategorizedToolResult:
        """Read the file ``a`` names, by its kind."""
        path = context.path(a.file_path)
        if not path.exists():
            return _error(f"File not found: {path}")
        if path.is_dir():
            return _error(f"{path} is a directory. Use List or Glob.")
        if path.suffix.lower() == ".pdf":
            return self._read_pdf(context, path, pages=a.pages)
        if path.suffix.lower() == ".ipynb":
            return _read_notebook(context, path)
        mime, _ = mimetypes.guess_type(path)
        # Not SVG: it is XML, and vendors refuse ``image/svg+xml``.
        if mime in {"image/png", "image/jpeg", "image/gif", "image/bmp", "image/webp"}:
            size = path.stat().st_size
            if size > self.max_image_bytes:
                return _error(
                    f"{path} is {size:,} bytes, over the {self.max_image_bytes:,}"
                    " a read takes. Shrink it with Bash first.",
                )
            context.files.saw(path)
            return FileReadResult(
                call_id="",
                path=str(path),
                content=f"[image: {path}]",
                attachments=(Attachment(mime_descriptor=mime, data=path.read_bytes()),),
            )
        return _read_text(
            context,
            path,
            offset=a.offset,
            limit=a.limit,
            last_lines=a.last_lines,
        )

    def _read_pdf(
        self,
        context: ToolContext,
        path: Path,
        *,
        pages: str,
    ) -> FileReadResult | UncategorizedToolResult:
        """Render a PDF's pages as JPEGs, as many as fit the byte budget."""
        with path.open("rb") as f:
            if f.read(5) != b"%PDF-":
                return _error(f"{path} is not a PDF: it lacks the %PDF- header.")
        size = path.stat().st_size
        if size > self.max_pdf_bytes:
            return _error(
                f"{path} is {size:,} bytes, over the {self.max_pdf_bytes:,} a read opens.",
            )
        match = re.fullmatch(r"(\d+)(?:-(\d*))?", pages.strip())
        if pages and match is None:
            return _error(f'Invalid pages {pages!r}: use "N", "N-M", or "N-".')
        with PDFIUM:
            try:
                doc = pdfium.PdfDocument(str(path))
            except pdfium.PdfiumError as e:
                return _error(f"{path} is not a readable PDF: {e}")
            try:
                total = len(doc)
                if match is None:
                    if total > self.max_pdf_pages:
                        return _error(
                            f"{path} has {total} pages, over {self.max_pdf_pages}."
                            ' Pass pages="1-N" to read some.',
                        )
                    first, last = 1, total
                elif match[2] is None:
                    first = last = int(match[1])
                else:
                    first = int(match[1])
                    last = min(int(match[2]), total) if match[2] else total
                if first < 1 or last < first or last > total:
                    return _error(f"{path} has no pages {pages} (it has {total}).")
                jpegs = _render(
                    doc,
                    first=first,
                    last=last,
                    max_bytes=self.max_rendered_bytes,
                )
            finally:
                doc.close()
        if not jpegs:
            return _error(f"Page {first} of {path} alone is too large to send.")
        context.files.saw(path)
        end = first + len(jpegs) - 1
        note = f"[PDF: {path.name}, pages {first}-{end} of {total}]"
        if end < last:
            note += f' Pass pages="{end + 1}-{last}" for the rest.'
        return FileReadResult(
            call_id="",
            path=str(path),
            content=note,
            attachments=tuple(
                Attachment(mime_descriptor="image/jpeg", data=jpeg) for jpeg in jpegs
            ),
        )


def _window(invocation: Invocation) -> tuple[str, Mapping[str, PlainTree]] | None:
    """Return the one file a read names and the Read window it shows."""
    args = invocation.args
    if invocation.exe == "sed":
        script = next(a for a in args if re.fullmatch(r"\d+(,(\d+|\$))?p", a))
        paths = [a for a in args if not a.startswith("-") and a != script]
        first, _, last = script.removesuffix("p").partition(",")
        if last in {"", first}:
            lines: dict[str, PlainTree] = {"offset": int(first), "limit": 1}
        elif last == "$":
            lines = {"offset": int(first)}
        elif int(last) >= int(first):
            lines = {"offset": int(first), "limit": int(last) - int(first) + 1}
        else:
            return None
        return (paths[0], lines) if len(paths) == 1 and int(first) >= 1 else None
    if invocation.exe == "cat":
        # ``-n`` only: Read numbers every line anyway.
        if any(a.startswith("-") and a != "-n" for a in args):
            return None
        paths = [a for a in args if a != "-n"]
        source: tuple[str, int, str] = ("cat", 0, "")
    else:
        paths, count, sign = _count(args)
        source = (invocation.exe, count, sign)
    sinks = [s for s in invocation.downstream() if s.exe in {"head", "tail"}]
    # ``-`` names the input, which a Bash call leaves empty, not a file.
    if paths[:1] == ["-"] or len(paths) != 1 or len(sinks) + (source[0] != "cat") > 1:
        return None
    sink = bounding_sink(invocation)
    exe, count, sign = (sink.exe, *_count(sink.args)[1:]) if sink else source
    if exe == "cat":
        return paths[0], dict[str, PlainTree]()
    # No Read takes a zero count; ``head -n 0`` and ``head -n abc`` show nothing.
    if count < 1:
        return None
    if exe == "head" and sign != "-":
        return paths[0], {"limit": count}
    if exe == "tail":
        return paths[0], {"offset": count} if sign == "+" else {"last_lines": count}
    return None


# The sign picks the end: ``tail -n +5`` starts at line 5; ``head -n -5``
# drops the last 5, which no Read window expresses.
def _count(args: tuple[str, ...]) -> tuple[list[str], int, str]:
    """Return a ``head`` or ``tail``'s operands, line count, and its sign."""
    paths: list[str] = []
    count, sign = 10, ""
    i = 0
    while i < len(args):
        arg, raw = args[i], ""
        if arg in {"-n", "--lines"} and i + 1 < len(args):
            raw, i = args[i + 1], i + 2
        elif arg.startswith("--lines="):
            raw, i = arg.partition("=")[2], i + 1
        elif arg.startswith("-n") and len(arg) > 2:
            raw, i = arg[2:], i + 1
        elif arg.startswith("-") and arg != "-":
            if arg[1:].isdigit():
                count, sign = int(arg[1:]), ""
            i += 1
            continue
        else:
            paths.append(arg)
            i += 1
            continue
        sign = raw[:1] if raw[:1] in {"+", "-"} else ""
        digits = raw.lstrip("+-")
        count = int(digits) if digits.isdigit() else 0
    return paths, count, sign


def _sed_prints(args: tuple[str, ...]) -> bool:
    """Return whether ``sed`` with ``args`` only prints a line range, ``-n 'M,Np'``."""
    quiet = any(a.startswith("-") and not a.startswith("--") and "n" in a for a in args)
    scripts = [a for a in args if re.fullmatch(r"\d+(,(\d+|\$))?p", a)]
    return not sed_mutates(args) and quiet and len(scripts) == 1


def _read_text(
    context: ToolContext,
    path: Path,
    *,
    offset: int,
    limit: int | None,
    last_lines: int | None,
) -> FileReadResult | UncategorizedToolResult:
    """Read ``path`` as numbered lines, windowed and bounded."""
    lines_asked = (offset, limit, last_lines)
    if context.files.shown(path, lines=lines_asked):
        return UncategorizedToolResult(
            call_id="",
            content=f"[File unchanged since last read: {path}]",
        )
    with path.open("rb") as f:
        if b"\x00" in f.read(8192):
            return UncategorizedToolResult(
                call_id="",
                content=f"[Binary file: {path} ({path.stat().st_size:,} bytes).]",
            )
    # Seen before read, so a write in between shows as a change.
    context.files.saw(path)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return UncategorizedToolResult(
            call_id="",
            content=f"[{path} is not UTF-8. Use Bash with an explicit decoder.]",
        )
    if not text:
        return UncategorizedToolResult(
            call_id="",
            content=f"[File exists but is empty: {path}]",
        )
    lines = text.splitlines(keepends=True)
    total = len(lines)
    start = offset - 1 if last_lines is None else max(0, total - last_lines)
    if start >= total:
        return UncategorizedToolResult(
            call_id="",
            content=f"[offset {offset} is past the end: {path} has {total} lines]",
        )
    end = total if limit is None else min(start + limit, total)
    note = f"\n... ({total:,} more lines. Use offset={total + 1} to continue.)"
    body, kept = context.fit(
        (f"{start + i + 1}\t{line}" for i, line in enumerate(lines[start:end])),
        reserve=len(note),
    )
    if start + kept < total:
        body += (
            f"\n... ({total - start - kept:,} more lines."
            f" Use offset={start + kept + 1} to continue.)"
        )
    context.files.showed(path, lines=lines_asked)
    whole = start == 0 and kept == total
    return FileReadResult(
        call_id="",
        path=str(path),
        content=body,
        ranges=() if whole else ((start + 1, kept),),
    )


def _read_notebook(
    context: ToolContext,
    path: Path,
) -> FileReadResult | UncategorizedToolResult:
    """Render a notebook's cells and their outputs as text."""
    try:
        notebook = json_loads(path.read_text(encoding="utf-8"), dict[str, object])
        cells = from_plain(notebook.get("cells", []), list[dict[str, object]])
    except (json.JSONDecodeError, UnicodeDecodeError, ReadError) as e:
        return _error(f"{path} is not a Jupyter notebook: {e}")
    context.files.saw(path)
    parts: list[str] = []
    for i, cell in enumerate(cells):
        parts.append(f"--- Cell {i + 1} ({cell.get('cell_type') or 'code'}) ---\n")
        parts.append(f"{_joined(cell.get('source'))}\n")
        outputs = cell.get("outputs")
        for output in (
            cast("list[object]", outputs) if isinstance(outputs, list) else ()
        ):
            if isinstance(output, dict):
                text = _output_text(cast("dict[str, object]", output))
                if text:
                    parts.append(f"[output] {text}\n")
    note = f"\n... ({len(parts):,} more notebook sections omitted.)"
    body, kept = context.fit(parts, reserve=len(note))
    if kept < len(parts):
        body += f"\n... ({len(parts) - kept:,} more notebook sections omitted.)"
    return FileReadResult(
        call_id="",
        path=str(path),
        content=body or "(empty notebook)",
    )


# An output says what it is three ways: ``stream`` under ``text``, a value under
# ``data['text/plain']``, and an ``error`` under ``traceback``.
def _output_text(output: dict[str, object]) -> str:
    """Return what one cell output shows, as text."""
    data = output.get("data")
    plain = (
        cast("dict[str, object]", data).get("text/plain")
        if isinstance(data, dict)
        else None
    )
    return (
        _joined(output.get("text"))
        or _joined(plain)
        or _joined(output.get("traceback"))
    )


def _joined(raw: object) -> str:
    """Return a notebook text field, which is a string or a list of lines."""
    if isinstance(raw, list):
        return "".join(str(x) for x in cast("list[object]", raw))
    return raw if isinstance(raw, str) else ""


def _render(
    doc: pdfium.PdfDocument,
    *,
    first: int,
    last: int,
    max_bytes: int,
) -> list[bytes]:
    """Render pages ``first`` to ``last`` as JPEGs, stopping before ``max_bytes``."""
    jpegs: list[bytes] = []
    used = 0
    for i in range(first - 1, last):
        page = doc[i]
        try:
            # Twice 72 DPI: legible to a vision model, small enough to send.
            image = page.render(scale=2).to_pil().convert("RGB")
        finally:
            page.close()
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85, optimize=True)
        if used + buf.tell() > max_bytes:
            break
        used += buf.tell()
        jpegs.append(buf.getvalue())
    return jpegs


def _error(message: str) -> UncategorizedToolResult:
    return UncategorizedToolResult(call_id="", content=message, is_error=True)
