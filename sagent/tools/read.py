"""Read tool: text, image, PDF, and notebook file reading."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import asyncio
import json

from sagent.lib.json import (
    JSON,
    MutableJSON,
    MutableJSONValue,
    int_val,
    json_freeze,
)
from sagent.tools.core import (
    get_tool_state,
    load_tool_description,
    mark_read,
    resolve_tool_path,
)
from sagent.tools.lib.bash import (
    Node,
    match_pipeline,
    unwrap_cd_prefix,
)
from sagent.tools.lib.pdf import (
    MAX_INLINE_PAGES,
    MAX_PDF_BYTES,
    PdfError,
    extract_pdf_pages,
    get_pdf_page_count,
    is_pdf,
    parse_page_range,
)
from sagent.types.history import BytesMessage, ToolResult


_IMAGE_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".webp",
    ".svg",
}
_PDF_EXT = ".pdf"
_NOTEBOOK_EXT = ".ipynb"
_NUDGE = "cat via Bash is a bad UX. Use the Read tool."
_CAT_SHAPERS: frozenset[str] = frozenset({"head", "tail", "less", "more"})
_MIME_BY_EXT: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
}


class Read:
    """Read file contents: text, image, PDF, and notebook."""

    name: str = "Read"
    tool_id: str = "application/x-tool-read"
    description: str = load_tool_description("Read")
    supports_microcompaction: bool = True
    emit_tool_summary: bool = False
    directive_schema: JSON = json_freeze(
        {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the file.",
                },
                "offset": {
                    "type": "integer",
                    "minimum": 1,
                    "description": (
                        "Line number to start from (1-based, text files only)."
                        " Must be ≥ 1."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "description": (
                        "Max lines to return (default 2000, text files only)."
                        " Must be ≥ 1."
                    ),
                },
                "last_lines": {
                    "type": "integer",
                    "minimum": 1,
                    "description": (
                        "Read from the final N lines of a text file. Use when"
                        " the needed content is expected near EOF, such as"
                        " logs, recent output, appended config, stack traces,"
                        " or files whose relevant section is at the end."
                        " Composes with ``limit``: ``last_lines=100,"
                        " limit=10`` returns the first 10 of the final 100"
                        " lines. ``offset`` is ignored when ``last_lines`` is"
                        " set. Must be ≥ 1."
                    ),
                },
                "pages": {
                    "type": "string",
                    "description": 'Page range for PDFs, e.g. "1-5" or "3".',
                },
            },
            "required": ["file_path"],
        }
    )

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        """Read a file and return its contents.

        Args:
          args: Directive with ``file_path`` and optional windowing keys
              (``offset``, ``limit``, ``last_lines``, ``pages``).

        Returns:
          result: Text content (with line numbers), image/PDF attachments,
              or an error when the file is missing or unsupported.

        """
        file_path = resolve_tool_path(str(args.get("file_path", "")))
        offset = int_val(args.get("offset"), 1)
        limit = int_val(args.get("limit"), 2000)
        last_lines = int_val(args.get("last_lines"), 0)
        pages = str(args.get("pages", ""))
        return await asyncio.to_thread(
            self._run,
            file_path=file_path,
            offset=offset,
            limit=limit,
            last_lines=last_lines,
            pages=pages,
        )

    def summary(self, args: Mapping[str, object]) -> str:
        """Return a short label for this tool invocation.

        Args:
          args: Directive carrying ``file_path`` and windowing keys.

        Returns:
          label: ``Read <basename><range>`` line shown before invocation.

        """
        file_path = str(args.get("file_path", ""))
        fname = Path(file_path).name if file_path else "?"
        offset = int_val(args.get("offset"), 0)
        limit = int_val(args.get("limit"), 0)
        last_lines = int_val(args.get("last_lines"), 0)
        if last_lines > 0:
            suffix = f":last-{last_lines}"
        elif offset > 0 and limit > 0:
            suffix = f":{offset}-{offset + limit}"
        elif offset > 0:
            suffix = f":{offset}+"
        elif limit > 0:
            suffix = f":1-{limit}"
        else:
            suffix = ""
        return f"Read {fname}{suffix}"

    def summary_result(self, result: ToolResult) -> str | None:
        """One-line receipt: line count for text, ``binary``/``unchanged`` markers.

        Args:
          result: Completed ``ToolResult`` from ``run``.

        Returns:
          receipt: Short receipt line, or ``None`` when suppressed/empty.

        """
        if not self.emit_tool_summary or result.is_error:
            return None
        text = result.content
        has_binary = bool(result.attachments)
        if not text:
            return "binary" if has_binary else None
        if has_binary:
            return "binary"
        if text.startswith("[File unchanged"):
            return "unchanged"
        return f"{text.count(chr(10))} lines"

    def prompt(self) -> str:
        """Return supplemental prompt text for this tool.

        Returns:
          contribution: Empty string (no per-request prompt fragment).

        """
        return ""

    def _run(
        self,
        *,
        file_path: str,
        offset: int = 1,
        limit: int = 2000,
        last_lines: int = 0,
        pages: str = "",
    ) -> ToolResult:
        """Dispatch to the type-specific reader (text, image, PDF, notebook)."""
        p = Path(file_path)
        if not p.exists():
            return ToolResult(
                call_id="",
                content=f"File not found: {file_path}",
                is_error=True,
            )
        if p.is_dir():
            return ToolResult(
                call_id="",
                content=(
                    f"{file_path} is a directory, not a file."
                    " Use Glob to inspect directory contents."
                ),
                is_error=True,
            )

        suffix = p.suffix.lower()

        if (
            suffix not in _IMAGE_EXTS
            and suffix not in {_PDF_EXT, _NOTEBOOK_EXT}
            and get_tool_state().check_unchanged(
                file_path,
                offset,
                limit,
                last_lines=last_lines,
            )
        ):
            return ToolResult(
                call_id="",
                content=f"[File unchanged since last read: {file_path}]",
            )

        if suffix in _IMAGE_EXTS:
            mark_read(file_path, offset=offset, limit=limit, last_lines=last_lines)
            return _read_image(p, file_path=file_path, suffix=suffix)

        if suffix == _PDF_EXT:
            mark_read(file_path, offset=offset, limit=limit, last_lines=last_lines)
            return _read_pdf(p, pages)

        if suffix == _NOTEBOOK_EXT:
            mark_read(file_path, offset=offset, limit=limit, last_lines=last_lines)
            return _read_notebook(p, file_path=file_path)

        return _read_text(
            p,
            file_path=file_path,
            offset=offset,
            limit=limit,
            last_lines=last_lines,
        )

    def bash_match(self, trees: Sequence[Node]) -> str | None:
        """Emit a hint if the command is ``cat``/``head``/``tail``.

        Args:
          trees: Parsed bashlex command trees from the active Bash call.

        Returns:
          hint: Nudge string redirecting to the Read tool, or ``None``.

        """
        single = self._match_single(trees)
        if single is not None:
            return single
        return _match_pipeline_read(trees)

    def _match_single(self, trees: Sequence[Node]) -> str | None:
        """Match a single ``cat``/``head``/``tail`` command for a Read nudge."""
        unwrapped = unwrap_cd_prefix(trees)
        if unwrapped is None:
            return None
        cwd, cmd = unwrapped
        if cmd.env_prefix:
            return None
        if cmd.exe == "cat":
            return _match_cat(cwd, cmd.args)
        if cmd.exe == "head":
            return _match_head_tail(cwd, cmd.args, which="head")
        if cmd.exe == "tail":
            return _match_head_tail(cwd, cmd.args, which="tail")
        return None


def _read_image(p: Path, *, file_path: str, suffix: str) -> ToolResult:
    """Return an image file as a ToolResult with a JPEG/PNG attachment."""
    return ToolResult(
        call_id="",
        content=f"[image: {file_path}]",
        attachments=(BytesMessage(p.read_bytes(), _image_mime(suffix)),),
    )


def _read_notebook(p: Path, *, file_path: str) -> ToolResult:
    """Parse a Jupyter notebook and return cell contents as text."""
    try:
        nb = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return ToolResult(
            call_id="",
            content=f"[Invalid notebook JSON: {file_path}: {e}]",
        )
    except UnicodeDecodeError as e:
        return ToolResult(
            call_id="",
            content=f"[Non-UTF-8 notebook: {file_path}: {e}]",
        )
    if not isinstance(nb, dict):
        return ToolResult(
            call_id="",
            content=f"[Not a valid Jupyter notebook: {file_path}]",
        )
    nb_d = cast(MutableJSON, nb)
    cells_raw = cast(list[MutableJSONValue], nb_d.get("cells") or [])
    parts: list[str] = []
    for i, cell in enumerate(cells_raw):
        if not isinstance(cell, dict):
            continue
        cell_d = cast(MutableJSON, cell)
        ctype = str(cell_d.get("cell_type") or "code")
        source_raw = cell_d.get("source")
        source = (
            "".join(str(x) for x in source_raw)
            if isinstance(source_raw, list)
            else str(source_raw or "")
        )
        parts.append(f"--- Cell {i + 1} ({ctype}) ---")
        parts.append(source)
        _collect_cell_outputs(cell_d, parts)
    return ToolResult(
        call_id="",
        content="\n".join(parts) or "(empty notebook)",
    )


def _collect_cell_outputs(cell: MutableJSON, parts: list[str]) -> None:
    """Append text outputs from a notebook cell to ``parts``."""
    outputs_raw = cell.get("outputs")
    if not isinstance(outputs_raw, list):
        return
    for out in outputs_raw:
        if not isinstance(out, dict):
            continue
        out_d = cast(MutableJSON, out)
        text_raw = out_d.get("text")
        if text_raw is None:
            continue
        text_out = (
            "".join(str(x) for x in text_raw)
            if isinstance(text_raw, list)
            else str(text_raw)
        )
        parts.append("[output] " + text_out)


def _read_text(
    p: Path,
    *,
    file_path: str,
    offset: int,
    limit: int,
    last_lines: int,
) -> ToolResult:
    """Read a text file with offset/limit/last_lines windowing."""
    with p.open("rb") as f:
        head = f.read(8192)
    if b"\x00" in head:
        size = p.stat().st_size
        return ToolResult(
            call_id="",
            content=f"[Binary file: {file_path} ({size} bytes). Use Bash to inspect.]",
        )
    try:
        mtime = p.stat().st_mtime
    except OSError:
        mtime = None
    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        size = p.stat().st_size
        return ToolResult(
            call_id="",
            content=(
                f"[Non-UTF-8 file: {file_path} ({size} bytes)."
                " Use Bash with an explicit decoder to inspect.]"
            ),
        )
    mark_read(
        file_path,
        offset=offset,
        limit=limit,
        last_lines=last_lines,
        content=text,
        mtime=mtime,
    )
    if not text:
        return ToolResult(
            call_id="",
            content=f"[File exists but is empty: {file_path}]",
        )
    body = _window_text(
        text, file_path=file_path, offset=offset, limit=limit, last_lines=last_lines
    )
    return ToolResult(call_id="", content=body)


def _window_text(
    text: str,
    *,
    file_path: str,
    offset: int,
    limit: int,
    last_lines: int,
) -> str:
    """Apply offset/limit/last_lines windowing and add line numbers."""
    lines = text.splitlines(keepends=True)
    total = len(lines)
    if last_lines > 0:
        start = max(0, total - last_lines)
    else:
        offset = max(1, offset)
        start = offset - 1
    if start >= total:
        return f"[offset {offset} beyond EOF: {file_path} has {total} lines]"
    end = total if limit <= 0 else min(start + limit, total)
    selected = lines[start:end]
    numbered = [f"{start + i + 1}\t{line}" for i, line in enumerate(selected)]
    result_str = "".join(numbered)
    if end < total:
        result_str += (
            f"\n... ({total - end} more lines. Use offset={end + 1} to continue.)"
        )
    return result_str


def _match_cat(cwd: str | None, args: tuple[str, ...]) -> str | None:
    """Match ``cat FILE`` (exactly one positional, no flags) for a Read nudge."""
    del cwd  # Hint is a fixed string; path resolution is the LLM's job.
    if len(args) != 1 or args[0].startswith("-"):
        return None
    return "cat via Bash is a bad UX. Use the Read tool."


def _match_head_tail(
    cwd: str | None,
    args: tuple[str, ...],
    *,
    which: str,
) -> str | None:
    """Validate ``head``/``tail`` args and return a fixed hint.

    Supported shapes: ``<cmd> FILE``, ``<cmd> -n N FILE``,
    ``<cmd> -N FILE``. Anything else (e.g. ``-c`` bytes, bundled
    flags) bails.
    """
    del cwd  # Hint is a fixed string; path resolution is the LLM's job.
    positional: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith("--"):
            return None
        if a == "-n":
            if i + 1 >= len(args):
                return None
            try:
                int(args[i + 1])
            except ValueError:
                return None
            i += 2
            continue
        if a.startswith("-") and a != "-":
            rest = a[1:]
            if rest.isdigit():
                i += 1
                continue
            return None
        positional.append(a)
        i += 1
    if len(positional) != 1:
        return None
    return f"{which} via Bash is a bad UX. Use the Read tool."


def _match_pipeline_read(trees: Sequence[Node]) -> str | None:
    """Match ``cat FILE | head/tail/less/more``."""
    pair = match_pipeline(trees)
    if pair is None:
        return None
    first, second = pair
    if first.exe != "cat":
        return None
    if len(first.args) != 1 or first.args[0].startswith("-"):
        return None
    if second.exe not in _CAT_SHAPERS:
        return None
    return _NUDGE


def _image_mime(suffix: str) -> str:
    """MIME type for an image file suffix (defaults to ``application/octet-stream``)."""
    return _MIME_BY_EXT.get(suffix, "application/octet-stream")


def _read_pdf(path: Path, pages: str) -> ToolResult:
    """Rasterize (a range of) a PDF's pages to JPEG attachments."""
    if not is_pdf(path):
        return ToolResult(
            call_id="",
            content=f"Not a PDF: {path} (missing %PDF- header)",
            is_error=True,
        )
    size = path.stat().st_size
    if size > MAX_PDF_BYTES:
        return ToolResult(
            call_id="",
            content=(
                f"PDF too large: {size} bytes > {MAX_PDF_BYTES}."
                ' Use pages="N-M" to read a range.'
            ),
            is_error=True,
        )

    resolved = _resolve_page_range(path, pages)
    if isinstance(resolved, ToolResult):
        return resolved
    first, last = resolved

    try:
        page_jpegs = extract_pdf_pages(path, first=first, last=last)
    except PdfError as e:
        return ToolResult(call_id="", content=f"PDF: {e}", is_error=True)

    range_note = f" pages {first}-{last or 'end'}" if first is not None else ""
    return ToolResult(
        call_id="",
        content=f"[PDF: {path.name} ({len(page_jpegs)} page(s){range_note})]",
        attachments=tuple(BytesMessage(jpeg, "image/jpeg") for jpeg in page_jpegs),
    )


def _resolve_page_range(
    path: Path, pages: str
) -> tuple[int | None, int | None] | ToolResult:
    """Validate page-range spec and return ``(first, last)`` or an error."""
    if pages:
        parsed = parse_page_range(pages)
        if parsed is None:
            return ToolResult(
                call_id="",
                content=(
                    f"Invalid pages spec {pages!r}: "
                    'use "N" / "N-M" / "N-" (1-indexed, inclusive)'
                ),
                is_error=True,
            )
        return parsed
    count = get_pdf_page_count(path)
    if count is not None and count > MAX_INLINE_PAGES:
        return ToolResult(
            call_id="",
            content=(
                f"PDF has {count} pages (> {MAX_INLINE_PAGES})."
                ' Pass pages="1-N" to read a window.'
            ),
            is_error=True,
        )
    return None, None
