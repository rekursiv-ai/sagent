"""Read tool: text, image, PDF, and notebook file reading."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import json

from sagent.custom_types import (
    BytesDescriptor,
    BytesMessage,
    Message,
    MultipartMessage,
    TextMessage,
)
from sagent.lib.json import (
    JSON,
    MutableJSON,
    MutableJSONValue,
    int_val,
    json_freeze,
)
from sagent.lib.message import get_directive
from sagent.tools.core import (
    get_tool_state,
    load_tool_description,
    mark_read,
    resolve_tool_path,
    run_sync,
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

    async def run(self, msg: Message) -> Message:
        """Read a file and return its contents.

        Args:
          msg: Incoming tool-use message containing the directive.

        Returns:
          result: Tool result Message with file contents.

        """
        directive = get_directive(msg)
        file_path = resolve_tool_path(str(directive.get("file_path", "")))
        offset = int_val(directive.get("offset"), 1)
        limit = int_val(directive.get("limit"), 2000)
        last_lines = int_val(directive.get("last_lines"), 0)
        pages = str(directive.get("pages", ""))
        return await run_sync(
            self._run,
            parent_id=msg.id,
            file_path=file_path,
            offset=offset,
            limit=limit,
            last_lines=last_lines,
            pages=pages,
        )

    def summary(self, msg: Message) -> str:
        """Return a short label for this tool invocation.

        Args:
          msg: Incoming tool-use message.

        Returns:
          label: Human-readable summary with filename and range.

        """
        directive = get_directive(msg)
        file_path = str(directive.get("file_path", ""))
        fname = Path(file_path).name if file_path else "?"
        offset = int_val(directive.get("offset"), 0)
        limit = int_val(directive.get("limit"), 0)
        last_lines = int_val(directive.get("last_lines"), 0)
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

    def summary_result(self, result: Message) -> str | None:
        """One-line receipt summarizing the read.

        ``{N} lines`` for plain-text reads, ``image``/``pdf`` for
        binary formats, ``(unchanged)`` when the file matches the
        last-read snapshot, ``None`` for errors.
        """
        if result.descriptor == "text/x-error":
            return None
        if result.descriptor != "text/plain":
            # Multipart (image, PDF) or notebook -- the inner descriptor
            # carries the format. Fall back to a generic marker.
            return "binary"
        text = str(result.content)
        if text.startswith("[File unchanged"):
            return "unchanged"
        # Line-numbered output: count newlines in the rendered body.
        lines = text.count("\n")
        return f"{lines} lines"

    def prompt(self) -> str:
        """Return supplemental prompt text for this tool.

        Returns:
          prompt: Always empty.

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
    ) -> str | Message:
        p = Path(file_path)
        if not p.exists():
            return TextMessage(f"File not found: {file_path}", "text/x-error")
        if p.is_dir():
            return TextMessage(
                (
                    f"{file_path} is a directory, not a file."
                    " Use Glob to inspect directory contents."
                ),
                "text/x-error",
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
            return f"[File unchanged since last read: {file_path}]"

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

        Handles simple commands (via ``unwrap_cd_prefix``) and
        pipelines like ``cat FILE | head -N``.

        Args:
          trees: Parsed shell AST nodes.

        Returns:
          hint: Suggested Read invocation, or ``None`` if no match.

        """
        single = self._match_single(trees)
        if single is not None:
            return single
        return _match_pipeline_read(trees)

    def _match_single(self, trees: Sequence[Node]) -> str | None:
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


def _read_image(p: Path, *, file_path: str, suffix: str) -> Message:
    """Return an image file as a multipart message with JPEG/PNG bytes."""
    return MultipartMessage(
        (
            TextMessage(f"[image: {file_path}]", "text/plain"),
            BytesMessage(p.read_bytes(), cast("BytesDescriptor", _image_mime(suffix))),
        ),
        "multipart/mixed",
    )


def _read_notebook(p: Path, *, file_path: str) -> str | Message:
    """Parse a Jupyter notebook and return cell contents as text."""
    try:
        nb = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return f"[Invalid notebook JSON: {file_path}: {e}]"
    except UnicodeDecodeError as e:
        return f"[Non-UTF-8 notebook: {file_path}: {e}]"
    if not isinstance(nb, dict):
        return f"[Not a valid Jupyter notebook: {file_path}]"
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
    return "\n".join(parts) or "(empty notebook)"


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
) -> str | Message:
    """Read a text file with offset/limit/last_lines windowing."""
    with p.open("rb") as f:
        head = f.read(8192)
    if b"\x00" in head:
        size = p.stat().st_size
        return f"[Binary file: {file_path} ({size} bytes). Use Bash to inspect.]"
    try:
        mtime = p.stat().st_mtime
    except OSError:
        mtime = None
    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        size = p.stat().st_size
        return (
            f"[Non-UTF-8 file: {file_path} ({size} bytes)."
            " Use Bash with an explicit decoder to inspect.]"
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
        return f"[File exists but is empty: {file_path}]"
    return _window_text(
        text, file_path=file_path, offset=offset, limit=limit, last_lines=last_lines
    )


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
    # ``cat FILE`` - exactly one positional, no flags.
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
    return _MIME_BY_EXT.get(suffix, "application/octet-stream")


def _read_pdf(path: Path, pages: str) -> Message:
    """Rasterize (a range of) a PDF's pages to JPEG attachments."""
    if not is_pdf(path):
        return TextMessage(f"Not a PDF: {path} (missing %PDF- header)", "text/x-error")
    size = path.stat().st_size
    if size > MAX_PDF_BYTES:
        return TextMessage(
            (
                f"PDF too large: {size} bytes > {MAX_PDF_BYTES}."
                ' Use pages="N-M" to read a range.'
            ),
            "text/x-error",
        )

    resolved = _resolve_page_range(path, pages)
    if not isinstance(resolved, tuple):
        return resolved
    first, last = resolved

    try:
        page_paths = extract_pdf_pages(path, first=first, last=last)
    except PdfError as e:
        return TextMessage(f"PDF: {e}", "text/x-error")

    range_note = f" pages {first}-{last or 'end'}" if first is not None else ""
    return MultipartMessage(
        (
            TextMessage(
                f"[PDF: {path.name} ({len(page_paths)} page(s){range_note})]",
                "text/plain",
            ),
            *(BytesMessage(pp.read_bytes(), "image/jpeg") for pp in page_paths),
        ),
        "multipart/mixed",
    )


def _resolve_page_range(
    path: Path, pages: str
) -> tuple[int | None, int | None] | Message:
    """Validate page-range spec and return ``(first, last)`` or an error."""
    if pages:
        parsed = parse_page_range(pages)
        if parsed is None:
            return TextMessage(
                (
                    f"Invalid pages spec {pages!r}: "
                    'use "N" / "N-M" / "N-" (1-indexed, inclusive)'
                ),
                "text/x-error",
            )
        return parsed
    count = get_pdf_page_count(path)
    if count is not None and count > MAX_INLINE_PAGES:
        return TextMessage(
            (
                f"PDF has {count} pages (> {MAX_INLINE_PAGES})."
                ' Pass pages="1-N" to read a window.'
            ),
            "text/x-error",
        )
    return None, None
