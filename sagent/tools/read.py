"""Read tool: text, image, PDF, and notebook file reading."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final, cast

import asyncio
import json
import re

from sagent.agent.state import current_agent_var, get_tool_state
from sagent.lib.custom_json import (
    MutableJSON,
    MutableJSONValue,
    int_val,
    json_freeze,
)
from sagent.tools.core import (
    file_lock_key,
    load_tool_description,
    mark_read,
    resolve_tool_path,
)
from sagent.tools.display import Toggle, Wrap
from sagent.tools.lib.bash import (
    Invocation,
    Node,
    sed_mutates,
    walk_commands,
)
from sagent.tools.lib.pdf import (
    MAX_INLINE_PAGES,
    MAX_PDF_BYTES,
    MAX_RENDERED_BYTES,
    PdfError,
    extract_pdf_pages,
    get_pdf_page_count,
    is_pdf,
    parse_page_range,
)
from sagent.tools.tool_spec import CLI_SETTABLE
from sagent.types.runtime import BytesMessage, ToolResult


# Single source of truth for image handling: extension -> wire MIME. The set of
# image extensions is DERIVED from these keys (below) so the two can never drift
# -- adding a format here is the only edit needed.
# SVG is intentionally absent: it is XML, sent as text so providers do not
# reject ``image/svg+xml``.
_MIME_BY_EXT: Final[dict[str, str]] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
}
_IMAGE_EXTS = frozenset(_MIME_BY_EXT)
_PDF_EXT: Final = ".pdf"
_NOTEBOOK_EXT: Final = ".ipynb"
_CAT_SHAPERS: frozenset[str] = frozenset({"head", "tail", "less", "more"})


@dataclass(frozen=True, slots=True, kw_only=True)
class Read:
    """Read file contents: text, image, PDF, and notebook."""

    name = "Read"
    tool_id = "application/x-tool-read"
    clearable_results = True
    description = load_tool_description("Read")
    directive_schema = json_freeze(
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
                        "Max lines to return (text files only). Omit for a"
                        " budget-derived default; the reply names the offset"
                        " to resume from. Must be ≥ 1."
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
        limit = int_val(args.get("limit"), _default_line_limit())
        last_lines = int_val(args.get("last_lines"), 0)
        pages = str(args.get("pages", ""))
        # Schema declares ``offset``/``limit``/``last_lines`` as
        # ``minimum: 1`` integers but ``int_val`` accepts any int
        # (including 0 and negatives). Reject schema violations at the
        # entrypoint -- ``offset=0`` previously fell through to the
        # ``max(1, offset)`` clamp in ``_window_text`` which masked the
        # error and made ``offset=0`` silently mean ``offset=1``.
        bounds_err = _check_minimum(
            ("offset", offset, args.get("offset")),
            ("limit", limit, args.get("limit")),
            ("last_lines", last_lines, args.get("last_lines")),
        )
        if bounds_err is not None:
            return bounds_err
        return await asyncio.to_thread(
            self._run,
            file_path=file_path,
            offset=offset,
            limit=limit,
            last_lines=last_lines,
            pages=pages,
        )

    output: Annotated[Toggle, CLI_SETTABLE] = "off"
    """Whether the result body renders in the pane."""

    output_head_rows: Annotated[int, CLI_SETTABLE] = 2
    """Leading body rows kept."""

    output_tail_rows: Annotated[int, CLI_SETTABLE] = 2
    """Trailing body rows kept, after a ``⋯ N lines ⋯`` marker."""

    output_max_width: Annotated[int, CLI_SETTABLE] = 0
    """Cell width cap; ``0`` uses the pane width."""

    output_wrap: Annotated[Wrap, CLI_SETTABLE] = "wrap"
    """``wrap`` continues an over-wide line, ``chop`` marks the cut."""

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        """Serialize same-file Read/Edit/Write within a cohort.

        Ordering a Read against same-file Edit/Write in one cohort makes
        a batched Read observe the post-edit content deterministically,
        and keeps the read-cache stamp consistent with the final write.

        Args:
          args: Directive carrying ``file_path``.

        Returns:
          key: Canonical path, or ``None`` when no path was supplied.

        """
        path = resolve_tool_path(str(args.get("file_path", "")))
        return file_lock_key(path) if path else None

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
            # End line is inclusive: ``offset=10, limit=3`` covers lines
            # 10, 11, 12 -- the last line is ``offset + limit - 1``.
            suffix = f":{offset}-{offset + limit - 1}"
        elif offset > 0:
            suffix = f":{offset}+"
        elif limit > 0:
            suffix = f":1-{limit}"
        else:
            suffix = ""
        return f"Read {fname}{suffix}"

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
        limit: int = 0,  # 0 = read to EOF; callers pass _default_line_limit()
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
        """Emit a hint if any command reads a file the Read tool could.

        Policy only: :func:`walk_commands` supplies every simple command
        with its context, so a leading ``cd``, an enclosing loop, and a
        trailing ``| head`` all reach this matcher without it re-deriving
        AST shape -- the omission that left ``cd X && cat f | head``
        silent while ``cat f | head`` nudged.

        Args:
          trees: Parsed bashlex command trees from the active Bash call.

        Returns:
          hint: Nudge string redirecting to the Read tool, or ``None``.

        """
        for inv in walk_commands(trees):
            if inv.env_prefix or inv.captures_stdout:
                continue
            if _reads_a_file(inv):
                return f"{inv.exe} via Bash is a bad UX. Use the Read tool."
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
    # Stamp the mtime AFTER reading. A writer that lands between stat and read
    # would otherwise pair a pre-write mtime with post-write content, and the
    # next ``check_stale`` would see disk-mtime > cached-mtime, treat the entry
    # as fresh, and silently adopt the new content. Stamping after read means a
    # racing write bumps the mtime past what we cached, so staleness fires.
    try:
        mtime = p.stat().st_mtime
    except OSError:
        mtime = None
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


def _reads_a_file(inv: Invocation) -> bool:
    """Whether this invocation is a file read the Read tool replaces.

    A sink that TRANSFORMS (``sort``, ``awk``) is doing work Read cannot,
    so the shape is left alone; one that merely truncates or paginates is
    what Read's own windowing already does.
    """
    if any(d.captures_stdout for d in inv.downstream()):
        return False
    if inv.piped_into is not None and inv.piped_into.exe not in _CAT_SHAPERS:
        return False
    if inv.exe == "cat":
        return _cat_reads(inv.args)
    if inv.exe in ("head", "tail"):
        return _head_tail_reads(inv.args)
    if inv.exe == "sed":
        return _sed_reads(inv.args)
    return False


def _cat_reads(args: tuple[str, ...]) -> bool:
    """``cat FILE...`` with no flags and no glob.

    Multiple files count: the nudge is a suggestion to batch Read calls,
    which is exactly what several ``cat`` positionals are doing by hand.
    A GLOB does not: Read takes one ``file_path`` and cannot expand one,
    so pointing there sends the caller to a tool that cannot do the job.
    """
    if not args or any(a.startswith("-") for a in args):
        return False
    return not any(_has_glob(a) for a in args)


def _has_glob(arg: str) -> bool:
    """Whether a positional is a shell glob rather than a literal path."""
    return any(ch in arg for ch in "*?[")


def _head_tail_reads(args: tuple[str, ...]) -> bool:
    """``head``/``tail`` limited to line counts, not byte offsets.

    ``-c`` reads bytes, which Read's line windowing cannot express, so
    that shape is deliberately left alone.
    """
    positional: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith("--"):
            return False
        if a == "-n":
            if i + 1 >= len(args) or not args[i + 1].lstrip("+-").isdigit():
                return False
            i += 2
            continue
        if a.startswith("-") and a != "-":
            if not a[1:].isdigit():
                return False
            i += 1
            continue
        positional.append(a)
        i += 1
    return bool(positional) and not any(_has_glob(a) for a in positional)


def _sed_reads(args: tuple[str, ...]) -> bool:
    """``sed -n 'M,Np'`` -- the hand-rolled line window Read's args express.

    Gated hard on a bare print range: a substitution or an in-place edit
    is Edit's business, and silently nudging those toward Read would
    send the caller to a tool that cannot do the job.
    """
    # ``-in`` is quiet PLUS in-place: the file is rewritten, so this is
    # Edit's business. Share the predicate rather than re-deriving it --
    # matching only on "a short flag containing n" accepted ``-in`` and
    # advertised a destructive command as a Read.
    if sed_mutates(args):
        return False
    if not any(a.startswith("-") and not a.startswith("--") and "n" in a for a in args):
        return False
    scripts = [a for a in args if _LINE_RANGE_SCRIPT.fullmatch(a)]
    others = [a for a in args if not a.startswith("-") and a not in scripts]
    return len(scripts) == 1 and bool(others)


# ``sed -n`` scripts that only PRINT a line range: ``5p``, ``1,50p``,
# ``10,$p``. Anything else (``s/…/…/``, ``d``, ``w file``) transforms or
# writes, and belongs to Edit rather than Read.
_LINE_RANGE_SCRIPT: Final = re.compile(r"^\d+(,(\d+|\$))?p$")


def _image_mime(suffix: str) -> str:
    """MIME type for an image file suffix.

    The only caller gates on ``_IMAGE_EXTS``, which IS the key set of
    this table, so a miss cannot happen -- a default would be a branch
    for an impossible case.
    """
    return _MIME_BY_EXT[suffix]


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
        page_jpegs, total_pages = _render_pdf_jpegs(path, first=first, last=last)
    except PdfError as e:
        return ToolResult(call_id="", content=f"PDF: {e}", is_error=True)

    start = first if first is not None else 1
    rendered = len(page_jpegs)
    # The full count comes from the same render call -- no second open that
    # could transiently fail and silently mark a partial read as complete.
    # Clamp to ``total_pages``: ``extract_pdf_pages`` renders at most the real
    # pages (``hi = min(last, n_pages)``), so an over-range ``last`` (e.g.
    # ``pages="1-9999"`` on a short PDF) is a COMPLETE read, not a truncation.
    requested_last = min(last, total_pages) if last is not None else total_pages
    # ``extract_pdf_pages`` returns a prefix when the rendered-byte budget
    # truncated the read. Make the truncation VISIBLE so the model doesn't
    # mistake a partial read for a complete one, and name the resume range.
    truncated = start + rendered - 1 < requested_last
    if truncated:
        resume = start + rendered
        note = (
            f"[PDF: {path.name} -- rendered pages {start}-{start + rendered - 1} "
            f"of requested {start}-{requested_last}; output truncated at the "
            f'request byte budget. Pass pages="{resume}-{requested_last}" to read '
            f"the remaining pages.]"
        )
    else:
        # Report the clamped end, never the caller's raw over-range ``last``.
        range_note = f" pages {start}-{requested_last}" if first is not None else ""
        note = f"[PDF: {path.name} ({rendered} page(s){range_note})]"
    return ToolResult(
        call_id="",
        content=note,
        attachments=tuple(BytesMessage(jpeg, "image/jpeg") for jpeg in page_jpegs),
    )


def _render_pdf_jpegs(
    path: Path, *, first: int | None, last: int | None
) -> tuple[list[bytes], int]:
    """Rasterize the page range to JPEGs under the rendered-byte budget.

    Returns the rendered JPEGs and the PDF's full page count (for the
    continuation hint, without a second open).
    """
    return extract_pdf_pages(
        path, first=first, last=last, max_total_bytes=_rendered_byte_budget()
    )


# Line cap used when no agent is in context (standalone tool use, tests).
_FALLBACK_LINE_LIMIT: Final = 2_000

# Characters per line assumed when turning a character budget into a line
# count. Set above typical source-line width so the derived cap errs
# small: over-estimating width under-counts lines, which is the safe
# direction.
_ASSUMED_CHARS_PER_LINE: Final = 80


def _default_line_limit() -> int:
    """Line cap for a windowless Read, derived from the active budget.

    One result must stay under ``max_result_chars`` or it is off-loaded
    to disk (replaced by a ~2 KB preview) or, past the per-request
    budget, elided to a placeholder -- both of which hand back LESS of
    the file than a plain windowed read. Read is additionally exempt
    from disk offload, so without a cap here nothing bounds it at all.

    Deriving the cap from the agent keeps one read whole across models
    whose windows differ by two orders of magnitude, where any single
    constant is wrong at one end. Mirrors :func:`_rendered_byte_budget`,
    which sizes PDF rasterization from the same handle.

    Returns:
      limit: Maximum lines a default Read returns; the fallback constant
          when no agent is in context.

    """
    agent = current_agent_var.get(None)
    ceiling = agent.max_result_chars if agent is not None else 0
    if ceiling <= 0:
        return _FALLBACK_LINE_LIMIT
    return max(_FALLBACK_LINE_LIMIT, ceiling // _ASSUMED_CHARS_PER_LINE)


def _rendered_byte_budget() -> int:
    """Per-read cap on cumulative rendered JPEG bytes, from the active model.

    Provider request ceilings vary by orders of magnitude (a small local
    model may allow far less than a 32 MB Anthropic request), so the bound
    must follow the ACTIVE model's ``max_request_bytes``, not a single
    constant. The rendered JPEGs are raw bytes that ship base64-expanded
    (``4/3``), and the request also carries system prompt + history + text,
    so reserve headroom: budget the raw render at half the ceiling's
    base64-deflated size. Falls back to ``MAX_RENDERED_BYTES`` when no agent
    is in context (standalone tool use / tests) or the model declares no
    ceiling (``<= 0``).
    """
    agent = current_agent_var.get(None)
    ceiling = agent.max_request_bytes if agent is not None else 0
    if ceiling <= 0:
        return MAX_RENDERED_BYTES
    # Half the ceiling (headroom for system/history/text), then deflate by
    # the base64 4/3 expansion to bound the RAW rendered bytes.
    return max(1, (ceiling // 2) * 3 // 4)


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


def _check_minimum(
    *fields: tuple[str, int, object],
) -> ToolResult | None:
    """Reject schema-violating windowing args at the tool entrypoint.

    Each tuple is ``(name, coerced, raw)``: ``coerced`` is the
    ``int_val`` result we'd otherwise pass downstream; ``raw`` is the
    untouched directive value used to detect "the caller supplied it"
    (an absent key has ``raw is None`` and is allowed to fall through
    to the default).
    """
    # Defense-in-depth: ``validate_tool_input`` (the JSON-schema gate run by
    # ``_AgentTool.run``) is the primary enforcer of these minima; this re-check
    # covers direct ``._run()`` callers (tests, internal reuse) that bypass it.
    for name, coerced, raw in fields:
        if raw is None:
            continue
        if coerced < 1:
            return ToolResult(
                call_id="",
                content=f"'{name}' must be ≥ 1, got {coerced}.",
                is_error=True,
            )
    return None
