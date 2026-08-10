"""Tool-private utilities.

Modules here serve the ``tools/`` package and have no dependents
outside of it (apart from ``agent/dispatch.py`` which uses the bash
parser for concurrency classification).
"""

from sagent.tools.lib.bash import (
    Command,
    cached_parse_bash,
    is_read_only,
    parse_bash,
    resolve_cwd_path,
    sed_mutates,
    walk_commands,
)
from sagent.tools.lib.pdf import (
    MAX_INLINE_PAGES,
    MAX_PDF_BYTES,
    PDF_MAGIC,
    PdfError,
    extract_pdf_pages,
    get_pdf_page_count,
    is_pdf,
    parse_page_range,
)


__all__ = [
    "MAX_INLINE_PAGES",
    "MAX_PDF_BYTES",
    "PDF_MAGIC",
    "Command",
    "PdfError",
    "cached_parse_bash",
    "extract_pdf_pages",
    "get_pdf_page_count",
    "is_pdf",
    "is_read_only",
    "parse_bash",
    "parse_page_range",
    "resolve_cwd_path",
    "sed_mutates",
    "walk_commands",
]
