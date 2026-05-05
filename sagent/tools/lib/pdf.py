"""PDF helpers for the Read tool.

Rasterizes PDF pages to JPEGs via poppler's ``pdftoppm`` so tool
results can be returned as image attachments the model's vision
path actually consumes. ``pdfinfo`` provides page counts.

Page-range syntax:

- ``"5"``    → page 5
- ``"1-10"`` → pages 1 through 10 inclusive
- ``"3-"``   → page 3 to end

1-indexed, inclusive. Ranges are handed directly to pdftoppm's
``-f`` / ``-l`` flags.
"""

from __future__ import annotations

from pathlib import Path

import atexit
import logging
import shutil
import subprocess
import tempfile
import threading
import uuid


logger = logging.getLogger(__name__)


# Raw PDF size at which we refuse to rasterize (protects against
# an unbounded pdftoppm run on a multi-GB dump).
MAX_PDF_BYTES = 100 * 1024 * 1024

# Refuse to inline a full PDF without a ``pages`` range when it
# exceeds this page count. Forces the caller to pick a window
# rather than blowing the context budget.
MAX_INLINE_PAGES = 20

PDF_MAGIC = b"%PDF-"


class PdfError(Exception):
    """Raised when PDF inspection / rasterization fails."""


def is_pdf(path: Path) -> bool:
    """Check whether the file starts with the PDF magic bytes.

    Args:
      path: File path to inspect.

    Returns:
      is_pdf: True if the file begins with ``%PDF-``.

    """
    try:
        with path.open("rb") as f:
            return f.read(len(PDF_MAGIC)) == PDF_MAGIC
    except OSError:
        return False


def get_pdf_page_count(path: Path) -> int | None:
    """Return page count via ``pdfinfo``, or ``None`` if unavailable.

    Args:
      path: PDF file path.

    Returns:
      count: Number of pages, or None if pdfinfo is missing or fails.

    """
    pdfinfo = shutil.which("pdfinfo")
    if not pdfinfo:
        return None
    try:
        result = subprocess.run(  # noqa: S603 -- trusted argv
            [pdfinfo, str(path)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def parse_page_range(spec: str) -> tuple[int, int | None] | None:
    """Parse ``"5"`` / ``"1-10"`` / ``"3-"`` into ``(first, last)``.

    ``last=None`` means open-ended (to end of document). Returns
    ``None`` for invalid / empty input. 1-indexed, inclusive.

    Args:
      spec: Page range string (e.g. ``"5"``, ``"1-10"``, ``"3-"``).

    Returns:
      range: ``(first, last)`` tuple, or None for invalid input.

    """
    s = spec.strip()
    if not s:
        return None
    if s.endswith("-"):
        try:
            first = int(s[:-1])
        except ValueError:
            return None
        return (first, None) if first >= 1 else None
    if "-" in s:
        a, b = s.split("-", 1)
        try:
            first, last = int(a), int(b)
        except ValueError:
            return None
        if first < 1 or last < first:
            return None
        return (first, last)
    try:
        page = int(s)
    except ValueError:
        return None
    return (page, page) if page >= 1 else None


def extract_pdf_pages(
    path: Path,
    *,
    first: int | None = None,
    last: int | None = None,
    dpi: int = 100,
) -> list[Path]:
    """Rasterize ``path`` to JPEGs via ``pdftoppm``.

    Writes ``page-NN.jpg`` into a fresh temp dir and returns the
    sorted list of paths. ``first`` / ``last`` are 1-indexed,
    inclusive; ``None`` means "no bound on that side".

    Args:
      path: PDF file path.
      first: First page to rasterize (1-indexed).
      last: Last page to rasterize (1-indexed).
      dpi: Resolution for rasterization.

    Returns:
      pages: Sorted list of JPEG file paths.

    Raises:
      PdfError: If pdftoppm is unavailable or fails.

    """
    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm:
        raise PdfError(
            "pdftoppm not installed. Install poppler-utils "
            "(`apt-get install poppler-utils` or `brew install poppler`).",
        )
    out_dir = _workdir() / uuid.uuid4().hex
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "page"
    args = [pdftoppm, "-jpeg", "-r", str(dpi)]
    if first is not None:
        args.extend(["-f", str(first)])
    if last is not None:
        args.extend(["-l", str(last)])
    args.extend([str(path), str(prefix)])
    try:
        result = subprocess.run(  # noqa: S603 -- trusted argv
            args,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise PdfError(f"pdftoppm timed out on {path}") from e
    if result.returncode != 0:
        stderr = result.stderr.lower()
        if "password" in stderr:
            raise PdfError(f"PDF is password-protected: {path}")
        if any(k in stderr for k in ("damaged", "corrupt", "invalid")):
            raise PdfError(f"PDF is corrupt or invalid: {path}")
        raise PdfError(f"pdftoppm failed: {result.stderr.strip()}")
    pages = sorted(out_dir.glob("page*.jpg"))
    if not pages:
        raise PdfError(f"pdftoppm produced no pages for {path}")
    return pages


# Process-lifetime workdir for rasterized PDF pages. Each call to
# ``extract_pdf_pages`` creates a UUID subdir under this root so
# concurrent reads don't collide; the root is rmtree'd at
# interpreter shutdown. Bounds disk usage at session lifetime
# rather than letting ``/tmp`` grow forever.
_workdir_root: Path | None = None
_workdir_lock = threading.Lock()


def _workdir() -> Path:
    """Return (lazily creating) the process-lifetime PDF workdir.

    Thread-safe: ``extract_pdf_pages`` runs inside ``asyncio.to_thread``
    so two concurrent PDF reads at startup would otherwise race
    ``tempfile.mkdtemp`` and leak one of the two directories (the
    unwinner's ``atexit`` handler still fires, but its path is
    unreachable during the session).
    """
    global _workdir_root  # noqa: PLW0603 -- one-shot lazy init for an atexit-cleaned root
    if _workdir_root is not None:
        return _workdir_root
    with _workdir_lock:
        if _workdir_root is None:
            _workdir_root = Path(tempfile.mkdtemp(prefix="sagent-pdf-"))
            atexit.register(shutil.rmtree, _workdir_root, ignore_errors=True)
        return _workdir_root
