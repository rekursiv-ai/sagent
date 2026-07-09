"""Functional library for scholarly-paper lookup over multiple backends.

The paper analogue of :mod:`sagent.lib.web.search`: sync, backend-agnostic
functions over Semantic Scholar, OpenAlex, SearXNG, and (internal build only)
Google Scholar. Per-source cross-process rate limiting is built in (each backend
paces through :func:`sagent.lib.ratelimit.cross_process_limiter`). A coroutine call
site lifts any function into a thread with ``asyncio.to_thread``.

Import each name from the submodule that defines it (this package's
``__init__`` re-exports nothing, per STYLE), exactly as ``sagent.lib.web`` is
used (``from sagent.lib.web.search import search``):

- :mod:`.search` -- ``search`` + ``SearchResult`` / ``Source``.
- :mod:`.details` -- ``metadata``, ``metadata_batch``, ``references``,
  ``citations``, ``cited_by`` + ``Listing``.
- :mod:`.authors` -- ``search_authors``, ``author_metadata``,
  ``author_papers`` + ``AuthorSearchResult``.
- :mod:`.fetch` -- ``download`` (PDF source cascade).
- :mod:`.custom_types` -- ``PaperRecord``, ``AuthorRecord``, ``IdType``.
- :mod:`.errors` -- ``PaperError`` and its subclasses.
- :mod:`.ids` -- ``normalize_id``, ``s2_wire_id``, ``id_slug``.

Usage::

    from sagent.lib.web.paper.search import search
    from sagent.lib.web.paper.details import metadata
    from sagent.lib.web.paper.ids import normalize_id

    hits = search("denoising recursion models", limit=10)
    for rec in hits.records:
        print(rec.title, rec.year)

    meta = metadata(*normalize_id("arXiv:1706.03762"))
"""

from __future__ import annotations
