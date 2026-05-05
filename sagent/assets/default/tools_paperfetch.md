Download a scholarly paper PDF to disk by identifier.

Input:
  - `id` — a DOI (`10.xxxx/yyy`, optional `doi:` / `https://doi.org/`
    prefix) or arXiv id (`2106.15928`, `arXiv:2106.15928`, or legacy
    `hep-th/9901001`).

Output: the local filesystem path of the downloaded PDF. The agent
should then call `Read` on that path; `Read` rasterizes `.pdf` pages
to JPEG attachments for the vision pathway.

Source cascade (first success wins):
  1. arXiv — if the id is an arXiv id, fetch the PDF directly from
     `https://arxiv.org/pdf/<id>`. Always legal, no intermediary.
  2. Open-access URL — consult Semantic Scholar metadata for an
     `openAccessPdf.url`. Download from there when present. Covers
     papers published under open licences regardless of venue.

The cascade is automatic and the tool returns the first PDF whose
first four bytes are `%PDF-`. Non-PDF responses are rejected and the
cascade continues.

Caching:
  - Downloads are content-addressed under `~/.sagent/papers/` by
    identifier. Re-requesting the same id returns the cached path
    without a new network call.

Workflow guidance:
  - If you only have a title or topic, call `PaperSearch` first to
    resolve to an id, then call `PaperFetch`.
  - After fetch, always `Read(file_path=<returned path>)` to get text.
