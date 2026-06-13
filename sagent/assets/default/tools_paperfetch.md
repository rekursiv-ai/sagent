Download scholarly paper PDFs to disk by identifier.

Input:
  - `ids` — one identifier as a bare string, or several as an array.
    Each is a DOI
    (`10.xxxx/yyy`, optional `doi:` / `https://doi.org/` prefix) or an
    arXiv id (`2106.15928`, `arXiv:2106.15928`, or legacy
    `hep-th/9901001`). Pass every paper you need at once: the
    open-access URL lookups are resolved in ONE batched Semantic Scholar
    request (up to 500) rather than one per id — far more efficient
    against the 1 request/second rate limit — then the PDFs download
    concurrently.

Output: the local filesystem path of each downloaded PDF, one line per
id. The agent should then `Read` each path; `Read` rasterizes `.pdf`
pages to JPEG attachments for the vision pathway.

Source cascade (first success wins), per id:
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
    resolve to identifiers, then call `PaperFetch`.
  - After fetch, always `Read(file_path=<returned path>)` to get text.
