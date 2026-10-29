Fetch a URL. GET (default) or POST.

- GET -- main content as markdown, read-only intent. Cached 15 min.
- POST -- `json` or `form` (exactly one); response verbatim, truncated. Never cached. Side effects possible -- use only when needed.
- GET/POST only; PUT/PATCH/DELETE rejected.
- Every parameter is described in this tool's schema, generated from one spec shared with the MCP surface. What the schema cannot say:
- Reach for `extractor: trafilatura` only when you already know the page is ONE contiguous prose body: an encyclopedia article, a PEP/RFC/spec, a long-form post. There it is both smaller and lossless.
- Everywhere else it silently drops content, and no length or structure check predicts it. Measured on an 11-page corpus by wesearch's `scripts/compare_extractors.py`: `html2text` loses 0 of 37 content probes, `trafilatura` loses 12 -- a Q&A thread keeps one answer of dozens, a profile timeline returns 864 of 4,530 chars.
- URLs fully qualified; `http://` upgraded to HTTPS.
- No custom headers (`Authorization`, `Cookie`).
- Cross-host redirects reported, not followed -- re-issue.
- GitHub resources -- prefer `gh` via Bash.
