Fetch a URL. GET (default) or POST.

- GET -- main content as markdown, read-only intent. Cached 15 min.
- POST -- `json` or `form` (exactly one); response verbatim, truncated. Never cached. Side effects possible -- use only when needed.
- GET/POST only; PUT/PATCH/DELETE rejected.
- `transport` selects retrieval: `auto` (default), `curl`, `curl-then-zendriver`, `zendriver`, or `stdlib`. `auto` tries curl and escalates to Zendriver when a site bot-blocks it, routing straight to Zendriver for domains already learned to require it. Set it explicitly to stress a path or isolate transport failures.
- `extractor` selects how the page becomes text: `html2text` (default) renders every text node as markdown; `markdownify` converts the document's elements instead, keeping nested lists and tables that a text walk flattens; `trafilatura` returns only what it scores as the article -- smaller, but it drops the substance of any page that is not article-shaped (a dictionary entry loses its pronunciation, a Q&A thread loses every answer), and the loss is invisible because the output still looks complete; `raw` returns the HTML source.
- URLs fully qualified; `http://` upgraded to HTTPS.
- No custom headers (`Authorization`, `Cookie`).
- Cross-host redirects reported, not followed -- re-issue.
- GitHub resources -- prefer `gh` via Bash.
