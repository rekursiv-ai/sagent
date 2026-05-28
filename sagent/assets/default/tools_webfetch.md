Fetch a URL. GET (default) or POST.

- GET -- main content as markdown, read-only intent. Cached 15 min.
- POST -- `json` or `form` (exactly one); response verbatim, truncated. Never cached. Side effects possible -- use only when needed.
- GET/POST only; PUT/PATCH/DELETE rejected.
- URLs fully qualified; `http://` upgraded to HTTPS.
- No custom headers (`Authorization`, `Cookie`).
- Cross-host redirects reported, not followed -- re-issue.
- GitHub resources -- prefer `gh` via Bash.
