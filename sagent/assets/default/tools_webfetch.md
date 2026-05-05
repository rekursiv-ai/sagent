- Retrieves a URL via HTTP. Supports GET (default) and POST.
- For GET, extracts the main page content as clean markdown text.
- For POST, sends an optional JSON or form body and returns the response verbatim (truncated).
- Returns the page text or response body as a single text result.

Usage notes:
  - URLs must be fully qualified (scheme + host + path)
  - Plain HTTP is silently upgraded to HTTPS
  - For POST: pass `method: "POST"` plus exactly one of `json` (JSON body) or `form` (form-urlencoded fields)
  - Only GET and POST are supported; PUT/PATCH/DELETE are rejected
  - Custom request headers are not accepted (no Authorization, Cookie, etc.)
  - Read-only intent for GET; POST may have side effects on the target server -- use only when needed
  - Large pages are truncated before return
  - GET responses are cached for 15 minutes; POST responses are never cached
  - Cross-host redirects are reported rather than followed automatically -- re-issue the request with the redirect target
  - For GitHub resources, prefer the `gh` CLI via Bash (e.g., `gh pr view`, `gh issue view`, `gh api`).
