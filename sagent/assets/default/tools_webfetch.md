- Retrieves a URL, converts the HTML to markdown, and summarizes it via a lightweight model
- Accepts a URL and a natural-language prompt describing what to extract
- Returns the model's distilled answer about the page content
- Useful whenever you need to pull and analyze live web content

Usage notes:
  - URLs must be fully qualified (scheme + host + path)
  - Plain HTTP is silently upgraded to HTTPS
  - The prompt should specify what information you want from the page
  - Read-only — no files are created or modified
  - Very large pages may be condensed before analysis
  - Responses are cached for 15 minutes; repeated fetches of the same URL reuse the cache
  - Cross-host redirects are reported rather than followed automatically — re-issue the request with the redirect target
  - For GitHub resources, prefer the `gh` CLI via Bash (e.g., `gh pr view`, `gh issue view`, `gh api`).
