Ripgrep-backed content search. Never shell out to `grep`/`rg`.

- Full Rust regex: `"log.*Error"`, `"function\s+\w+"`. Escape literal braces: `interface\{\}`.
- `path` defaults to the session cwd; pass absolute to scope elsewhere.
- Narrow with `glob` or `type`; exclude with `exclude`.
- `output_mode`: `files_with_matches` (default), `content`, `count`.
- `multiline: true` -- `.` crosses newlines.
- `pcre: true` -- PCRE2 (lookaround, backrefs).
- Multi-round exploration: delegate to subagent.
