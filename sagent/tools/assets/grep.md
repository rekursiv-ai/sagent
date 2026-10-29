Ripgrep-backed content search. Never shell out to `grep`/`rg`.

Required: `pattern` (string). Optional: `path` (string, absolute or cwd-relative; defaults to session cwd). All other args are keyword fields, never positional, never English (no `Grep "pat" in /dir`).

- Full Rust regex: `"log.*Error"`, `"function\s+\w+"`. Escape literal braces: `interface\{\}`.
- Narrow with `glob` or `type`; exclude with `exclude`.
- `output_mode`: `files_with_matches` (default), `content`, `count`.
- `multiline: true` -- pattern may match across newlines (literal `\n`, or `.` spanning lines). Without it, `\n` in the pattern errors.
- `pcre: true` -- PCRE2 (lookaround, backrefs).
- Multi-round exploration: delegate to subagent.
