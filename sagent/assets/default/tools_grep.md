Ripgrep-backed content search across files.

  Guidelines:
  - This is the designated search tool. Never shell out to `grep` or `rg` via Bash. This wrapper handles permissions and access correctly.
  - Files with permission errors are silently skipped — no stderr redirection needed.
  - Full regular expression syntax is available (e.g., "log.*Error", "function\s+\w+").
  - Narrow results by file name via the glob parameter (e.g., "*.js", "**/*.tsx") or by language via the type parameter (e.g., "js", "py", "rust").
  - Three output modes exist: "content" displays matched lines, "files_with_matches" returns only paths (the default), "count" reports hit totals.
  - For exploratory searches that may need several iterations, delegate to the Agent tool.
  - Regex flavor is ripgrep's Rust engine — literal braces must be escaped (e.g., `interface\{\}` to locate `interface{}` in Go).
  - Patterns are single-line by default. To span multiple lines (e.g., `struct \{[\s\S]*?field`), enable `multiline: true`.
  - The `exclude` parameter accepts a glob for omitting files (e.g., `exclude="*.test.py"`). It composes naturally with `glob` for combined include/exclude filtering.
  - Setting `pcre: true` activates PCRE2 features such as lookaround and backreferences. The default Rust regex engine omits these.
