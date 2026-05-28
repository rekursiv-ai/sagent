Filename pattern matcher.

- Syntax: `"**/*.js"`, `"src/**/*.ts"`.
- `sort`: `name` (default), `name_desc`, `mtime`, `mtime_desc`, `size`, `size_desc`.
- `long=true` adds size + mtime.
- Multi-round explore-and-grep: delegate to subagent.
