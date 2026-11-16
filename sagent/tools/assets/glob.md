Filename pattern matcher.

- Syntax: `"**/*.js"`, `"src/**/*.ts"`.
- `path` defaults to the session cwd; pass absolute to scope elsewhere.
- `sort`: `none` (default: as found, each path as soon as found), `name`, `name_desc`, `mtime`, `mtime_desc`, `size`, `size_desc`.
- `long=true` adds size + mtime.
- Multi-round explore-and-grep: delegate to subagent.
