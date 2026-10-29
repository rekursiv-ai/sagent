List a directory (`ls`). Dirs suffixed `/`.

- Use this, not Bash `ls`. Tree-wide patterns: `Glob`.
- `path` absolute; relative paths resolve against the session cwd.
- `show_hidden` (default false), `long=true` (size + mtime).
- `sort`: `name` (default), `name_desc`, `mtime`, `mtime_desc`, `size`, `size_desc`.
