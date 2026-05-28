Read a file. Full FS reach.

- `file_path` absolute; relative paths resolve against the session cwd. Default: first 2000 lines, `cat -n` numbered.
- `offset`/`limit` for ranges; `last_lines` tails the file (ignores `offset`, composes with `limit`).
- Images render visually. PDFs >10 pages require `pages` (cap 20/call). `.ipynb` returns cells + outputs.
- Directories: use `List`/`Glob`. Skip re-reads after Edit/Write -- staleness is tracked.
