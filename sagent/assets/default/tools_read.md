Read a file. Full FS reach.

- `file_path` absolute; relative paths resolve against the session cwd. Default: first 2000 lines, line-numbered (right-aligned number + tab + content, like `cat -n`).
- `offset`/`limit` for ranges; `last_lines` tails the file (ignores `offset`, composes with `limit`).
- Multimodal: image files (PNG/JPG/etc) render visually as attachments. PDFs >10 pages require `pages` (e.g. `"1-5"`). `.ipynb` returns cells + outputs.
- Empty files return a system reminder, not blank content.
- Directories: use `List`/`Glob`.
