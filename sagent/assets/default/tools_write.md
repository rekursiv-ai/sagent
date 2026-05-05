Overwrites or creates a file on disk with the specified content.

Guidelines:
- Any pre-existing file at the target path will be replaced entirely. For files that already exist, you MUST first inspect them via Read. The operation aborts if the file was not previously read.
- For partial modifications to existing files, prefer Edit (it transmits only the delta). Reserve this tool for brand-new files or full-content replacements.
- Do NOT generate markdown docs (*.md) or README files without an explicit user request.
- Emoji characters belong in output only when the user has specifically asked for them. Omit emoji from file content by default.
