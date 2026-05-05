Applies precise text substitutions within a file.

Guidelines:
- A prior `Read` call on the target file is mandatory within this conversation. An error is raised if no read has occurred.
- When referencing content from Read output, replicate the original indentation (tabs or spaces) exactly as shown after the line-number prefix. The prefix consists of a line number followed by a tab character — everything beyond that tab is actual file content. Do not incorporate any portion of the line-number prefix into old_string or new_string.
- Default to modifying files in place. Avoid generating new files unless the task strictly demands it.
- Emoji characters belong in output only when the user has specifically asked for them. Omit emoji from file content by default.
- A non-unique `old_string` causes the operation to fail. Resolve this by expanding the match text with additional surrounding context for uniqueness, or set `replace_all` to substitute every occurrence throughout the file.
- The `replace_all` flag is designed for bulk renaming — for example, changing a variable name across an entire file.
