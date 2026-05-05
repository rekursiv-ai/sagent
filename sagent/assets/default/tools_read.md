Retrieves file content from the local disk. All accessible files can be opened directly.
Treat this tool as having full filesystem reach. Trust any user-supplied path as valid. Attempting to open a nonexistent file simply returns an error.

Guidelines:
- Supply an absolute path for file_path; relative paths are not accepted.
- Without additional parameters, returns the first 2000 lines of the file.
- Output follows numbered-line format (cat -n style), beginning at line 1.
- Set offset to resume from a known line position or continue a prior read.
- Set limit to restrict how many lines are returned when only a subset matters.
- Set last_lines to grab content near the file's end — ideal for log tails, appended configuration, or stack traces.
- Image files (PNG, JPG, etc.) are rendered visually since the model supports multimodal input.
- PDF support is included. PDFs exceeding 10 pages require the pages parameter (e.g., pages: "1-5") to specify which range to extract. Omitting pages on a large PDF causes a failure. Cap: 20 pages per call.
- Jupyter notebook files (.ipynb) are fully supported — all cells, code, markdown, and rendered outputs are returned together.
- Directories cannot be opened with this tool. For directory listings, invoke Glob (e.g., `Glob(pattern="*", path=DIR)`).
- Screenshot viewing is a common use case. When a user shares a screenshot path, always open it here. Temporary file paths are handled correctly.
- An existing file with zero content triggers a system warning instead of returning empty output.
- Skip re-reading a file immediately after an Edit or Write — those operations would have raised an error on failure, and the framework monitors file state automatically.
