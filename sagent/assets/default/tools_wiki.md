Locate and query an initialized llm-wiki.

Use this only for a personal wiki with `SCHEMA.md`. Do not use for ordinary repo docs, code search, planning, or broad discovery. If locate fails once, do not retry unless the user creates a wiki or points you to one.

Operations:
- `locate` — return the absolute wiki root found by walking up from `cwd` or the current tool cwd.
- `list` — list all page slugs under `<root>/pages/`.
- `read_page` — read one page by `slug`.
- `read_index` — read `<root>/index.md`.
- `lint` — report deterministic broken wikilinks and missing frontmatter.

For ingesting, updating, querying, or linting workflows, invoke the corresponding wiki skill first; this tool only provides structural primitives.
