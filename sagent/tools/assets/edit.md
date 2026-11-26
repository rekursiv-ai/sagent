Substitute exact text in a file.

- Prefer Edit over Write for modifying existing files.
- Read once before editing a file. Subsequent Edits don't need a re-Read.
- `old_string` matches verbatim including indentation. Read output is `<lineno>\t<content>` -- use content only.
- Non-unique match fails; add context or set `replace_all`.
- `replace_all` substitutes every occurrence -- useful for renaming a variable.
