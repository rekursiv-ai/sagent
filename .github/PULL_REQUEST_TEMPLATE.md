<!-- This template keeps public contributions reviewable and prevents accidental leaks of credentials, generated files, or private environment state. -->

## Summary


## Testing

- [ ] `uv run pytest`
- [ ] `uv run ruff check --no-fix --no-cache .`
- [ ] `uv run ruff format --check --no-cache .`
- [ ] `uv run ty check`
- [ ] `uv run basedpyright sagent`

## Checklist

- [ ] I updated docs for public behavior changes.
- [ ] I did not include secrets, credentials, generated caches, or local environment files.
- [ ] I kept the change focused and reviewable.
