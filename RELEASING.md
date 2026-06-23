# Releasing sagent to PyPI

Maintainer-only.

Publishing is fully automated: a published GitHub Release on
`rekursiv-ai/sagent` triggers `.github/workflows/publish-pypi.yml`
(`release: published`, `publish-pypi.yml:4-5`), which builds with
`uv build`, validates, and uploads to PyPI via OIDC trusted publishing
(`id-token: write`, `environment: pypi`, no API token).

## Steps

Replace `X.Y.Z` with the new version.

1. Bump the version in `pyproject.toml` (source of truth). PyPI rejects
   re-uploads, so this must increase.
2. Validate locally (same checks CI runs, CONTRIBUTING.md:18-30):
   ```bash
   uv build
   uv run python sagent/bin/check_wheel.py
   ```
3. Commit and merge to `main`. Confirm the published `pyproject.toml`
   shows the new version before continuing:
   ```bash
   gh api repos/rekursiv-ai/sagent/contents/pyproject.toml --jq .content | base64 -d | grep '^version'
   ```
4. Cut the release (this is what actually triggers PyPI publish):
   ```bash
   gh release create vX.Y.Z --repo rekursiv-ai/sagent --title "sagent X.Y.Z" --generate-notes
   ```
5. Watch the workflow:
   ```bash
   gh run watch --repo rekursiv-ai/sagent $(gh run list --repo rekursiv-ai/sagent --workflow publish-pypi.yml --limit 1 --json databaseId --jq '.[0].databaseId')
   ```

## Check what's live on PyPI

```bash
# Latest version
curl -s https://pypi.org/pypi/sagent/json | jq -r '.info.version'

# All published versions
curl -s https://pypi.org/pypi/sagent/json | jq -r '.releases | keys[]'

# Or via pip
pip index versions sagent
```

Browser: https://pypi.org/project/sagent/

## Manual re-publish

If a release exists but the workflow needs to re-run (e.g. transient
PyPI failure), trigger it without cutting a new tag via
`workflow_dispatch` (`publish-pypi.yml:6`):

```bash
gh workflow run publish-pypi.yml --repo rekursiv-ai/sagent
```

## Notes

- Bumping the version alone does not deploy. The release event is the
  trigger.
- PyPI does not allow overwriting a published version. To re-upload,
  bump to `X.Y.Z+1`.
- The `pypi` GitHub environment may have required reviewers/protection
  rules gating the publish step -- check repo Settings -> Environments
  if it stalls.
- Trusted publishing is configured on the `pypi` GitHub environment in
  `rekursiv-ai/sagent`; rotate via PyPI project settings, not here.
