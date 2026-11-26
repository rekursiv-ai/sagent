# Releasing sagent to PyPI

Maintainer-only. Replace `X.Y.Z` with the new version throughout.

A published GitHub Release triggers `.github/workflows/publish-pypi.yml`,
which builds, validates, and uploads to PyPI via OIDC trusted publishing
(no API token). Follow the steps in order.

## Steps

1. **(First release only) Register the trusted publisher on PyPI.**
   Skip if already done. Web-console only, no CLI. Go to
   https://pypi.org/manage/project/sagent/settings/publishing/ and add
   a publisher with these exact values:

   | Field | Value |
   |---|---|
   | Owner | `rekursiv-ai` |
   | Repository name | `sagent` |
   | Workflow filename | `publish-pypi.yml` |
   | Environment name | `pypi` |

   Environment name **must** be `pypi` (matches `publish-pypi.yml:16`).
   Blank here = `invalid-publisher` at publish time.

2. **Pick the version.** The release tag is the version (hatch-vcs);
   nothing in the tree states it. Must increase; PyPI rejects re-uploads.
   Release notes are generated from the commits since the last tag.

3. **Cut the release** from a `main` whose `package-validation` passed
   (this triggers the PyPI publish):
   ```bash
   gh release create vX.Y.Z --repo rekursiv-ai/sagent --title "vX.Y.Z" --generate-notes
   ```

4. **Watch the workflow:**
   ```bash
   gh run watch --repo rekursiv-ai/sagent $(gh run list --repo rekursiv-ai/sagent --workflow publish-pypi.yml --limit 1 --json databaseId --jq '.[0].databaseId')
   ```

5. **Confirm it's live:**
   ```bash
   curl -s https://pypi.org/pypi/sagent/json | jq -r '.info.version'   # should print X.Y.Z
   ```
   Browser: https://pypi.org/project/sagent/

## If the workflow fails

- **`invalid-publisher`** ("valid token, but no corresponding
  publisher") — the trusted publisher isn't registered or doesn't match.
  Do step 1, then re-run on the tag:
  `gh workflow run publish-pypi.yml --repo rekursiv-ai/sagent --ref vX.Y.Z`.

- **Transient PyPI failure** — re-run on the tag, not `main` (which builds a
  dev version): `gh workflow run publish-pypi.yml --repo rekursiv-ai/sagent --ref vX.Y.Z`.

- **Must ship now, publishing still broken** — upload directly with the
  project's API token (`rekursiv-ai`-owned, e.g. `$PYPI_TOKEN_WORK`; a
  personal token can't publish `sagent`):
  ```bash
  git checkout vX.Y.Z
  uv build --out-dir dist
  uv run --with twine twine check dist/sagent-X.Y.Z*
  uv run --with twine twine upload dist/sagent-X.Y.Z* -u __token__ -p "$PYPI_TOKEN_WORK"
  ```
  Permanent and public; PyPI never allows re-uploading a version.
