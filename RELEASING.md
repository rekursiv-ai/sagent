Release is fully automated via GitHub Releases. Publishing triggers on `release: published` (`publish-pypi.yml:4-5`), builds with uv build, and pushes to PyPI via trusted publishing (`id-token: write`, `environment: pypi`, no API token).

## Steps

1. Bump the version in `pyproject.toml`. PyPI rejects re-uploads, so this must increase.
2. Commit and merge to main.
3. Validate locally (same checks CI runs, CONTRIBUTING.md:18-30):
   ```sh
   uv build
   uv run python sagent/bin/check_wheel.py
   ```
4. Create a GitHub Release with a new tag — this is the trigger:
   ```sh
   gh release create vX.Y.Z --title vX.Y.Z --notes ""
   ```

Publishing the release fires the workflow, which builds and uploads to PyPI automatically.


## Notes

- No manual `twine upload` / API token needed -- uses OIDC trusted publishing.
- `workflow_dispatch` (`publish-pypi.yml:6`) also lets you run it manually from the Actions tab without a release, if needed.
- The pypi environment may have required reviewers/protection rules gating the publish step -- check repo Settings → Environments if it stalls.
