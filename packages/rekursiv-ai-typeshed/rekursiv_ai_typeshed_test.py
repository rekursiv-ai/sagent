"""The installed trees carry exactly the tracked patch and match the checkers."""

from pathlib import Path
from typing import Final

import re
import sys

import pytest
import rekursiv_ai_typeshed


_CWD: Final = Path(__file__).resolve().parent
_INSTALLED = Path(sys.prefix) / "share" / "rekursiv-ai-typeshed"
_TYPINGS = _CWD / "rekursiv_ai_typeshed" / "typings"
_PATCH = _CWD / "rekursiv_ai_typeshed" / "typeshed.patch"

_POW_ANY = re.compile(r"def __r?pow__\(.*\) -> Any:")


def test_patch_is_only_the_pow_lines() -> None:
    hunks = [line for line in _PATCH.read_text().splitlines() if line[:1] in "+-"]
    changed = [h for h in hunks if not h.startswith(("---", "+++"))]
    removed = [h[1:] for h in changed if h.startswith("-")]
    added = [h[1:] for h in changed if h.startswith("+")]
    assert len(removed) == len(added) == 4
    for before, after in zip(removed, added, strict=True):
        assert _POW_ANY.search(before), before
        assert after == before.replace("-> Any:", "-> float:"), (before, after)


@pytest.mark.parametrize("checker", ["ty", "basedpyright"])
def test_installed_numeric_pow_returns_float(checker: str) -> None:
    klass = ""
    offenders: list[str] = []
    text = (_INSTALLED / checker / "stdlib" / "builtins.pyi").read_text()
    for line in text.splitlines():
        if line.startswith("class "):
            klass = line.split()[1].rstrip(":").split("(")[0]
        if klass in {"int", "float"} and _POW_ANY.search(line):
            offenders.append(f"{klass}: {line.strip()}")
    assert not offenders, offenders


def test_installed_tree_matches_the_installed_checkers() -> None:
    """A checker bump without a package rebuild leaves the tree stale."""
    bundle_commit = (
        (rekursiv_ai_typeshed.bundle_dir() / "commit.txt").read_text().strip()
    )
    bpr = _INSTALLED / "basedpyright"
    assert (bpr / "stubs.commit.txt").read_text().strip() == bundle_commit
    for checker in ("ty", "basedpyright"):
        stdlib = _INSTALLED / checker / "stdlib"
        assert "ty_extensions: 3.0-" in (stdlib / "VERSIONS").read_text()
        assert "closed=True" in (stdlib / "typing.pyi").read_text()


def test_installed_trees_carry_the_overlays() -> None:
    shared = _TYPINGS / "stdlib"
    for source in shared.rglob("*.pyi"):
        for checker in ("ty", "basedpyright"):
            installed = _INSTALLED / checker / "stdlib" / source.relative_to(shared)
            assert installed.read_bytes() == source.read_bytes(), installed
    bpr_only = _TYPINGS / "basedpyright"
    for source in bpr_only.rglob("*.pyi"):
        rel = source.relative_to(bpr_only)
        installed = _INSTALLED / "basedpyright" / "stdlib" / rel
        assert installed.read_bytes() == source.read_bytes(), installed
        assert (_INSTALLED / "ty" / "stdlib" / rel).read_bytes() != source.read_bytes()


def test_bundle_still_needs_the_patch(tmp_path: Path) -> None:
    """Upstream fixed it: drop this package instead of carrying a no-op."""
    rekursiv_ai_typeshed.extract_ty_stdlib(tmp_path)
    text = (tmp_path / "stdlib" / "builtins.pyi").read_text()
    assert _POW_ANY.search(text) is not None


# Two full tree builds (1.5 s): copies both stub bundles and applies the patch.
@pytest.mark.compute_large_fixture
def test_build_is_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "typeshed.d"
    rekursiv_ai_typeshed.build(target)
    first = {
        p.relative_to(target): p.read_bytes() for p in target.rglob("*") if p.is_file()
    }
    rekursiv_ai_typeshed.build(target)
    second = {
        p.relative_to(target): p.read_bytes() for p in target.rglob("*") if p.is_file()
    }
    assert first == second


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
