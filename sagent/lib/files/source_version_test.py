"""The shared fingerprint retains trax's source and racy-clean contract."""

from __future__ import annotations

from typing import TYPE_CHECKING

import os

from sagent.lib.files.source_version import _read, source_version


if TYPE_CHECKING:
    from pathlib import Path


def test_fingerprint_preserves_the_existing_fixed_tree_digest(tmp_path: Path) -> None:
    root = tmp_path / "pkg"
    for name, text, tick in (
        ("pkg/a.py", "x = 1\n", 2),
        ("pkg/sub/b.py", "y = 2\n", 4),
        ("pkg/sub/b_test.py", "", 6),
        ("pkg/conftest.py", "", 6),
        ("lib/c.py", "z = 3\n", 5),
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        os.utime(path, ns=(tick * 1_000_000_000, tick * 1_000_000_000))
    assert (
        source_version(root, outside=("../lib/c.py",))
        == "66aaaa21a3f8174fcd4ee66fc314b957"
    )


def test_empty_deleted_and_racy_sources(tmp_path: Path) -> None:
    empty = source_version(tmp_path)
    assert empty == "cae66941d9efbd404e4d88758ea67670"
    path = tmp_path / "a.py"
    path.write_text("x = 1\n")
    tick = path.stat().st_mtime_ns
    before = source_version(tmp_path)
    assert source_version(tmp_path) == before
    path.write_text("x = 2\n")
    os.utime(path, ns=(tick, tick))
    assert source_version(tmp_path) != before
    assert source_version(tmp_path, outside=("missing.py",)) == source_version(tmp_path)
    (tmp_path / "gone.py").symlink_to(tmp_path / "missing.py")
    assert source_version(tmp_path) == source_version(tmp_path, outside=("gone.py",))
    assert _read(path) == b"x = 2\n"
    assert _read(tmp_path / "missing.py") == b""


def test_missing_dependency_does_not_hide_later_racy_edits(tmp_path: Path) -> None:
    root = tmp_path / "pkg"
    root.mkdir()
    dependency = tmp_path / "dependency.py"
    dependency.write_text("x = 1\n")
    status = dependency.stat()
    outside = ("../missing.py", "../dependency.py")
    before = source_version(root, outside=outside)

    dependency.write_text("x = 2\n")
    os.utime(dependency, ns=(status.st_atime_ns, status.st_mtime_ns))
    after = dependency.stat()
    assert after.st_size == status.st_size
    assert after.st_mtime_ns == status.st_mtime_ns
    assert source_version(root, outside=outside) != before


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
