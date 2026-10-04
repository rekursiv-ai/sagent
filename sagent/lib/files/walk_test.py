"""Tests for ``walk``: a tree listed the way ripgrep lists one."""

from pathlib import Path

import subprocess

import pytest

from sagent.lib.files import walk


@pytest.fixture(autouse=True)
def repository(tmp_path: Path) -> None:
    """Make ``tmp_path`` a git root: ripgrep reads ``.gitignore`` only inside one."""
    (tmp_path / ".git").mkdir()


def _write(root: Path, name: str, body: str = "x\n") -> Path:
    """Write ``body`` to ``root / name``, creating parents."""
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def _listed(directory: Path) -> list[str]:
    """Return ``walk.files(directory)`` as paths relative to ``directory``."""
    return [path.relative_to(directory).as_posix() for path in walk.files(directory)]


def test_every_file_is_listed_in_sorted_depth_first_order(tmp_path: Path) -> None:
    for name in ("b.py", "a/z.txt", "a/b/c.py", "a/a.md", "data.bin"):
        _write(tmp_path, name)

    assert _listed(tmp_path) == ["a/a.md", "a/b/c.py", "a/z.txt", "b.py", "data.bin"]


def test_hidden_files_and_directories_are_skipped(tmp_path: Path) -> None:
    _write(tmp_path, ".env")
    _write(tmp_path, ".venv/lib/site.py")
    _write(tmp_path, "pkg/.cache/blob")
    kept = _write(tmp_path, "pkg/mod.py")

    assert list(walk.files(tmp_path)) == [kept]


def test_a_gitignore_drops_files_and_directories(tmp_path: Path) -> None:
    _write(tmp_path, ".gitignore", "*.log\nbuild/\n")
    _write(tmp_path, "run.log")
    _write(tmp_path, "build/out.py")
    _write(tmp_path, "pkg/build/deep.py")

    assert _listed(tmp_path) == []


def test_a_directory_pattern_does_not_drop_a_file_of_that_name(tmp_path: Path) -> None:
    """``build/`` names directories only; git keeps a file called ``build``."""
    _write(tmp_path, ".gitignore", "build/\n")
    _write(tmp_path, "build")

    assert _listed(tmp_path) == ["build"]


def test_an_anchored_pattern_matches_only_at_its_own_level(tmp_path: Path) -> None:
    _write(tmp_path, ".gitignore", "/runs/\n")
    _write(tmp_path, "runs/a.py")
    _write(tmp_path, "pkg/runs/b.py")

    assert _listed(tmp_path) == ["pkg/runs/b.py"]


def test_a_nested_gitignore_applies_relative_to_its_own_directory(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "pkg/.gitignore", "/gen/\n")
    _write(tmp_path, "pkg/gen/a.py")
    _write(tmp_path, "gen/b.py")

    assert _listed(tmp_path) == ["gen/b.py"]


def test_a_deeper_negation_overrides_a_shallower_ignore(tmp_path: Path) -> None:
    _write(tmp_path, ".gitignore", "*.ipynb\n")
    _write(tmp_path, "sub/.gitignore", "!keep.ipynb\n")
    _write(tmp_path, "sub/keep.ipynb")
    _write(tmp_path, "sub/drop.ipynb")
    _write(tmp_path, "top.ipynb")

    assert _listed(tmp_path) == ["sub/keep.ipynb"]


def test_a_later_line_in_one_file_overrides_an_earlier_one(tmp_path: Path) -> None:
    _write(tmp_path, ".gitignore", "*.py\n!keep.py\n")
    _write(tmp_path, "keep.py")
    _write(tmp_path, "drop.py")

    assert _listed(tmp_path) == ["keep.py"]


def test_an_ignored_directory_is_not_entered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pruned tree costs nothing: ``.venv`` and ``node_modules`` are huge."""
    _write(tmp_path, ".gitignore", "node_modules/\n")
    _write(tmp_path, "node_modules/pkg/index.js")
    _write(tmp_path, "src/a.py")
    entered: list[Path] = []
    real_iterdir = Path.iterdir

    def recording_iterdir(self: Path):
        entered.append(self)
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", recording_iterdir)

    assert _listed(tmp_path) == ["src/a.py"]
    assert tmp_path / "node_modules" not in entered


def test_gitignores_above_the_walked_directory_apply(tmp_path: Path) -> None:
    """``tool sub`` honors the root's ``.gitignore``, as ``tool .`` would."""
    _write(tmp_path, "pyproject.toml")
    _write(tmp_path, ".gitignore", "sub/gen/\n*.tmp\n")
    _write(tmp_path, "sub/gen/a.py")
    _write(tmp_path, "sub/b.tmp")
    _write(tmp_path, "sub/c.py")

    assert _listed(tmp_path / "sub") == ["c.py"]


def test_a_gitignore_above_the_repository_never_applies(tmp_path: Path) -> None:
    """A ``.gitignore`` outside the repository must not hide its files."""
    _write(tmp_path, ".gitignore", "*.py\n")
    project = tmp_path / "project"
    (project / ".git").mkdir(parents=True)
    _write(project, "sub/a.py")

    assert _listed(project / "sub") == ["a.py"]


def test_a_symlink_is_listed_and_never_followed(tmp_path: Path) -> None:
    """A linked directory is one entry, not a second copy of its tree."""
    _write(tmp_path, "real/a.py")
    (tmp_path / "linked").symlink_to("real", target_is_directory=True)
    (tmp_path / "alias.py").symlink_to("real/a.py")

    assert _listed(tmp_path) == ["alias.py", "linked", "real/a.py"]


def test_a_symlink_is_ignored_by_where_it_sits_not_where_it_points(
    tmp_path: Path,
) -> None:
    _write(tmp_path, ".gitignore", "links/\n")
    _write(tmp_path, "real/a.py")
    (tmp_path / "links").mkdir()
    (tmp_path / "links/a.py").symlink_to("../real/a.py")
    (tmp_path / "pointer.py").symlink_to("links/a.py")

    assert _listed(tmp_path) == ["pointer.py", "real/a.py"]


def test_a_kept_symlink_to_an_ignored_target_is_listed(tmp_path: Path) -> None:
    """The target's path is never matched: only the link's own location is."""
    _write(tmp_path, ".gitignore", "vendored/\n")
    _write(tmp_path, "vendored/a.py")
    (tmp_path / "shim.py").symlink_to("vendored/a.py")

    assert _listed(tmp_path) == ["shim.py"]


def test_a_dangling_symlink_is_still_listed(tmp_path: Path) -> None:
    (tmp_path / "gone.py").symlink_to("missing.py")

    assert _listed(tmp_path) == ["gone.py"]


def test_paths_are_joined_to_the_directory_as_given(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A relative directory yields relative paths, as a CLI argument would."""
    _write(tmp_path, "pkg/a.py")
    monkeypatch.chdir(tmp_path)

    assert list(walk.files(Path("pkg"))) == [Path("pkg/a.py")]


def test_walking_never_runs_a_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write(tmp_path, ".gitignore", "*.log\n")
    _write(tmp_path, "a.py")

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"ran a subprocess: {args} {kwargs}")

    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(subprocess, "Popen", refuse)

    assert _listed(tmp_path) == ["a.py"]


_SUFFIXES = {".py", ".pyi", ".ipynb"}


def _kinds(root: Path) -> dict[str, Path]:
    """Return a tree holding one wanted file per suffix, plus files to drop."""
    _write(root, ".gitignore", "ignored.py\nbuild/\n")
    _write(root, "sub/deep/.gitignore", "*.ipynb\n!keep.ipynb\n")
    for dropped in (
        "ignored.py",
        "build/gen.py",
        "notes.txt",
        ".venv/lib/site.py",
        ".hidden/h.py",
        "sub/deep/dropped.ipynb",
    ):
        _write(root, dropped)
    return {
        "py": _write(root, "a.py"),
        "pyi": _write(root, "sub/b.pyi"),
        "ipynb": _write(root, "sub/deep/keep.ipynb"),
    }


def test_a_directory_expands_to_files_with_the_wanted_suffixes(tmp_path: Path) -> None:
    files = _kinds(tmp_path)

    assert sorted(walk.expand([tmp_path], suffixes=_SUFFIXES)) == sorted(
        files.values(),
    )


def test_expanding_a_directory_never_runs_a_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = _kinds(tmp_path)

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"ran a subprocess: {args} {kwargs}")

    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(subprocess, "Popen", refuse)

    assert sorted(walk.expand([tmp_path], suffixes=_SUFFIXES)) == sorted(
        files.values(),
    )


def test_expanding_honors_a_gitignore_above_the_named_directory(
    tmp_path: Path,
) -> None:
    _write(tmp_path, ".gitignore", "sub/gen/\n")
    _write(tmp_path, "pyproject.toml")
    kept = _write(tmp_path, "sub/a.py")
    _write(tmp_path, "sub/gen/b.py")

    assert walk.expand([tmp_path / "sub"], suffixes=_SUFFIXES) == [kept]


def test_a_subdirectory_expands_to_only_its_own_files(tmp_path: Path) -> None:
    files = _kinds(tmp_path)

    assert walk.expand([tmp_path / "sub"], suffixes=_SUFFIXES) == [
        files["pyi"],
        files["ipynb"],
    ]


def test_only_the_wanted_suffixes_are_expanded(tmp_path: Path) -> None:
    files = _kinds(tmp_path)

    assert walk.expand([tmp_path], suffixes={".py"}) == [files["py"]]


def test_a_named_file_is_kept_as_given(tmp_path: Path) -> None:
    """A file passes through unexpanded, as pre-commit's file lists do."""
    named = _write(tmp_path, "notes.txt")

    assert walk.expand([named], suffixes=_SUFFIXES) == [named]


def test_a_file_named_twice_is_listed_once(tmp_path: Path) -> None:
    files = _kinds(tmp_path)

    assert walk.expand(
        [files["py"], tmp_path, files["py"]],
        suffixes=_SUFFIXES,
    ) == [files["py"], files["pyi"], files["ipynb"]]


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
