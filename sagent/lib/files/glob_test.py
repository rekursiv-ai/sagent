"""Tests for ``files.glob``: fd and the rignore backend answer alike."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import shutil
import subprocess

import pytest

from sagent.lib.files.glob import GlobError, fd_command, glob


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture(autouse=True)
def backend(fd: bool) -> Iterator[None]:
    """Hide ``fd`` unless the test runs against it."""
    found = shutil.which("fd") or shutil.which("fdfind")
    if fd and found is None:
        pytest.skip("fd is not installed")

    def which(name: str) -> str | None:
        return found if fd and name in {"fd", "fdfind"} else None

    # A test for the fd backend must reach it, whatever the entry cap says;
    # without this the in-process walk answers every small tree and the ``fd``
    # parameter tests nothing.
    with patch("shutil.which", which), patch("sagent.lib.files.glob._FD_ENTRIES", -1):
        yield


@pytest.fixture
def fd() -> bool:
    """Default to the rignore backend; ``_BOTH`` overrides per test."""
    return False


_BOTH = pytest.mark.parametrize("fd", [True, False], ids=["fd", "rignore"])


def _tree(root: Path) -> Path:
    """Make ``root`` a git root with ignored, hidden, nested, and linked files."""
    (root / ".git").mkdir()
    (root / ".gitignore").write_text("build/\n")
    for name in ("a.py", "b.txt", "sub/c.py", ".env", ".venv/site.py", "build/gen.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    (root / "link.py").symlink_to("a.py")
    return root


@_BOTH
@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("**/*.py", ["a.py", "link.py", "sub/c.py"]),
        ("*.py", ["a.py", "link.py"]),
        ("sub/*", ["sub/c.py"]),
        (".*", [".env", ".gitignore"]),
        (".venv/*.py", [".venv/site.py"]),
        ("*.md", []),
        ("*.PY", []),
        ("**/a.PY", []),
    ],
)
def test_a_glob_names_kept_files_and_dot_segments_opt_into_hidden(
    tmp_path: Path,
    pattern: str,
    expected: list[str],
) -> None:
    root = _tree(tmp_path)
    assert glob(root, pattern) == [root / name for name in expected]


@_BOTH
def test_a_missing_or_file_root_matches_nothing(tmp_path: Path) -> None:
    (tmp_path / "f.py").write_text("")
    assert glob(tmp_path / "nope", "*") == []
    assert glob(tmp_path / "f.py", "*") == []


@_BOTH
@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("build/*", []),
        ("sub/*", ["sub/c.py"]),
        (".venv/*.py", [".venv/site.py"]),
        ("nosuch/*", []),
    ],
    ids=["ignored-prefix", "kept-prefix", "hidden-prefix", "missing-prefix"],
)
def test_a_literal_prefix_must_itself_be_a_kept_directory(
    tmp_path: Path,
    pattern: str,
    expected: list[str],
) -> None:
    """``rignore`` never judges its own start, so the prefix is checked here."""
    root = _tree(tmp_path)
    assert glob(root, pattern) == [root / name for name in expected]


@_BOTH
def test_a_symlink_to_a_directory_is_listed_but_never_descended(
    tmp_path: Path,
) -> None:
    """It is an entry in its own right, so its target's files match only once."""
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "deep.py").write_text("")
    (tmp_path / "link").symlink_to(tmp_path / "real")
    assert glob(tmp_path, "*") == [tmp_path / "link"]
    assert glob(tmp_path, "**/*.py") == [tmp_path / "real" / "deep.py"]


@_BOTH
def test_dot_git_is_skipped_even_when_a_pattern_names_it(tmp_path: Path) -> None:
    """A dot segment opts into hidden entries; ``.git`` is excluded regardless."""
    root = _tree(tmp_path)
    (root / ".git" / "config").write_text("")
    assert glob(root, ".git/*") == []
    assert glob(root, ".*/*.py") == [root / ".venv" / "site.py"]


@_BOTH
def test_a_symlinked_prefix_is_never_descended(tmp_path: Path) -> None:
    """A symlink is an entry, not a way in, so nothing matches below one."""
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "x.py").write_text("")
    (tmp_path / "link").symlink_to(tmp_path / "real")
    assert glob(tmp_path, "link/*") == []
    assert glob(tmp_path, "real/*") == [tmp_path / "real" / "x.py"]


def test_a_prefix_is_not_satisfied_by_a_root_sharing_its_name(
    tmp_path: Path,
) -> None:
    """A walk lists its own start, whose name must not stand in for a child."""
    root = tmp_path / "build"
    (root / "build").mkdir(parents=True)
    (root / ".git").mkdir()
    (root / ".gitignore").write_text("build/\n")
    (root / "build" / "gen.py").write_text("")
    assert glob(root, "build/*") == []


def test_an_entry_that_vanished_since_the_walk_stays_listed(tmp_path: Path) -> None:
    """A failed ``lstat`` is a race, not a directory, so the name is still named."""
    (tmp_path / "gone.py").write_text("")
    with patch("pathlib.Path.lstat", side_effect=OSError):
        assert glob(tmp_path, "*.py") == [tmp_path / "gone.py"]


@pytest.mark.parametrize("fd", [True], ids=["fd"])
def test_either_side_of_the_entry_limit_answers_alike(tmp_path: Path) -> None:
    """Over ``_FD_ENTRIES`` the in-process walk gives up and fd finishes instead."""
    for i in range(6):
        (tmp_path / f"pkg{i}").mkdir()
        (tmp_path / f"pkg{i}" / f"mod{i}.py").write_text("")
    expected = [tmp_path / f"pkg{i}" / f"mod{i}.py" for i in range(6)]
    with patch("sagent.lib.files.glob._FD_ENTRIES", 2):
        assert glob(tmp_path, "**/*.py") == expected
    with patch("sagent.lib.files.glob._FD_ENTRIES", 10_000):
        assert glob(tmp_path, "**/*.py") == expected


@_BOTH
def test_a_root_reached_through_a_symlink_names_files_under_that_spelling(
    tmp_path: Path,
) -> None:
    (tmp_path / "real").mkdir()
    real = _tree(tmp_path / "real")
    linked = tmp_path / "linked"
    linked.symlink_to(real)
    assert glob(linked, "**/*.py") == [
        linked / name for name in ("a.py", "link.py", "sub/c.py")
    ]


@_BOTH
@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("**/*.{py,txt}", ["a.py", "b.txt", "link.py", "sub/c.py"]),
        ("**/{a,b}.*", ["a.py", "b.txt"]),
        ("**/[!a]*.py", ["link.py", "sub/c.py"]),
        ("**/[ab].*", ["a.py", "b.txt"]),
        ("**/[^a]*.py", ["link.py", "sub/c.py"]),
        ("**/?.py", ["a.py", "sub/c.py"]),
        ("**", ["a.py", "b.txt", "link.py", "sub/c.py"]),
        ("**/{a,b}.{py,txt}", ["a.py", "b.txt"]),
        ("**/[ab].[pt]*", ["a.py", "b.txt"]),
        ("**/.venv/**/*.py", [".venv/site.py"]),
        ("*/**", ["sub/c.py"]),
    ],
    ids=[
        "braces",
        "brace-stems",
        "bang-negated-class",
        "class",
        "caret-negated-class",
        "question",
        "bare-star",
        "two-brace-groups",
        "two-classes",
        "two-globstars",
        "globstar-after-a-wildcard",
    ],
)
def test_every_glob_metacharacter_reads_as_fd_reads_it(
    tmp_path: Path,
    pattern: str,
    expected: list[str],
) -> None:
    """``{a,b}``, ``[!a]``, ``?`` and a bare ``**`` each have one meaning."""
    root = _tree(tmp_path)
    assert glob(root, pattern) == [root / name for name in expected]


@_BOTH
def test_a_leading_bracket_in_a_class_is_a_member_not_the_close(
    tmp_path: Path,
) -> None:
    """``[]]`` names ``]`` itself, as globset and a shell both read it."""
    (tmp_path / "]").write_text("")
    (tmp_path / "a.py").write_text("")
    assert glob(tmp_path, "**/[]]") == [tmp_path / "]"]


@_BOTH
def test_a_negated_class_may_hold_the_bracket_itself(tmp_path: Path) -> None:
    """``[!]]`` is "anything but ``]``", so the close sits past the negation."""
    (tmp_path / "]").write_text("")
    (tmp_path / "x").write_text("")
    assert glob(tmp_path, "**/[!]]") == [tmp_path / "x"]


@_BOTH
@pytest.mark.parametrize(
    ("pattern", "message"),
    [
        ("**/[abc", "unclosed character class"),
        ("**/{a,b", "unclosed alternate group"),
        ("**/{a,{b,c}}", "nested alternate group"),
    ],
    ids=["unclosed-class", "unclosed-group", "nested-group"],
)
def test_a_malformed_pattern_is_one_named_error_in_either_backend(
    tmp_path: Path,
    pattern: str,
    message: str,
) -> None:
    """Fd rejects each of these, so neither backend may crash its own way."""
    (tmp_path / "a.py").write_text("")
    with pytest.raises(GlobError, match=message):
        glob(tmp_path, pattern)


@_BOTH
def test_a_hidden_target_is_found_when_the_pattern_opts_in(tmp_path: Path) -> None:
    """``**/.env`` names a hidden entry, so both backends must opt into them."""
    root = _tree(tmp_path)
    assert glob(root, "**/.env") == [root / ".env"]


@_BOTH
def test_paths_sort_by_component_not_as_plain_text(tmp_path: Path) -> None:
    """``a/z.py`` precedes ``a-b/z.py``, though ``-`` sorts before ``/`` as text."""
    for directory in ("a", "a-b"):
        (tmp_path / directory).mkdir()
        (tmp_path / directory / "z.py").write_text("")
    assert glob(tmp_path, "**/*.py") == [
        tmp_path / "a" / "z.py",
        tmp_path / "a-b" / "z.py",
    ]


@pytest.mark.parametrize("fd", [True], ids=["fd"])
def test_a_failed_fd_run_raises_rather_than_answering_empty(tmp_path: Path) -> None:
    """A non-zero exit is a fault to surface, not an empty answer to believe."""
    (tmp_path / "a.py").write_text("")
    argv = [shutil.which("fd") or "", "--not-a-real-flag"]
    with (
        patch("sagent.lib.files.glob.fd_command", return_value=argv),
        pytest.raises(subprocess.CalledProcessError),
    ):
        glob(tmp_path, "**/*.py")


@pytest.mark.parametrize("name", ["fd", "fdfind"])
def test_either_spelling_of_the_fd_executable_is_used(
    tmp_path: Path,
    name: str,
) -> None:
    """Debian ships fd as ``fdfind``, so a host with only one name still uses it."""
    (tmp_path / "a.py").write_text("")

    def which(asked: str) -> str | None:
        return f"/usr/bin/{name}" if asked == name else None

    with (
        patch("shutil.which", which),
        patch("sagent.lib.files.glob._FD_ENTRIES", -1),
        patch("sagent.lib.files.glob._glob_fd", return_value=[]) as ran,
    ):
        assert glob(tmp_path, "**/*.py") == []
        # A bounded pattern has no deep walk to hand off, so fd stays unused.
        assert glob(tmp_path, "*.py") == [tmp_path / "a.py"]
    assert ran.call_args.args[0] == f"/usr/bin/{name}"
    assert ran.call_count == 1


@_BOTH
def test_dot_git_is_excluded_from_the_walk_itself(tmp_path: Path) -> None:
    """Hidden entries are searched, so ``.git`` needs excluding on its own."""
    root = _tree(tmp_path)
    (root / ".git" / "config").write_text("")
    assert glob(root, "**/.git/**") == []


@_BOTH
def test_a_literal_prefix_ends_at_the_first_wildcard_segment(
    tmp_path: Path,
) -> None:
    """``a/*/c.py`` walks from ``a``; taking only the first segment would not."""
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "c.py").write_text("")
    assert glob(tmp_path, "a/*/c.py") == [tmp_path / "a" / "b" / "c.py"]


@pytest.mark.parametrize(("hidden", "flag"), [(False, []), (True, ["--hidden"])])
def test_fd_command_anchors_the_glob_at_the_root(
    tmp_path: Path,
    hidden: bool,
    flag: list[str],
) -> None:
    assert fd_command("/fd", tmp_path, "**/*.py", hidden=hidden) == [
        *("/fd", "--type", "file", "--type", "symlink", "--glob", "--case-sensitive"),
        "--full-path",
        *("--exclude", ".git", *flag, "--absolute-path", "--"),
        *(f"{tmp_path.resolve()}/**/*.py", str(tmp_path)),
    ]


@_BOTH
def test_invalid_ranges_raise_the_public_error(tmp_path: Path) -> None:
    (tmp_path / "a").touch()
    with pytest.raises(GlobError):
        glob(tmp_path, "**/[z-a]")


@_BOTH
@pytest.mark.parametrize("name", ["we[ir]d", "we{ir,d}", "we?ird", "we*ird"])
def test_root_names_are_literal(tmp_path: Path, name: str) -> None:
    root = tmp_path / name
    root.mkdir()
    (root / "a.py").touch()
    assert glob(root, "**/*.py") == [root / "a.py"]


@_BOTH
def test_current_directory_segment_does_not_enable_hidden(tmp_path: Path) -> None:
    (tmp_path / ".env").touch()
    (tmp_path / "a").touch()
    assert glob(tmp_path, "./*") == [tmp_path / "a"]
    assert glob(tmp_path, "./**/*") == [tmp_path / "a"]


@_BOTH
@pytest.mark.filterwarnings("error::FutureWarning")
@pytest.mark.parametrize(
    ("pattern", "member"),
    [("[[]", "["), ("[&&]", "&"), ("[~~]", "~")],
)
def test_class_members_are_not_regex_set_operations(
    tmp_path: Path,
    pattern: str,
    member: str,
) -> None:
    (tmp_path / member).touch()
    (tmp_path / "x").touch()
    assert glob(tmp_path, f"**/{pattern}") == [tmp_path / member]


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
