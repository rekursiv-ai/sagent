"""Tests for ``files.glob``: fd and the rignore backend answer alike."""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

import os
import shutil
import subprocess

import pytest

from sagent.lib.files import glob as files_glob
from sagent.lib.files.glob import GlobError, fd_command, glob


@pytest.fixture(autouse=True)
def backend(fd: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide ``fd`` unless the test runs against it."""
    found = shutil.which("fd") or shutil.which("fdfind")
    if fd and found is None:
        pytest.skip("fd is not installed")

    def which(name: str) -> str | None:
        return found if fd and name in {"fd", "fdfind"} else None

    # A test for the fd backend must reach it, whatever the entry cap says;
    # without this the in-process walk answers every small tree and the ``fd``
    # parameter tests nothing.
    monkeypatch.setattr(shutil, "which", which)
    monkeypatch.setattr(files_glob, "_FD_ENTRIES", -1)


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


@_BOTH
@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("build/*", ["build/gen.py"]),
        ("build/**/*.py", ["build/gen.py"]),
        ("linked/*.py", ["linked/c.py"]),
        (".venv/*.py", [".venv/site.py"]),
        ("*.py", ["a.py", "link.py"]),
        ("nosuch/*", []),
    ],
    ids=["ignored", "ignored-deep", "symlinked", "hidden", "the-root", "missing"],
)
def test_an_absolute_pattern_enters_its_directory_as_a_root_is_entered(
    tmp_path: Path,
    pattern: str,
    expected: list[str],
) -> None:
    """The directory it spells out is named, as ``root`` is, not judged as a prefix.

    The ``root`` given is passed over, even one that does not exist.
    """
    (tmp_path / "repo").mkdir()
    root = _tree(tmp_path / "repo")
    (root / "linked").symlink_to(root / "sub")
    assert glob(tmp_path / "nowhere", f"{root}/{pattern}") == [
        root / name for name in expected
    ]


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


def test_an_entry_that_vanished_since_the_walk_stays_listed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed ``lstat`` is a race, not a directory, so the name is still named."""
    (tmp_path / "gone.py").write_text("")

    def vanished(*_args: object, **_kwargs: object) -> NoReturn:
        raise OSError

    with monkeypatch.context() as patch:
        patch.setattr(Path, "lstat", vanished)
        assert glob(tmp_path, "*.py") == [tmp_path / "gone.py"]


def test_the_walk_hands_off_to_fd_only_past_the_entry_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Up to ``_FD_ENTRIES`` entries below the root, the walk's answer stands."""
    for i in range(3):
        (tmp_path / f"{i}.py").write_text("")

    def which(name: str) -> str:
        return f"/usr/bin/{name}"

    ran_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def ran(*args: object, **kwargs: object) -> object:
        ran_calls.append((args, kwargs))
        return list[Path]()

    monkeypatch.setattr(shutil, "which", which)
    monkeypatch.setattr(files_glob, "_FD_ENTRIES", 3)
    monkeypatch.setattr(files_glob, "_glob_fd", ran)
    assert glob(tmp_path, "**/*.py") == [tmp_path / f"{i}.py" for i in range(3)]
    assert ran_calls == []
    (tmp_path / "3.py").write_text("")
    assert glob(tmp_path, "**/*.py") == []
    assert len(ran_calls) == 1


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a mode-000 directory")
def test_a_bounded_glob_never_reads_below_its_depth(tmp_path: Path) -> None:
    """An unreadable directory past the pattern's depth cannot fail the glob."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.py").write_text("")
    (tmp_path / "sub" / "b.py").write_text("")
    locked = [tmp_path / "locked", tmp_path / "sub" / "locked"]
    for directory in locked:
        directory.mkdir()
        directory.chmod(0)
    try:
        assert glob(tmp_path, "*.py") == [tmp_path / "a.py"]
        assert glob(tmp_path, "sub/*.py") == [tmp_path / "sub" / "b.py"]
    finally:
        for directory in locked:
            directory.chmod(0o700)


def test_a_parent_segment_does_not_opt_into_hidden_entries(tmp_path: Path) -> None:
    """``..`` starts with a dot but names a directory, not a hidden entry."""
    root = _tree(tmp_path)
    assert glob(root, f"{root}/sub/../*") == [
        root / "sub" / ".." / name for name in ("a.py", "b.txt", "link.py")
    ]


@pytest.mark.parametrize("fd", [True], ids=["fd"])
def test_either_side_of_the_entry_limit_answers_alike(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Over ``_FD_ENTRIES`` the in-process walk gives up and fd finishes instead."""
    for i in range(6):
        (tmp_path / f"pkg{i}").mkdir()
        (tmp_path / f"pkg{i}" / f"mod{i}.py").write_text("")
    expected = [tmp_path / f"pkg{i}" / f"mod{i}.py" for i in range(6)]
    monkeypatch.setattr(files_glob, "_FD_ENTRIES", 2)
    assert glob(tmp_path, "**/*.py") == expected
    monkeypatch.setattr(files_glob, "_FD_ENTRIES", 10_000)
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
        ("**a.py", ["a.py"]),
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
        "globstar-inside-a-name",
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
@pytest.mark.parametrize("bang", ["!", "^"])
def test_a_negated_class_may_hold_the_bracket_itself(
    tmp_path: Path,
    bang: str,
) -> None:
    """``[!]]`` is "anything but ``]``", so the close sits past the negation."""
    (tmp_path / "]").write_text("")
    (tmp_path / "x").write_text("")
    assert glob(tmp_path, f"**/[{bang}]]") == [tmp_path / "x"]


@_BOTH
def test_a_backslash_in_a_class_is_a_member(tmp_path: Path) -> None:
    r"""``[\]`` names ``\`` itself, as fd reads it."""
    (tmp_path / "\\").write_text("")
    (tmp_path / "X").write_text("")
    assert glob(tmp_path, "**/[\\]") == [tmp_path / "\\"]


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
def test_a_failed_fd_run_raises_rather_than_answering_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-zero exit is a fault to surface, not an empty answer to believe."""
    (tmp_path / "a.py").write_text("")
    argv = [shutil.which("fd") or "", "--not-a-real-flag"]

    def fd_command(*_args: object, **_kwargs: object) -> object:
        return argv

    monkeypatch.setattr(files_glob, "fd_command", fd_command)
    with pytest.raises(subprocess.CalledProcessError):
        glob(tmp_path, "**/*.py")


@pytest.mark.parametrize("fd", [True], ids=["fd"])
def test_the_default_fd_timeout_is_thirty_seconds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "a.py").write_text("")
    run_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def run(*args: object, **kwargs: object) -> object:
        run_calls.append((args, kwargs))
        return subprocess.CompletedProcess([], 0, "", "")

    with monkeypatch.context() as patch:
        patch.setattr(subprocess, "run", run)
        assert glob(tmp_path, "**/*.py") == []
    (call,) = run_calls
    assert call[1]["timeout"] == 30


@pytest.mark.parametrize("fd", [True], ids=["fd"])
def test_an_fd_that_hangs_is_stopped_at_the_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A glob run in a thread cannot be cancelled, so only a limit bounds it."""
    (tmp_path / "a.py").write_text("")
    pid = tmp_path / "fd.pid"
    argv = ["/bin/sh", "-c", f'echo $$ > "{pid}"; exec sleep 60']

    def fd_command(*_args: object, **_kwargs: object) -> object:
        return argv

    monkeypatch.setattr(files_glob, "fd_command", fd_command)
    with pytest.raises(GlobError, match=r"^fd timed out after 0\.05s; narrow"):
        glob(tmp_path, "**/*.py", timeout_sec=0.05)
    assert not Path(f"/proc/{pid.read_text().strip()}").exists()


@pytest.mark.parametrize("name", ["fd", "fdfind"])
def test_either_spelling_of_the_fd_executable_is_used(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    """Debian ships fd as ``fdfind``, so a host with only one name still uses it."""
    (tmp_path / "a.py").write_text("")

    def which(asked: str) -> str | None:
        return f"/usr/bin/{name}" if asked == name else None

    ran_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def ran(*args: object, **kwargs: object) -> object:
        ran_calls.append((args, kwargs))
        return list[Path]()

    monkeypatch.setattr(shutil, "which", which)
    monkeypatch.setattr(files_glob, "_FD_ENTRIES", -1)
    monkeypatch.setattr(files_glob, "_glob_fd", ran)
    assert glob(tmp_path, "**/*.py") == []
    # A bounded pattern has no deep walk to hand off, so fd stays unused.
    assert glob(tmp_path, "*.py") == [tmp_path / "a.py"]
    (call,) = ran_calls
    assert call[0][0] == f"/usr/bin/{name}"


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
    with pytest.raises(GlobError, match=r"^invalid glob '\*\*/\[z-a\]': bad"):
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
    (tmp_path / "X").touch()
    assert glob(tmp_path, f"**/{pattern}") == [tmp_path / member]


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
