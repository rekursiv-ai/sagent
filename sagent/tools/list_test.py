"""Tests for the List tool, its sorting and ``ls -l`` rows, and ``ls`` taken on."""

from __future__ import annotations

from typing import TYPE_CHECKING

import os
import time

import pytest

from sagent.tools.command_analyzer import invocations
from sagent.tools.list import List, ls_row, sort_paths
from sagent.tools.types import ToolContext, parse


if TYPE_CHECKING:
    from pathlib import Path

    from treekle import PlainTree


def test_summary_names_the_directory() -> None:
    assert List().summary({"path": "src"}) == "List src"
    assert List().summary({"path": ""}) == "List ."
    assert List().summary({}) == "List ."


def test_listing_runs_beside_anything(tmp_path: Path) -> None:
    assert List().serialize_key(ToolContext(cwd=tmp_path), {"path": "."}) is None


@pytest.mark.asyncio
async def test_entries_are_listed_by_name_with_directories_marked(
    tmp_path: Path,
) -> None:
    for name in ("b.txt", "A.txt", "a.txt", ".hidden"):
        (tmp_path / name).write_text("")
    (tmp_path / "sub").mkdir()
    assert await _list(tmp_path, path=".") == "A.txt\na.txt\nb.txt\nsub/"
    assert await _list(tmp_path, path="") == "A.txt\na.txt\nb.txt\nsub/"
    assert await _list(tmp_path, path=".", show_hidden=True) == (
        ".hidden\nA.txt\na.txt\nb.txt\nsub/"
    )


@pytest.mark.asyncio
async def test_a_relative_or_tilde_path_is_found_from_where_it_says(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "home").mkdir()
    (tmp_path / "home" / "h.txt").write_text("")
    (tmp_path / "work" / "sub").mkdir(parents=True)
    (tmp_path / "work" / "sub" / "s.txt").write_text("")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert await _list(tmp_path / "work", path="~") == "h.txt"
    assert await _list(tmp_path / "work", path="sub") == "s.txt"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sort", "expected"),
    [
        ("name", "a\nb\nc"),
        ("name_desc", "c\nb\na"),
        ("mtime", "b\nc\na"),
        ("mtime_desc", "a\nc\nb"),
        ("size", "c\na\nb"),
        ("size_desc", "b\na\nc"),
    ],
)
async def test_a_sort_orders_the_entries(
    tmp_path: Path,
    sort: str,
    expected: str,
) -> None:
    for name, size, age in (("a", 2, 0), ("b", 3, 200), ("c", 1, 100)):
        (tmp_path / name).write_bytes(b"x" * size)
        when = time.time() - age
        os.utime(tmp_path / name, (when, when))
    assert await _list(tmp_path, path=".", sort=sort) == expected


@pytest.mark.asyncio
async def test_a_cap_keeps_the_head_or_the_tail_and_counts_the_rest(
    tmp_path: Path,
) -> None:
    for i in range(1, 6):
        (tmp_path / f"f{i}").write_text("")
    assert await _list(tmp_path, path=".", max_results=2) == "f1\nf2\n... (3 more)"
    assert await _list(tmp_path, path=".", keep_last=2) == "f4\nf5\n... (3 more)"
    assert await _list(tmp_path, path=".", keep_last=2, sort="name_desc") == (
        "f2\nf1\n... (3 more)"
    )
    assert await _list(tmp_path, path=".", max_results=5) == "f1\nf2\nf3\nf4\nf5"


@pytest.mark.asyncio
async def test_long_rows_hold_size_and_mtime(tmp_path: Path) -> None:
    (tmp_path / "f").write_bytes(b"12345")
    os.utime(tmp_path / "f", (0, 1_700_000_000))
    (tmp_path / "d").mkdir()
    os.utime(tmp_path / "d", (0, 1_700_000_000))
    when = time.strftime("%Y-%m-%d %H:%M", time.localtime(1_700_000_000))
    assert await _list(tmp_path, path=".", long=True) == (
        f"{(tmp_path / 'd').stat().st_size:>10}  {when}  d/\n         5  {when}  f"
    )


def test_a_broken_link_has_no_size_or_time(tmp_path: Path) -> None:
    (tmp_path / "dead").symlink_to(tmp_path / "nowhere")
    assert ls_row(tmp_path / "dead", name="dead") == "         0  ?  dead"


def test_name_sort_folds_case_and_breaks_ties_by_spelling(tmp_path: Path) -> None:
    paths = [tmp_path / n for n in ("b", "B", "a", "A")]
    sort_paths(paths, "name")
    assert [p.name for p in paths] == ["A", "a", "B", "b"]


@pytest.mark.asyncio
async def test_an_empty_directory_says_so(tmp_path: Path) -> None:
    (tmp_path / ".only-hidden").write_text("")
    assert await _list(tmp_path, path=".") == "(empty directory)"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("setup", "message"),
    [
        ("missing", "Not found: {}"),
        ("file", "Not a directory: {}"),
        ("locked", "Error reading {}: "),
    ],
)
async def test_a_path_that_lists_nothing_is_an_error(
    tmp_path: Path,
    setup: str,
    message: str,
) -> None:
    target = tmp_path / "t"
    if setup == "file":
        target.write_text("")
    elif setup == "locked":
        target.mkdir(mode=0)
    try:
        result = await List().call(ToolContext(cwd=tmp_path), {"path": "t"})
    finally:
        if setup == "locked":
            target.chmod(0o700)
    assert result.is_error is True
    assert result.call_id == ""
    assert result.text().startswith(message.format(target))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"path": ".", "keep_last": 0},
        {"path": ".", "max_results": 0},
        {"path": ".", "sort": "age"},
    ],
)
async def test_arguments_that_do_not_fit_are_refused(
    tmp_path: Path,
    arguments: dict[str, PlainTree],
) -> None:
    with pytest.raises(ValueError, match=r"argument|must be"):
        await List().call(ToolContext(cwd=tmp_path), arguments)


@pytest.mark.parametrize(
    ("command", "arguments"),
    [
        ("ls", {"path": "."}),
        ("ls src", {"path": "src"}),
        ("ls -- -odd", {"path": "-odd"}),
        ("ls -la", {"path": ".", "long": True, "show_hidden": True}),
        ("ls -A", {"path": ".", "show_hidden": True}),
        ("ls -1h", {"path": "."}),
        ("ls -r", {"path": ".", "sort": "name_desc"}),
        ("ls -t", {"path": ".", "sort": "mtime_desc"}),
        ("ls -tr", {"path": ".", "sort": "mtime"}),
        ("ls -S", {"path": ".", "sort": "size_desc"}),
        ("ls -S -r", {"path": ".", "sort": "size"}),
        ("ls | head", {"path": ".", "max_results": 10}),
        ("ls -t | head -n 5", {"path": ".", "sort": "mtime_desc", "max_results": 5}),
        ("ls | cat | head -3", {"path": ".", "max_results": 3}),
        ("ls | tail -n 3", {"path": ".", "keep_last": 3}),
        ("ls -t | tail -n 5", {"path": ".", "sort": "mtime_desc", "keep_last": 5}),
        ("ls ~", {"path": "~"}),
    ],
)
def test_a_plain_ls_becomes_the_list_that_shows_the_same_entries(
    command: str,
    arguments: dict[str, PlainTree],
) -> None:
    assert _from_shell(command) == arguments
    parse(arguments, List.Arguments)


def test_an_ls_after_a_cd_lists_where_the_cd_went() -> None:
    assert _from_shell("cd /srv && ls") == {"path": "/srv"}
    assert _from_shell("cd /srv && ls sub") == {"path": "/srv/sub"}


@pytest.mark.parametrize(
    "command",
    [
        "ls --color=auto",
        "ls -I '*.pyc' src",
        "ls -X",
        "ls -Z",
        "ls -R",
        "ls a b",
        "ls -tS",
        "ls *.py",
        "ls $D",
        "cd - && ls",
        "ls | head -c 5",
        "ls | head -nabc",
        "ls | head -0",
    ],
)
def test_an_ls_no_list_matches_is_refused_with_advice(command: str) -> None:
    assert _from_shell(command) == (
        "List one directory per List call; for a pattern, use Glob."
    )


@pytest.mark.parametrize(
    "command",
    ["ls | sort", "ls > out", "FOO=1 ls", "ls | wc -l", "echo hi"],
)
def test_a_command_that_is_no_plain_ls_is_not_lists_job(command: str) -> None:
    assert _from_shell(command) is None


async def _list(cwd: Path, **arguments: PlainTree) -> str:
    """Return what List shows for ``arguments`` in a fresh context at ``cwd``."""
    result = await List().call(ToolContext(cwd=cwd), arguments)
    assert result.call_id == ""
    assert result.is_error is None, result.text()
    return result.text()


def _from_shell(command: str) -> PlainTree:
    """Return what List makes of the first command in ``command``."""
    found = invocations(command)
    assert found is not None
    result = List().from_shell(found[0])
    return result if result is None or isinstance(result, str) else dict(result)


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
