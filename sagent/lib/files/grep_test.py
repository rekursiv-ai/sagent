"""Tests for ``files.grep``: ripgrep and the Python backend answer alike."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import contextlib
import itertools
import os
import pathlib
import shutil
import subprocess

import pytest

from sagent.lib.files.grep import (
    ARGV_CHARS,
    GrepError,
    OutputMode,
    Query,
    Row,
    _sort_key,
    grep,
    grep_rows,
    rg_command,
)


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


_RG_BASE = [
    "/rg",
    "--no-heading",
    "--with-filename",
    "--hidden",
    "--no-mmap",
    "--no-messages",
    "--field-match-separator=\x01",
    "--field-context-separator=\x02",
]
_RG_VCS = ["--glob", "!.git", "--glob", "!.svn", "--glob", "!.hg"]


@pytest.fixture(autouse=True)
def backend(rg: bool) -> Iterator[None]:
    """Hide ``rg`` unless the test runs against it."""
    found = shutil.which("rg")
    if rg and found is None:
        pytest.skip("ripgrep is not installed")
    with patch("shutil.which", return_value=found if rg else None):
        yield


@pytest.fixture
def rg() -> bool:
    """Default to the Python backend; ``_BOTH`` overrides per test."""
    return False


_BOTH = pytest.mark.parametrize("rg", [True, False], ids=["rg", "python"])


def _tree(root: Path) -> Path:
    """Make ``root`` a git root holding a small mixed tree."""
    (root / ".git").mkdir()
    (root / ".gitignore").write_text("build/\n")
    (root / "a.py").write_text("x\nhit one\ny\nhit two\n")
    (root / "b.txt").write_text("hit\n")
    (root / "sub").mkdir()
    (root / "sub" / "c.py").write_text("HIT\n")
    (root / ".hidden.py").write_text("hit\n")
    (root / "build").mkdir()
    (root / "build" / "gen.py").write_text("hit\n")
    return root


@_BOTH
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (Query(pattern="hit"), [".hidden.py", "a.py", "b.txt"]),
        (
            Query(pattern="hit", case_insensitive=True),
            [".hidden.py", "a.py", "b.txt", "sub/c.py"],
        ),
        (Query(pattern="hit", glob="*.py"), [".hidden.py", "a.py"]),
        (Query(pattern="hit", exclude="*.txt"), [".hidden.py", "a.py"]),
        (
            Query(pattern="hit", file_type="py", case_insensitive=True),
            [".hidden.py", "a.py", "sub/c.py"],
        ),
        (
            Query(pattern="hit", glob="*.{py,txt}", exclude=".hidden.py"),
            ["a.py", "b.txt"],
        ),
    ],
)
def test_files_with_matches_skip_ignored_trees_and_search_hidden(
    tmp_path: Path,
    query: Query,
    expected: list[str],
) -> None:
    root = _tree(tmp_path)
    assert grep(root, query) == [str(root / name) for name in expected]


@_BOTH
@pytest.mark.parametrize(
    ("mode", "rows"),
    [
        ("content", ["a.py:2:hit one", "a.py:4:hit two", "b.txt:1:hit"]),
        ("count", ["a.py:2", "b.txt:1"]),
    ],
)
def test_content_and_count_render_like_rg(
    tmp_path: Path,
    mode: OutputMode,
    rows: list[str],
) -> None:
    root = _tree(tmp_path)
    query = Query(pattern="hit", output_mode=mode, exclude=".hidden.py")
    assert grep(root, query) == [f"{root}/{row}" for row in rows]


@_BOTH
def test_overlapping_context_merges_into_one_group(
    tmp_path: Path,
) -> None:
    f = tmp_path / "f.txt"
    f.write_text("a\nHIT\nHIT\nb\nc\nd\nHIT\n")
    query = Query(
        pattern="HIT",
        output_mode="content",
        context_before=1,
        context_after=1,
    )
    assert grep(f, query) == [
        f"{f}-1-a",
        f"{f}:2:HIT",
        f"{f}:3:HIT",
        f"{f}-4-b",
        "--",
        f"{f}-6-d",
        f"{f}:7:HIT",
    ]


@_BOTH
@pytest.mark.parametrize("numbered", [True, False])
def test_rows_locate_their_first_match_in_characters(
    tmp_path: Path,
    numbered: bool,
) -> None:
    f = tmp_path / "f.txt"
    f.write_text("a\nééx hit hit\n")
    query = Query(
        pattern="hit",
        output_mode="content",
        context_before=1,
        line_numbers=numbered,
    )
    context = f"{f}-1-" if numbered else f"{f}-"
    prefix = f"{f}:2:" if numbered else f"{f}:"
    assert grep_rows(f, query) == [
        Row(text=f"{context}a", prefix_end=len(context)),
        Row(
            text=f"{prefix}ééx hit hit",
            prefix_end=len(prefix),
            column=len(prefix) + 4,
        ),
    ]


@_BOTH
def test_a_path_holding_a_separator_keeps_its_prefix_whole(tmp_path: Path) -> None:
    f = tmp_path / "a-1-b.txt"
    f.write_text("x:5: hit\n")
    query = Query(pattern="hit", output_mode="content")
    assert grep_rows(f, query) == [
        Row(
            text=f"{f}:1:x:5: hit",
            prefix_end=len(f"{f}:1:"),
            column=len(f"{f}:1:") + 5,
        ),
    ]


@_BOTH
def test_a_multiline_run_is_located_on_its_first_line(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("x foo\nbar\n  zz\n")
    query = Query(pattern=r"foo\nbar|zz", output_mode="content", multiline=True)
    assert [row.column for row in grep_rows(f, query)] == [len(f"{f}:1:") + 2, -1, -1]


@pytest.mark.parametrize("rg", [True], ids=["rg"])
def test_a_posix_class_is_located_by_ripgrep(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("-- abc123\n")
    query = Query(pattern="[[:alpha:]]+[[:digit:]]+", output_mode="content")
    prefix = f"{f}:1:"
    assert grep_rows(f, query) == [
        Row(text=f"{prefix}-- abc123", prefix_end=len(prefix), column=len(prefix) + 3),
    ]


@_BOTH
def test_line_numbers_off_drops_the_number(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("hit\n")
    query = Query(pattern="hit", output_mode="content", line_numbers=False)
    assert grep(f, query) == [f"{f}:hit"]


@_BOTH
def test_multiline_matches_across_lines(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("a\nb\n")
    assert grep(f, Query(pattern=r"a\nb", multiline=True)) == [str(f)]


@_BOTH
@pytest.mark.parametrize(
    ("query", "message"),
    [
        (Query(pattern="x", file_type="nosuch"), "unknown type"),
        (Query(pattern=r"a\nb"), "multiline is off"),
        (Query(pattern="("), "error"),
    ],
)
def test_a_bad_query_fails_in_every_backend(
    tmp_path: Path,
    query: Query,
    message: str,
) -> None:
    (tmp_path / "f.txt").write_text("x\n")
    with pytest.raises(GrepError, match=message):
        grep(tmp_path, query)


@_BOTH
def test_a_missing_path_is_an_error_not_no_matches(
    tmp_path: Path,
) -> None:
    with pytest.raises(GrepError) as raised:
        grep(tmp_path / "nope", Query(pattern="x"))
    assert str(raised.value) == f"no such file or directory: {tmp_path / 'nope'}"


@_BOTH
def test_lookaround_needs_no_flag(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("github.com/jv\njv\n")
    query = Query(pattern=r"(?<!com/)\bjv\b", output_mode="content")
    assert grep(f, query) == [f"{f}:2:jv"]


@_BOTH
def test_named_files_are_searched_in_order_whatever_their_name(
    tmp_path: Path,
) -> None:
    """A named file skips every walk filter: type, glob, ignore, hidden."""
    root = _tree(tmp_path)
    named = [root / "build" / "gen.py", root / "b.txt", root / ".hidden.py"]
    query = Query(pattern="hit", glob="*.md", exclude="*.txt", file_type="md")
    assert grep(named, query) == list(map(str, named))


@_BOTH
def test_a_directory_and_files_mix_and_an_empty_list_finds_nothing(
    tmp_path: Path,
) -> None:
    root = _tree(tmp_path)
    assert grep([root / "sub", root / "b.txt"], Query(pattern="(?i)hit")) == [
        str(root / "sub" / "c.py"),
        str(root / "b.txt"),
    ]
    assert grep([], Query(pattern="hit")) == []


@_BOTH
def test_a_walk_skips_binary_files_and_symlinks(tmp_path: Path) -> None:
    (tmp_path / "bin.dat").write_bytes(b"hit\n\0\n")
    (tmp_path / "ok.txt").write_text("hit\n")
    (tmp_path / "link.txt").symlink_to(tmp_path / "ok.txt")
    assert grep(tmp_path, Query(pattern="hit")) == [str(tmp_path / "ok.txt")]


@_BOTH
@pytest.mark.parametrize(
    ("mode", "rows"),
    [
        ("files_with_matches", ["{f}"]),
        ("count", ["{f}:2"]),
        ("content", ['{f}: binary file matches (found "\\0" byte around offset 4)']),
    ],
)
def test_a_named_binary_file_is_searched_but_never_printed(
    tmp_path: Path,
    mode: OutputMode,
    rows: list[str],
) -> None:
    f = tmp_path / "bin.dat"
    f.write_bytes(b"hit\n\0\nhit\n")
    query = Query(pattern="hit", output_mode=mode)
    assert grep(f, query) == [row.format(f=f) for row in rows]


@_BOTH
def test_text_searches_a_binary_file_line_by_line(tmp_path: Path) -> None:
    f = tmp_path / "bin.dat"
    f.write_bytes(b"x\0hit\n")
    query = Query(pattern="hit", output_mode="content", text=True)
    assert grep(f, query) == [f"{f}:1:x\0hit"]


@_BOTH
def test_bytes_that_are_not_utf8_match_lossily(tmp_path: Path) -> None:
    f = tmp_path / "latin.txt"
    f.write_bytes(b"calf\xe9 hit\n")
    query = Query(pattern="hit", output_mode="content")
    assert grep(f, query) == [f"{f}:1:calf\ufffd hit"]


@_BOTH
@pytest.mark.parametrize(
    "raw",
    [b"\xef\xbb\xbfhit\n", "hit\n".encode("utf-16")],
    ids=["utf8-bom", "utf16-bom"],
)
def test_a_byte_order_mark_picks_the_encoding(tmp_path: Path, raw: bytes) -> None:
    f = tmp_path / "bom.txt"
    f.write_bytes(raw)
    assert grep(f, Query(pattern="^hit$", output_mode="content")) == [f"{f}:1:hit"]


@_BOTH
def test_walked_files_sort_by_component_not_as_plain_text(tmp_path: Path) -> None:
    """``a/z.py`` precedes ``a-b/z.py``, though ``-`` sorts before ``/`` as text."""
    for directory in ("a", "a-b"):
        (tmp_path / directory).mkdir()
        (tmp_path / directory / "z.py").write_text("hit\n")
    assert grep(tmp_path, Query(pattern="hit")) == [
        str(tmp_path / "a" / "z.py"),
        str(tmp_path / "a-b" / "z.py"),
    ]


@_BOTH
def test_a_path_holding_a_colon_is_not_split_at_it(tmp_path: Path) -> None:
    """A count row reads ``name:count``, and the name may hold colons itself."""
    odd = tmp_path / "a:b.py"
    plain = tmp_path / "c.py"
    odd.write_text("hit\n")
    plain.write_text("hit\nhit\n")
    query = Query(pattern="hit", output_mode="count")
    assert grep([odd, plain], query) == [f"{odd}:1", f"{plain}:2"]


@_BOTH
def test_a_line_holding_the_field_separators_survives_them(tmp_path: Path) -> None:
    r"""``rg``'s own ``\x01``/``\x02`` can appear in a binary searched as text."""
    f = tmp_path / "b.dat"
    f.write_bytes(b"pre\x01hit\n" + b"two\x02hit\n")
    query = Query(pattern="hit", output_mode="content", text=True)
    assert grep(f, query) == [f"{f}:1:pre\x01hit", f"{f}:2:two\x02hit"]


@_BOTH
def test_a_leading_blank_line_still_numbers_from_one(tmp_path: Path) -> None:
    """A newline at offset 0 starts line 2, and line 1 is the empty line."""
    f = tmp_path / "f.txt"
    f.write_text("\nhit\n")
    query = Query(pattern="hit", output_mode="content")
    assert grep(f, query) == [f"{f}:2:hit"]
    blank = Query(pattern="^$", output_mode="content")
    assert grep(f, blank) == [f"{f}:1:"]
    # Two blank lines in a row: an empty line must not swallow the rest of the
    # text as one line, which would hide every blank line after the first.
    f.write_text("\n\nhit\n")
    assert grep(f, blank) == [f"{f}:1:", f"{f}:2:"]


@_BOTH
@pytest.mark.parametrize("pattern", ["$", "x*"], ids=["anchor", "star"])
def test_a_zero_width_pattern_counts_every_line_once(
    tmp_path: Path,
    pattern: str,
) -> None:
    """An empty match past the final newline sits on no line at all."""
    f = tmp_path / "f.txt"
    f.write_text("a\nb\n")
    query = Query(pattern=pattern, output_mode="count")
    assert grep(f, query) == [f"{f}:2"]


@_BOTH
def test_a_class_matching_a_newline_does_not_hide_a_later_line(
    tmp_path: Path,
) -> None:
    """A cross-line hit is no match; the scan resumes and finds the real one."""
    f = tmp_path / "f.txt"
    f.write_text("a\nb\nazb\n")
    query = Query(pattern="a[^x]b", output_mode="content")
    assert grep(f, query) == [f"{f}:3:azb"]


@_BOTH
def test_only_newline_ends_a_line(tmp_path: Path) -> None:
    f = tmp_path / "cr.txt"
    f.write_bytes(b"a\rhit\r\nb\x0bhit")
    assert grep(f, Query(pattern="hit", output_mode="content")) == [
        f"{f}:1:a\rhit\r",
        f"{f}:2:b\x0bhit",
    ]


@_BOTH
def test_multiline_reports_every_line_a_match_spans_once(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("a trax\nissue 1234 trax\nissue 9\nzz\n")
    content = Query(pattern=r"trax\s+issue", multiline=True, output_mode="content")
    assert grep(f, content) == [
        f"{f}:1:a trax",
        f"{f}:2:issue 1234 trax",
        f"{f}:3:issue 9",
    ]
    count = Query(pattern=r"trax\s+issue", multiline=True, output_mode="count")
    assert grep(f, count) == [f"{f}:2"]


@_BOTH
def test_anchors_match_at_every_line(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("ab\ncd\n")
    query = Query(pattern="^cd$", multiline=True, output_mode="content")
    assert grep(f, query) == [f"{f}:2:cd"]


@_BOTH
def test_context_groups_of_different_files_are_separated(tmp_path: Path) -> None:
    one, two = tmp_path / "1.txt", tmp_path / "2.txt"
    one.write_text("a\nhit\n")
    two.write_text("hit\nb\n")
    query = Query(pattern="hit", output_mode="content", context_before=1)
    assert grep([one, two], query) == [
        f"{one}-1-a",
        f"{one}:2:hit",
        "--",
        f"{two}:1:hit",
    ]


@_BOTH
def test_a_missing_named_file_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("x\n")
    with pytest.raises(GrepError, match="no such file or directory"):
        grep([tmp_path / "f.txt", tmp_path / "nope"], Query(pattern="x"))


@pytest.mark.parametrize("rg", [True], ids=["rg"])
def test_named_files_split_where_the_next_would_pass_the_argv_limit(
    tmp_path: Path,
) -> None:
    """Thousands of named files must not overflow one command line."""
    names = [tmp_path / f"{i:04}{'n' * 200}.txt" for i in range(1_200)]
    for path in names:
        path.write_text("hit\n")
    with patch(
        "sagent.lib.files.grep.subprocess.run",
        wraps=subprocess.run,
    ) as run:
        assert grep(names, Query(pattern="hit")) == list(map(str, names))
    batches = [
        [arg for arg in cast(list[str], call.args[0]) if arg.startswith(str(tmp_path))]
        for call in run.call_args_list
    ]
    assert [name for batch in batches for name in batch] == list(map(str, names))
    assert len(batches) > 1
    # Every batch, not just the first, is bounded by the caller's timeout.
    assert [call.kwargs["timeout"] for call in run.call_args_list] == [30] * len(
        batches,
    )
    for batch, after in itertools.pairwise(batches):
        assert len(" ".join(batch)) <= ARGV_CHARS
        assert len(" ".join([*batch, after[0]])) > ARGV_CHARS


@pytest.mark.parametrize("rg", [True], ids=["rg"])
def test_each_rg_run_gets_the_timeout(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("ab\n")
    with patch(
        "sagent.lib.files.grep.subprocess.run",
        wraps=subprocess.run,
    ) as run:
        assert grep(f, Query(pattern="(?<=a)b"), timeout_sec=7) == [str(f)]
    assert [call.kwargs for call in run.call_args_list] == [
        {"capture_output": True, "timeout": 7, "check": False},
    ] * 2
    assert "-P" in cast(list[str], run.call_args_list[1].args[0])


@pytest.mark.parametrize("rg", [True], ids=["rg"])
@pytest.mark.parametrize(
    "query",
    [
        Query(pattern="value", output_mode="count"),
        Query(pattern=r"async def t.*\n(.*\n){0,25}.*main\(", multiline=True),
    ],
    ids=["literal", "backtracking-multiline"],
)
def test_rg_answers_every_search_when_installed(tmp_path: Path, query: Query) -> None:
    r"""Python ``re`` backtracks without bound, and holds the GIL while it does.

    ``async def test_.*\\n(.*\\n){0,25}.*main\\._main\\(`` with ``multiline`` ran
    past 30 s over a 99 KB test file that ``rg`` searched in 13 ms, and froze the
    sagent process that asked. A 16 KB sample timed beforehand missed it: the
    blowup began after the sample ended.
    """
    f = tmp_path / "f.py"
    f.write_text("value = 1\n")
    with patch(
        "sagent.lib.files.grep.subprocess.run",
        wraps=subprocess.run,
    ) as run:
        grep(f, query)
    assert run.called


def test_a_file_vanishing_mid_walk_is_skipped_not_a_walk_stopper(
    tmp_path: Path,
) -> None:
    """A race must not truncate the file list the walk hands to the search.

    The first file walked is failed rather than a named one, because the walk
    order is rignore's: failing a file that happened to be last would let a
    walk that stopped dead look like one that skipped it.
    """
    for name in ("a.py", "b.py", "c.py"):
        (tmp_path / name).write_text("hit\n")
    real = pathlib.Path.lstat
    seen: list[object] = []

    def flaky(self: pathlib.Path) -> os.stat_result:
        seen.append(self)
        # The walk lists its start directory first, so the second call is the
        # first file.
        if len(seen) == 2:
            raise OSError
        return real(self)

    with patch("pathlib.Path.lstat", flaky):
        found = grep(tmp_path, Query(pattern="hit"))
    every = {str(tmp_path / name) for name in ("a.py", "b.py", "c.py")}
    assert set(found) < every
    assert len(found) == 2


@pytest.mark.parametrize("rg", [True], ids=["rg"])
def test_the_default_rg_timeout_is_thirty_seconds(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("x\n")
    with (
        patch("sagent.lib.files.grep.shutil.which", return_value="/rg") as which,
        patch(
            "sagent.lib.files.grep.subprocess.run",
            return_value=subprocess.CompletedProcess([], 1, b"", b""),
        ) as run,
    ):
        assert grep(f, Query(pattern="x")) == []
    which.assert_called_once_with("rg")
    assert cast(list[str], run.call_args.args[0])[0] == "/rg"
    assert run.call_args.kwargs["timeout"] == 30


@pytest.mark.parametrize("rg", [True], ids=["rg"])
def test_a_rg_failure_names_its_exit_and_stderr(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("x\n")
    with (
        patch("sagent.lib.files.grep.shutil.which", return_value="/rg"),
        patch(
            "sagent.lib.files.grep.subprocess.run",
            return_value=subprocess.CompletedProcess([], 2, b"", b"bad\xff\n"),
        ),
        pytest.raises(GrepError) as raised,
    ):
        grep(f, Query(pattern="x"))
    assert str(raised.value) == "ripgrep error (exit 2): bad\ufffd"


@pytest.mark.parametrize("rg", [True], ids=["rg"])
def test_a_rg_exit_without_a_message_keeps_what_it_found(tmp_path: Path) -> None:
    """``--no-messages`` silences per-file failures; their exit 2 is no error."""
    f = tmp_path / "f.txt"
    f.write_text("x\n")
    with (
        patch("sagent.lib.files.grep.shutil.which", return_value="/rg"),
        patch(
            "sagent.lib.files.grep.subprocess.run",
            return_value=subprocess.CompletedProcess([], 2, f"{f}\n".encode(), b" \n"),
        ),
    ):
        assert grep(f, Query(pattern="x")) == [str(f)]


@_BOTH
def test_a_nul_breaks_a_line_and_only_a_multiline_dot_crosses_it(
    tmp_path: Path,
) -> None:
    """A NUL at offset 0 still marks a file binary; one search holds it whole."""
    f = tmp_path / "bin.dat"
    f.write_bytes(b"\0x\0y\nhit\n")
    assert grep(f, Query(pattern="x.y", output_mode="count")) == []
    assert grep(f, Query(pattern="^y|xy|x$", output_mode="count")) == [f"{f}:2"]
    crossed = Query(pattern="x.y", multiline=True, output_mode="count")
    assert grep(f, crossed) == [f"{f}:1"]
    anchored = Query(pattern="^y|xy|x$", multiline=True, output_mode="count")
    assert grep(f, anchored) == []
    assert grep(f, Query(pattern="hit", output_mode="content")) == [
        f'{f}: binary file matches (found "\\0" byte around offset 0)',
    ]
    assert grep(tmp_path, Query(pattern="hit")) == []


@_BOTH
def test_a_big_endian_utf16_file_is_read(tmp_path: Path) -> None:
    f = tmp_path / "be.txt"
    f.write_bytes(b"\xfe\xff" + "hit\n".encode("utf-16-be"))
    assert grep(f, Query(pattern="^hit$", output_mode="content")) == [f"{f}:1:hit"]


@_BOTH
def test_dot_crosses_lines_and_case_folds_only_when_asked(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("a\nb\nA\n")
    assert grep(f, Query(pattern="a.b", multiline=True)) == [str(f)]
    assert grep(f, Query(pattern="a.b")) == []
    assert grep(f, Query(pattern="A", output_mode="count")) == [f"{f}:1"]


@_BOTH
def test_a_multiline_match_around_newlines_keeps_only_its_lines(
    tmp_path: Path,
) -> None:
    f = tmp_path / "f.txt"
    f.write_text("x\nab\ncd\n")
    around = Query(pattern=r"\nab\n", multiline=True, output_mode="content")
    assert grep(f, around) == [f"{f}:1:x", f"{f}:2:ab"]
    ends = Query(pattern="$", multiline=True, output_mode="count")
    assert grep(f, ends) == [f"{f}:3"]


@_BOTH
def test_a_multiline_match_ending_in_blank_lines_and_spaces(tmp_path: Path) -> None:
    """Only trailing newlines are dropped from a span, never other whitespace."""
    f = tmp_path / "f.txt"
    f.write_text("a \n\n\nb\nc\n")
    query = Query(pattern=r"a \n\n", multiline=True, output_mode="content")
    assert grep(f, query) == [f"{f}:1:a ", f"{f}:2:"]
    lead = Query(pattern=r"\n\nb", multiline=True, output_mode="content")
    assert grep(f, lead) == [f"{f}:2:", f"{f}:3:", f"{f}:4:b"]


@_BOTH
def test_context_groups_touching_end_to_end_merge(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("hit\na\nb\nhit\n")
    both = Query(
        pattern="hit",
        output_mode="content",
        context_after=1,
        context_before=1,
    )
    assert grep(f, both) == [f"{f}:1:hit", f"{f}-2-a", f"{f}-3-b", f"{f}:4:hit"]
    after = Query(pattern="hit", output_mode="content", context_after=1)
    assert grep(f, after) == [f"{f}:1:hit", f"{f}-2-a", "--", f"{f}:4:hit"]


@_BOTH
def test_a_bom_file_keeps_its_invalid_bytes_as_replacements(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_bytes(b"\xef\xbb\xbfhit \xff\n")
    query = Query(pattern="hit", output_mode="content")
    assert grep(f, query) == [f"{f}:1:hit \ufffd"]


@_BOTH
def test_a_named_binary_file_matches_across_a_nul_only_when_multiline(
    tmp_path: Path,
) -> None:
    f = tmp_path / "bin.dat"
    f.write_bytes(b"XX\0XX\n")
    assert grep(f, Query(pattern="XX.XX")) == []
    assert grep(f, Query(pattern="XX.XX", multiline=True)) == [str(f)]


@_BOTH
@pytest.mark.parametrize("multiline", [False, True])
def test_case_folding_keeps_line_anchors(tmp_path: Path, multiline: bool) -> None:
    f = tmp_path / "f.txt"
    f.write_text("x\nHIT\n")
    query = Query(
        pattern="^hit$",
        case_insensitive=True,
        multiline=multiline,
        output_mode="count",
    )
    assert grep(f, query) == [f"{f}:1"]


@_BOTH
@pytest.mark.parametrize(
    ("pattern", "count"),
    [
        (r"a\nb", 1),
        (r"a\nb|d", 2),
        (r"a\nb|c", 2),
        (r"a\nb|b\nc", 1),
        ("b|c", 2),
        ("b|b", 1),
        ("[ab]", 2),
        (r"b\n|c\n", 2),
        (r"\nb", 1),
    ],
)
def test_multiline_count_is_of_matches_when_one_crosses_a_line(
    tmp_path: Path,
    pattern: str,
    count: int,
) -> None:
    """Else, as ripgrep searches such a pattern line by line, of lines."""
    f = tmp_path / "f.txt"
    f.write_text("a\nb\nc\nd\n")
    query = Query(pattern=pattern, multiline=True, output_mode="count")
    assert grep(f, query) == [f"{f}:{count}"]


@_BOTH
@pytest.mark.parametrize(
    ("pattern", "count"),
    [("x", 1), (r"x|\ny", 3), (r"x|y\n", 3), (r"x\n", 1)],
)
def test_multiline_count_of_two_matches_on_one_line(
    tmp_path: Path,
    pattern: str,
    count: int,
) -> None:
    f = tmp_path / "f.txt"
    f.write_text("xx\ny\n")
    query = Query(pattern=pattern, multiline=True, output_mode="count")
    assert grep(f, query) == [f"{f}:{count}"]


@pytest.mark.parametrize("rg", [True], ids=["rg"])
def test_a_batch_may_fill_the_argv_limit_exactly(tmp_path: Path) -> None:
    """Names joined by spaces may total ``ARGV_CHARS`` and still share a run."""
    names = [tmp_path / f"{i:04}{'n' * 200}.txt" for i in range(1_200)]
    widths = [len(str(path)) + 1 for path in names]
    fit = next(k for k in range(len(names)) if sum(widths[: k + 1]) - 1 > ARGV_CHARS)
    # Lengthen the first ``fit`` names, a few characters each, until they
    # total exactly ``ARGV_CHARS``; the next name then cannot fit.
    pad = ARGV_CHARS - (sum(widths[:fit]) - 1)
    for i in range(fit):
        extra = pad // fit + (i < pad % fit)
        names[i] = names[i].with_name(f"{names[i].stem}{'p' * extra}.txt")
    for path in names:
        path.write_text("hit\n")
    assert len(" ".join(map(str, names[:fit]))) == ARGV_CHARS
    with patch(
        "sagent.lib.files.grep.subprocess.run",
        wraps=subprocess.run,
    ) as run:
        assert grep(names, Query(pattern="hit")) == list(map(str, names))
    first = [
        arg
        for arg in cast(list[str], run.call_args_list[0].args[0])
        if arg.startswith(str(tmp_path))
    ]
    assert len(first) == fit


def test_pcre_without_rg_names_the_reason_in_full(tmp_path: Path) -> None:
    with pytest.raises(GrepError) as raised:
        grep(tmp_path, Query(pattern="x", pcre=True))
    assert str(raised.value) == (
        "pcre=true requires ripgrep, which is not installed;"
        " the Python fallback cannot provide PCRE2 semantics."
    )


@pytest.mark.parametrize("rg", [True], ids=["rg"])
def test_batches_of_context_rows_are_separated_like_files(tmp_path: Path) -> None:
    names = [tmp_path / f"{i:04}{'n' * 200}.txt" for i in range(1_200)]
    for path in names:
        path.write_text("a\nhit\n")
    query = Query(pattern="hit", output_mode="content", context_before=1)
    rows = grep(names, query)
    assert rows.count("--") == len(names) - 1
    assert rows[:3] == [f"{names[0]}-1-a", f"{names[0]}:2:hit", "--"]


@_BOTH
def test_a_directory_named_with_a_trailing_slash(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("hit\n")
    named = type(tmp_path)(f"{tmp_path}/")
    assert grep(named, Query(pattern="hit")) == [f"{tmp_path}/f.txt"]


def test_pcre_without_rg_is_refused(tmp_path: Path) -> None:
    with pytest.raises(GrepError, match="pcre=true requires ripgrep"):
        grep(tmp_path, Query(pattern="(?<=a)b", pcre=True))


@pytest.mark.parametrize("rg", [True], ids=["rg"])
def test_a_rg_timeout_is_a_grep_error(tmp_path: Path) -> None:
    def timeout(*_args: object, **_kwargs: object) -> object:
        raise subprocess.TimeoutExpired(cmd="rg", timeout=1)

    with (
        patch("shutil.which", return_value="/rg"),
        patch("subprocess.run", timeout),
        pytest.raises(GrepError, match=r"^ripgrep timed out after 1s"),
    ):
        grep(tmp_path, Query(pattern="x"), timeout_sec=1)


def test_rg_command_with_every_knob_off(tmp_path: Path) -> None:
    query = Query(pattern="pat", output_mode="content", line_numbers=False)
    assert rg_command("/rg", [tmp_path, tmp_path / "f"], query) == [
        *_RG_BASE,
        *_RG_VCS,
        *("--column", "--", "pat", str(tmp_path), str(tmp_path / "f")),
    ]


def test_rg_command_with_every_knob_on(tmp_path: Path) -> None:
    query = Query(
        pattern="pat",
        glob="*.py",
        exclude="x_*",
        file_type="py",
        output_mode="files_with_matches",
        context_before=1,
        context_after=2,
        case_insensitive=True,
        multiline=True,
        pcre=True,
        text=True,
        word=True,
        whole_line=True,
        fixed=True,
        search_ignored=True,
        follow=True,
        max_depth=3,
    )
    assert rg_command("/rg", [tmp_path], query) == [
        *_RG_BASE,
        *_RG_VCS,
        *("--glob", "*.py", "--glob", "!x_*"),
        *("-n", "-i", "-U", "--multiline-dotall", "-l"),
        *("-B", "1", "-A", "2"),
        *("--type-clear", "py", "--type-add", "py:*.py", "--type", "py"),
        *("--text", "-P"),
        *("-w", "-x", "-F", "--no-ignore", "--follow", "--max-depth", "3"),
        *("--", "pat", str(tmp_path)),
    ]


def test_rg_command_for_line_selection_knobs(tmp_path: Path) -> None:
    """An inverted line has no match to locate, and ripgrep's columns then lie."""
    query = Query(pattern="pat", output_mode="content", invert=True, max_count=2)
    assert rg_command("/rg", [tmp_path], query) == [
        *_RG_BASE,
        *_RG_VCS,
        *("-n", "-v", "--max-count", "2", "--", "pat", str(tmp_path)),
    ]
    only = Query(pattern="pat", output_mode="content", only_matching=True)
    assert rg_command("/rg", [tmp_path], only) == [
        *_RG_BASE,
        *_RG_VCS,
        *("-n", "--column", "-o", "--", "pat", str(tmp_path)),
    ]
    without = Query(pattern="pat", output_mode="files_without_match")
    assert rg_command("/rg", [tmp_path], without) == [
        *_RG_BASE,
        *_RG_VCS,
        *("-n", "--files-without-match", "--", "pat", str(tmp_path)),
    ]


def test_rg_command_count_mode(tmp_path: Path) -> None:
    query = Query(pattern="pat", output_mode="count", line_numbers=False)
    assert rg_command("/rg", [tmp_path], query) == [
        *_RG_BASE,
        *_RG_VCS,
        *("-c", "--", "pat", str(tmp_path)),
    ]


@_BOTH
@pytest.mark.parametrize("file_type", ["py", "js", "cpp", "md", "sh"])
def test_file_types_use_the_declared_globs(tmp_path: Path, file_type: str) -> None:
    for name in (
        "a.py",
        "b.pyi",
        "a.js",
        "b.cjs",
        "a.cpp",
        "b.hh",
        "a.md",
        "b.markdown",
        "a.sh",
        "b.zsh",
    ):
        (tmp_path / name).write_text("hit\n")
    assert grep(tmp_path, Query(pattern="hit", file_type=file_type)) == [
        str(tmp_path / f"a.{file_type}"),
    ]


@_BOTH
@pytest.mark.parametrize("pattern", [r"a\nb\]", r"a\nb[xy]", r"a\nb\[\]"])
def test_newline_escapes_outside_classes_are_rejected(
    tmp_path: Path,
    pattern: str,
) -> None:
    file = tmp_path / "f"
    file.write_text("a\nb]\n")
    with pytest.raises(GrepError, match="multiline is off"):
        grep(file, Query(pattern=pattern))


@_BOTH
@pytest.mark.parametrize("mode", ["content", "count", "files_with_matches"])
def test_empty_multiline_matches_do_not_invent_a_line(
    tmp_path: Path,
    mode: OutputMode,
) -> None:
    file = tmp_path / "empty"
    file.touch()
    assert grep(file, Query(pattern="", multiline=True, output_mode=mode)) == []


@_BOTH
@pytest.mark.parametrize(("pattern", "numbered"), [("-l", True), ("-n", False)])
def test_patterns_that_spell_flags_do_not_change_output(
    tmp_path: Path,
    pattern: str,
    numbered: bool,
) -> None:
    file = tmp_path / "f"
    file.write_text(f"a {pattern} b\n")
    query = Query(pattern=pattern, output_mode="content", line_numbers=numbered)
    assert grep(file, query) == [f"{file}:{'1:' if numbered else ''}a {pattern} b"]


@_BOTH
def test_a_raw_newline_in_the_pattern_needs_multiline(tmp_path: Path) -> None:
    file = tmp_path / "f"
    file.write_text("a\nb\n")
    with pytest.raises(GrepError, match="multiline is off"):
        grep(file, Query(pattern="a\nb"))


def test_an_unreadable_file_is_skipped(tmp_path: Path) -> None:
    for name in ("a.txt", "b.txt"):
        (tmp_path / name).write_text("hit\n")
    real = pathlib.Path.read_bytes

    def unreadable(self: pathlib.Path) -> bytes:
        if self.name == "a.txt":
            raise PermissionError(self)
        return real(self)

    with patch("pathlib.Path.read_bytes", unreadable):
        assert grep(tmp_path, Query(pattern="hit")) == [str(tmp_path / "b.txt")]


@pytest.mark.parametrize("rg", [True], ids=["rg"])
def test_ripgrep_only_syntax_under_a_relative_path(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("hit\n")
    with contextlib.chdir(tmp_path):
        found = grep(
            pathlib.Path("f.txt"),
            Query(pattern=r"\Ahit", output_mode="content"),
        )
    assert found == ["f.txt:1:hit"]


@pytest.mark.parametrize("rg", [True], ids=["rg"])
def test_an_invalid_pattern_gets_rgs_error(tmp_path: Path) -> None:
    file = tmp_path / "f.txt"
    file.write_text("hit\n")
    with pytest.raises(GrepError, match="ripgrep error \\(exit 2\\)"):
        grep(file, Query(pattern="("))


_LINES = "foo bar\nfoobar\nbaz\nfoo foo\n"


def _rows(f: Path, *specs: tuple[str, str, int]) -> list[Row]:
    """Build rows of ``f`` from ``(prefix after the name, text, match column)``."""
    out: list[Row] = []
    for head, text, column in specs:
        prefix = f"{f}{head}"
        out.append(
            Row(
                text=prefix + text,
                prefix_end=len(prefix),
                column=len(prefix) + column if column >= 0 else -1,
            ),
        )
    return out


@_BOTH
@pytest.mark.parametrize(
    ("mode", "rows"),
    [
        ("content", ["{f}:3:baz", "{g}:1:nothing"]),
        ("count", ["{f}:1", "{g}:1"]),
        ("files_with_matches", ["{f}", "{g}"]),
        ("files_without_match", []),
    ],
)
def test_invert_selects_the_lines_that_do_not_match(
    tmp_path: Path,
    mode: OutputMode,
    rows: list[str],
) -> None:
    f, g = tmp_path / "f", tmp_path / "g"
    f.write_text(_LINES)
    g.write_text("nothing\n")
    query = Query(pattern="foo", output_mode=mode, invert=True)
    assert grep([f, g], query) == [row.format(f=f, g=g) for row in rows]


@_BOTH
def test_an_inverted_line_has_no_match_to_locate(tmp_path: Path) -> None:
    f = tmp_path / "f"
    f.write_text(_LINES)
    query = Query(pattern="foo", output_mode="content", invert=True)
    assert grep_rows(f, query) == _rows(f, (":3:", "baz", -1))


@_BOTH
def test_inverted_context_is_the_matching_lines(tmp_path: Path) -> None:
    f = tmp_path / "f"
    f.write_text(_LINES)
    query = Query(pattern="baz", output_mode="content", invert=True, context_before=1)
    assert grep(f, query) == [
        f"{f}:1:foo bar",
        f"{f}:2:foobar",
        f"{f}-3-baz",
        f"{f}:4:foo foo",
    ]


@_BOTH
def test_word_needs_no_word_character_beside_the_match(tmp_path: Path) -> None:
    f, h = tmp_path / "f", tmp_path / "h"
    f.write_text(_LINES)
    h.write_text("foo\nao-x b\n")
    word = Query(pattern="foo", output_mode="content", word=True)
    assert grep_rows(f, word) == _rows(f, (":1:", "foo bar", 0), (":4:", "foo foo", 0))
    assert grep(h, Query(pattern="-x", word=True)) == []
    alternation = Query(pattern="foo|ao", output_mode="count", word=True)
    assert grep(h, alternation) == [f"{h}:2"]


@_BOTH
def test_whole_line_wraps_the_whole_pattern(tmp_path: Path) -> None:
    h, h2 = tmp_path / "h", tmp_path / "h2"
    h.write_text("foo\nao-x b\n")
    h2.write_text("foo foo\nfoo\n")
    line = Query(pattern="foo|ao", output_mode="content", whole_line=True)
    assert grep(h, line) == [f"{h}:1:foo"]
    both = Query(pattern="foo", output_mode="content", whole_line=True, word=True)
    assert grep(h2, both) == [f"{h2}:2:foo"]


@_BOTH
def test_fixed_reads_the_pattern_as_text(tmp_path: Path) -> None:
    f = tmp_path / "f"
    f.write_text("a.b\naxb\nx\\ny (\n")
    assert grep(f, Query(pattern="a.b", output_mode="content", fixed=True)) == [
        f"{f}:1:a.b",
    ]
    assert grep(f, Query(pattern="x\\ny (", output_mode="count", fixed=True)) == [
        f"{f}:1",
    ]
    folded = Query(
        pattern="A.B",
        output_mode="count",
        fixed=True,
        case_insensitive=True,
    )
    assert grep(f, folded) == [f"{f}:1"]


@_BOTH
def test_fixed_text_holding_a_newline_needs_multiline(tmp_path: Path) -> None:
    f = tmp_path / "f"
    f.write_text(_LINES)
    with pytest.raises(GrepError, match="multiline is off"):
        grep(f, Query(pattern="bar\nfoo", fixed=True))
    assert grep(f, Query(pattern="bar\nfoo", fixed=True, multiline=True)) == [str(f)]


@_BOTH
def test_only_matching_prints_each_match_as_its_own_row(tmp_path: Path) -> None:
    """Each row's text is its match, so the match begins where the text does."""
    f = tmp_path / "f"
    f.write_text(_LINES)
    query = Query(
        pattern="FOO",
        output_mode="content",
        only_matching=True,
        case_insensitive=True,
    )
    assert grep_rows(f, query) == _rows(
        f,
        (":1:", "foo", 0),
        (":2:", "foo", 0),
        (":4:", "foo", 0),
        (":4:", "foo", 0),
    )
    count = Query(pattern="foo", output_mode="count", only_matching=True)
    assert grep(f, count) == [f"{f}:4"]


@_BOTH
def test_only_matching_skips_an_empty_match_where_one_just_ended(
    tmp_path: Path,
) -> None:
    """Ripgrep's engine, unlike Python's, finds no empty match at a match's end."""
    h = tmp_path / "h"
    h.write_text("foo\nao-x b\n")
    query = Query(pattern="o*", output_mode="content", only_matching=True)
    assert grep(h, query) == [
        f"{h}:1:",
        f"{h}:1:oo",
        *(f"{h}:2:{text}" for text in ("", "o", "", "", "", "")),
    ]
    count = Query(pattern="o*", output_mode="count", only_matching=True)
    assert grep(h, count) == [f"{h}:8"]


@_BOTH
def test_only_matching_keeps_context_lines_whole(tmp_path: Path) -> None:
    f = tmp_path / "f"
    f.write_text(_LINES)
    query = Query(
        pattern="foo",
        output_mode="content",
        only_matching=True,
        context_after=1,
    )
    assert grep(f, query) == [
        f"{f}:1:foo",
        f"{f}:2:foo",
        f"{f}-3-baz",
        f"{f}:4:foo",
        f"{f}:4:foo",
    ]


@_BOTH
def test_max_count_stops_but_shows_a_match_in_its_trailing_context(
    tmp_path: Path,
) -> None:
    f = tmp_path / "f"
    f.write_text(_LINES)
    query = Query(pattern="foo", output_mode="content", max_count=1, context_after=2)
    assert grep_rows(f, query) == _rows(
        f,
        (":1:", "foo bar", 0),
        (":2:", "foobar", 0),
        ("-3-", "baz", -1),
    )
    count = Query(pattern="foo", output_mode="count", max_count=1)
    assert grep(f, count) == [f"{f}:1"]
    inverted = Query(
        pattern="baz",
        output_mode="content",
        invert=True,
        max_count=1,
        context_after=2,
    )
    assert grep(f, inverted) == [f"{f}:1:foo bar", f"{f}:2:foobar", f"{f}-3-baz"]


@_BOTH
def test_max_count_counts_lines_not_matches(tmp_path: Path) -> None:
    h2 = tmp_path / "h2"
    h2.write_text("foo foo\nfoo\n")
    query = Query(pattern="foo", output_mode="content", only_matching=True, max_count=1)
    assert grep(h2, query) == [f"{h2}:1:foo", f"{h2}:1:foo"]
    count = Query(pattern="foo", output_mode="count", only_matching=True, max_count=1)
    assert grep(h2, count) == [f"{h2}:2"]


@_BOTH
def test_files_without_match_lists_what_a_walk_finds_no_match_in(
    tmp_path: Path,
) -> None:
    """A walked binary file is skipped outright, so it is not listed either."""
    (tmp_path / "bin").write_bytes(b"x\0zzz\n")
    (tmp_path / "e").touch()
    (tmp_path / "s").write_text("foo\n")
    (tmp_path / "t").write_text("no\n")
    query = Query(pattern="foo", output_mode="files_without_match")
    assert grep(tmp_path, query) == [str(tmp_path / "e"), str(tmp_path / "t")]


@_BOTH
def test_search_ignored_walks_what_gitignore_names(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    query = Query(pattern="hit", search_ignored=True)
    expected = [".hidden.py", "a.py", "b.txt", "build/gen.py"]
    assert grep(root, query) == [str(root / name) for name in expected]


@_BOTH
def test_follow_walks_through_symlinks_and_past_broken_ones(tmp_path: Path) -> None:
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "x.txt").write_text("hit\n")
    (tmp_path / "dir").symlink_to(tmp_path / "real")
    (tmp_path / "file.txt").symlink_to(tmp_path / "real" / "x.txt")
    (tmp_path / "broken").symlink_to(tmp_path / "nowhere")
    (tmp_path / "real" / "up").symlink_to(tmp_path)
    assert grep(tmp_path, Query(pattern="hit")) == [str(tmp_path / "real" / "x.txt")]
    assert grep(tmp_path, Query(pattern="hit", follow=True)) == [
        str(tmp_path / "dir" / "x.txt"),
        str(tmp_path / "file.txt"),
        str(tmp_path / "real" / "x.txt"),
    ]


@_BOTH
def test_max_depth_bounds_the_walk(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    shallow = Query(pattern="hit", case_insensitive=True, max_depth=1)
    assert grep(root, shallow) == [
        str(root / n) for n in (".hidden.py", "a.py", "b.txt")
    ]
    deep = Query(pattern="hit", case_insensitive=True, max_depth=2)
    assert grep(root, deep)[-1] == str(root / "sub" / "c.py")


@_BOTH
@pytest.mark.parametrize(
    "query",
    [
        Query(pattern="x", multiline=True, invert=True),
        Query(pattern="x", multiline=True, only_matching=True),
        Query(pattern="x", multiline=True, max_count=1),
        Query(pattern="x", invert=True, only_matching=True),
    ],
    ids=["multiline-invert", "multiline-only", "multiline-max", "invert-only"],
)
def test_combinations_ripgrep_answers_inconsistently_are_refused(
    tmp_path: Path,
    query: Query,
) -> None:
    (tmp_path / "f").write_text("x\n")
    with pytest.raises(GrepError, match="cannot combine"):
        grep(tmp_path, query)


@_BOTH
def test_an_unreadable_walked_file_is_skipped_in_every_backend(
    tmp_path: Path,
) -> None:
    if os.geteuid() == 0:
        pytest.skip("root reads every file")
    (tmp_path / "a.txt").write_text("hit\n")
    (tmp_path / "b.txt").write_text("hit\n")
    (tmp_path / "a.txt").chmod(0)
    assert grep(tmp_path, Query(pattern="hit")) == [str(tmp_path / "b.txt")]


def test_a_name_outside_every_path_sorts_last() -> None:
    paths = [pathlib.Path("a"), pathlib.Path("b/")]
    order = {"a": 0}
    keys = [_sort_key(name, paths, order) for name in ("z", "b/c", "a")]
    assert keys == [(2, ("z",)), (1, ("c",)), (0, ())]


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
