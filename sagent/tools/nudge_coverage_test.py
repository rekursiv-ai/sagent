"""Coverage matrix for tool-use nudges across every Bash shape.

A Bash line decomposes into four independent, optional axes -- a leading
``cd``, an enclosing loop or sequence, the command itself, and a trailing
filter. Sixteen combinations. Each matcher used to re-derive the shape it
understood, so coverage was per-tool and accidental: ``grep p f | head``
nudged while ``cd X && grep p f | head`` did not, because nobody had
written that particular combination.

These cases pin the matrix itself, so a matcher cannot regain private
structure handling without a hole showing up here.
"""

from __future__ import annotations

import pytest

from sagent.bin.cli import resolve_tools
from sagent.tools.lib.bash import BashMatcher, parse_bash


_MATCHERS = resolve_tools(["Read", "Grep", "Glob", "List", "Edit", "WebFetch"])


def _nudging_tools(command: str) -> set[str]:
    """Return the names of every tool that nudges for ``command``."""
    trees = parse_bash(command)
    assert trees is not None, f"failed to parse: {command!r}"
    return {
        t.name for t in _MATCHERS if isinstance(t, BashMatcher) and t.bash_match(trees)
    }


@pytest.mark.parametrize(
    ("command", "tool"),
    [
        # Axis 3 alone.
        ("cat foo.py", "Read"),
        ("head -20 foo.py", "Read"),
        ("tail -5 foo.py", "Read"),
        ("sed -n '1,50p' foo.py", "Read"),
        ("grep -n pat foo.py", "Grep"),
        # ``ls`` is List's executable even when the nudge text points at
        # Glob: the tool that recognizes the shape owns the suggestion.
        ("ls *.py", "List"),
        ("ls", "List"),
        ("find . -name '*.py'", "Glob"),
        ("sed -i 's/a/b/' foo.py", "Edit"),
        ("curl https://example.com", "WebFetch"),
        # Axis 1 + 3.
        ("cd /srv && cat foo.py", "Read"),
        ("cd /srv && sed -n '1,50p' foo.py", "Read"),
        ("cd /srv && grep -n pat foo.py", "Grep"),
        ("cd /srv && find . -name '*.py'", "Glob"),
        ("cd /srv && sed -i 's/a/b/' foo.py", "Edit"),
        ("cd /srv && curl https://example.com", "WebFetch"),
        # Axis 3 + 4.
        ("cat foo.py | head -20", "Read"),
        ("grep -n pat foo.py | head -40", "Grep"),
        ("find . -name '*.py' | xargs grep pat", "Grep"),
        # Axis 1 + 3 + 4 -- the combination that was silent.
        ("cd /srv && grep -n pat foo.py | head -40", "Grep"),
        ("cd /srv && cat foo.py | head -20", "Read"),
        # Axis 2: sequences and loops.
        ("cat a.py; head -3 b.py", "Read"),
        ("cd /srv && ls foo && cat bar.py", "Read"),
        ("for f in *.py; do cat $f; done", "Read"),
        ("for d in a b; do cd $d && grep -n p f; done", "Grep"),
        # Multiple positionals: batching Read calls by hand.
        ("cat a.py b.py", "Read"),
        ("head -20 a.py b.py", "Read"),
    ],
)
def test_shape_nudges_the_right_tool(command: str, tool: str) -> None:
    assert tool in _nudging_tools(command), command


@pytest.mark.parametrize(
    "command",
    [
        # Redirects WRITE a file; that is not the tool's job.
        "grep pat foo.py > out.txt",
        "curl https://example.com -o out.html",
        # A deliberate env prefix is not an accidental Bash reach.
        "LC_ALL=C grep pat foo.py",
        # Sinks that TRANSFORM do work the tools cannot express.
        "cat foo.py | sort",
        "grep pat foo.py | awk '{print $2}'",
        # ``-c`` counts bytes, which neither Read nor Grep expresses.
        "head -c 20 foo.py",
        "grep pat foo.py | wc -c",
    ],
)
def test_shape_is_deliberately_silent(command: str) -> None:
    assert not _nudging_tools(command), command


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        # A decoy statement must not change a verdict. Matchers that
        # recover their pipeline neighbour by scanning every invocation
        # for an executable NAME cross statement boundaries, so an
        # unrelated command elsewhere on the line flips the answer.
        ("grep pat foo.py | wc -c; ls | wc -l", set[str]()),
        ("cat -n a.py; grep -n pat foo.py", {"Grep"}),
        ("ls | xargs grep pat; find . -name '*.py'", {"Glob"}),
    ],
)
def test_a_decoy_statement_does_not_change_the_verdict(
    command: str, expected: set[str]
) -> None:
    assert _nudging_tools(command) == expected, command


def test_ls_pipeline_reads_its_own_sink() -> None:
    """The suggestion must quote the sink that actually follows ``ls``."""
    trees = parse_bash("cat f | head -5; ls | head -20")
    assert trees is not None
    tool = next(t for t in _MATCHERS if t.name == "List")
    assert isinstance(tool, BashMatcher)
    hint = tool.bash_match(trees) or ""
    assert "max_results=20" in hint, hint


def test_a_sed_that_edits_in_place_is_never_a_read() -> None:
    """``-in`` is quiet PLUS in-place: the file is rewritten.

    Matching on "a short flag containing n" accepts it, so a destructive
    command is advertised as a Read.
    """
    assert "Read" not in _nudging_tools("sed -in '1,5p' foo.py")


def test_a_glob_positional_is_not_a_read() -> None:
    """Read takes ONE ``file_path`` and cannot expand a glob."""
    assert "Read" not in _nudging_tools("cat *.py")
    assert "Read" not in _nudging_tools("head -20 *.py")


@pytest.mark.parametrize(
    "command",
    [
        "curl --output=out.html https://example.com",
        "curl -sO https://example.com",
        "wget --output-document=out https://example.com",
    ],
)
def test_a_fetch_that_writes_a_file_does_not_nudge(command: str) -> None:
    """Exact-string flag matching misses ``--flag=value`` and bundles."""
    assert not _nudging_tools(command), command


def test_a_redirect_on_the_pipeline_sink_disqualifies_the_source() -> None:
    """``grep p f | head > out`` writes a file; it is not a Grep call."""
    assert not _nudging_tools("grep pat foo.py | head -20 > out.txt")


def test_sed_print_goes_to_read_and_sed_edit_goes_to_edit() -> None:
    """One executable, two tools, split on whether it mutates."""
    assert _nudging_tools("sed -n '1,50p' foo.py") == {"Read"}
    assert _nudging_tools("sed -i 's/a/b/' foo.py") == {"Edit"}


def test_find_splits_on_its_sink() -> None:
    """``find`` enumerates for Glob, but feeds a search for Grep."""
    assert _nudging_tools("find . -name '*.py'") == {"Glob"}
    assert _nudging_tools("find . -name '*.py' | xargs grep pat") == {"Grep"}


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
