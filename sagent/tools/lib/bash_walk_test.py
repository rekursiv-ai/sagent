"""Tests for ``walk_commands``: one AST walk over every command shape.

A Bash line decomposes into four independent, optional axes -- a leading
``cd``, an enclosing loop or sequence, the command itself, and a
trailing filter. Sixteen combinations, and every matcher used to
re-derive its own subset by hand: ``grep p f | head`` nudged while
``cd X && grep p f | head`` did not, because nobody had written that
particular combination.
"""

from __future__ import annotations

from sagent.tools.lib.bash import parse_bash, walk_commands


def _walk(command: str) -> list[tuple[str, tuple[str, ...], str, str]]:
    """Return ``(exe, args, cwd, piped_into)`` for each command found."""
    trees = parse_bash(command)
    assert trees is not None, f"failed to parse: {command!r}"
    return [(i.exe, i.args, i.cwd, i.piped_into) for i in walk_commands(trees)]


def test_bare_command() -> None:
    assert _walk("cat foo.py") == [("cat", ("foo.py",), "", "")]


def test_cd_prefix_sets_the_cwd() -> None:
    assert _walk("cd /srv && cat foo.py") == [("cat", ("foo.py",), "/srv", "")]


def test_pipeline_records_the_sink() -> None:
    got = _walk("grep -n pat foo.py | head -40")
    assert got[0] == ("grep", ("-n", "pat", "foo.py"), "", "head")


def test_cd_and_pipeline_together() -> None:
    """The combination that was silent: axes 1 + 3 + 4."""
    got = _walk("cd /srv && grep -n pat foo.py | head -40")
    assert got[0] == ("grep", ("-n", "pat", "foo.py"), "/srv", "head")


def test_and_chain_yields_every_command() -> None:
    got = _walk("cd /srv && ls && cat foo.py")
    assert [(e, c) for e, _a, c, _p in got] == [("ls", "/srv"), ("cat", "/srv")]


def test_semicolon_sequence_yields_every_command() -> None:
    assert [e for e, _a, _c, _p in _walk("cat a.py; head -3 b.py")] == ["cat", "head"]


def test_loop_body_is_visited() -> None:
    """An unresolvable arg still names the command, which is what nudges."""
    got = _walk("for f in *.py; do cat $f; done")
    assert [e for e, _a, _c, _p in got] == ["cat"]


def test_command_substitution_is_visited() -> None:
    assert "ls" in [e for e, _a, _c, _p in _walk("cat $(ls *.py)")]


def test_head_is_a_command_not_only_a_sink() -> None:
    """``head -20 f`` is a Read call with no pipe anywhere."""
    assert _walk("head -20 foo.py") == [("head", ("-20", "foo.py"), "", "")]


def test_stdout_redirect_is_flagged() -> None:
    """``grep p f > out`` writes a file; it is not a Grep call."""
    trees = parse_bash("grep pat foo.py > out.txt")
    assert trees is not None
    assert [i.captures_stdout for i in walk_commands(trees)] == [True]


def test_env_prefix_is_flagged() -> None:
    trees = parse_bash("LC_ALL=C grep pat foo.py")
    assert trees is not None
    assert dict(walk_commands(trees)[0].env_prefix) == {"LC_ALL": "C"}


def test_nested_cd_inside_a_loop() -> None:
    got = _walk("for f in *.py; do cd /srv && cat $f; done")
    assert [(e, c) for e, _a, c, _p in got] == [("cat", "/srv")]


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
