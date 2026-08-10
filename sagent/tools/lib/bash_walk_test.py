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
    """Return ``(exe, args, cwd, sink_exe)`` for each command found."""
    trees = parse_bash(command)
    assert trees is not None, f"failed to parse: {command!r}"
    return [
        (i.exe, i.args, i.cwd, i.piped_into.exe if i.piped_into else "")
        for i in walk_commands(trees)
    ]


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


def test_a_redirect_downstream_is_reachable_from_the_source() -> None:
    """``grep p f | head > out`` writes a file; the source must see it."""
    trees = parse_bash("grep pat foo.py | head -20 > out.txt")
    assert trees is not None
    grep = next(i for i in walk_commands(trees) if i.exe == "grep")
    assert not grep.captures_stdout
    assert any(d.captures_stdout for d in grep.downstream())


def test_env_prefix_is_flagged() -> None:
    trees = parse_bash("LC_ALL=C grep pat foo.py")
    assert trees is not None
    assert dict(walk_commands(trees)[0].env_prefix) == {"LC_ALL": "C"}


def test_nested_cd_inside_a_loop() -> None:
    got = _walk("for f in *.py; do cd /srv && cat $f; done")
    assert [(e, c) for e, _a, c, _p in got] == [("cat", "/srv")]


def test_pipeline_neighbours_are_references_not_names() -> None:
    """A name cannot identify WHICH neighbour, only what kind it is.

    Two pipelines on one line both ending in ``wc`` are indistinguishable
    by name, so a matcher recovering its sink by scanning for an
    executable crosses the statement boundary and reads the wrong one.
    """
    trees = parse_bash("grep pat foo.py | wc -c; ls | wc -l")
    assert trees is not None
    invocations = walk_commands(trees)
    grep = next(i for i in invocations if i.exe == "grep")
    ls = next(i for i in invocations if i.exe == "ls")
    assert grep.piped_into is not None
    assert ls.piped_into is not None
    assert grep.piped_into.args == ("-c",)
    assert ls.piped_into.args == ("-l",)
    assert grep.piped_into is not ls.piped_into


def test_a_command_knows_its_upstream_source() -> None:
    """``cat -n f | grep p``: grep's rule is about what FED it.

    ``piped_into`` points downstream only, so a matcher asking "was I
    fed by a flagged cat" has to scan -- and then an unrelated ``cat``
    in another statement answers for it.
    """
    trees = parse_bash("cat -n f | grep p")
    assert trees is not None
    grep = next(i for i in walk_commands(trees) if i.exe == "grep")
    assert grep.piped_from is not None
    assert grep.piped_from.exe == "cat"
    assert grep.piped_from.args == ("-n", "f")


def test_a_bare_command_has_no_pipeline_neighbours() -> None:
    trees = parse_bash("cat -n a.py; grep -n pat foo.py")
    assert trees is not None
    for inv in walk_commands(trees):
        assert inv.piped_into is None, inv
        assert inv.piped_from is None, inv


def test_cd_does_not_thread_across_an_or_operator() -> None:
    """``cd X || CMD`` runs CMD only when the cd FAILED."""
    trees = parse_bash("cd /srv || cat foo.py")
    assert trees is not None
    cat = next(i for i in walk_commands(trees) if i.exe == "cat")
    assert cat.cwd == ""


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
