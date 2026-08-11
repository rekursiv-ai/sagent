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

# Flags that make Bash genuinely necessary: they act on what they match,
# window bytes rather than lines, follow a growing file, invert the
# match, or return a status instead of output. Everything else in a
# utility's alphabet must still reach its tool -- see
# ``test_every_documented_flag_still_nudges``.
#
# ``-v`` is here on a measurement, not a reading. On a file whose second
# line is ``ERROR DEBUG b``, ``grep ERROR f | grep -v DEBUG`` returns two
# lines while ``Grep(pattern='ERROR', exclude='DEBUG')`` returns all
# three: ``exclude`` becomes ``rg --glob '!PAT'``, a PATH filter, and
# ``directive_schema`` has no inverted-match property at all.
#
# ``-q`` prints NOTHING and is read for its exit status, which is the
# whole point of ``grep -q x f && action``. A tool call returns matches,
# not a status a later shell command can branch on.
_NECESSARY: frozenset[str] = frozenset(
    {
        "-exec",
        "-execdir",
        "-delete",
        "-ok",
        "-okdir",
        "-fprint",
        "-fprintf",
        "-fls",
        "-v",
        "-q",
    }
)

# A representative slice of each utility's real alphabet. Hard-coded
# rather than scraped from ``--help`` at runtime: the test must fail on
# THIS machine's parse and on CI's, and GNU/BSD spellings differ.
_GREP_HELP_FLAGS: tuple[str, ...] = (
    "-i",
    "-v",
    "-w",
    "-x",
    "-c",
    "-l",
    "-L",
    "-o",
    "-q",
    "-s",
    "-r",
    "-R",
    "-E",
    "-F",
    "-P",
    "-a",
    "-b",
    "-H",
    "-h",
    "-n",
    "-z",
    "--color=auto",
    "--exclude-dir=.git",
    "--include=*.py",
    "--line-buffered",
    "--null-data",
)
_FIND_HELP_PREDICATES: tuple[str, ...] = (
    "-name",
    "-iname",
    "-path",
    "-ipath",
    "-regex",
    "-type",
    "-maxdepth",
    "-mindepth",
    "-newer",
    "-mtime",
    "-size",
    "-perm",
    "-user",
    "-group",
    "-empty",
    "-print",
    "-print0",
    "-prune",
    "-follow",
    "-not",
)


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
        # ``tail -f`` follows a growing file; Read returns a snapshot.
        "tail -f /var/log/syslog",
        # ``find -exec``/``-delete`` act on what they match.
        "find . -name '*.pyc' -delete",
        "find . -name '*.py' -exec wc -l {} +",
        # ``-v`` inverts the match, which no tool property expresses --
        # ``exclude`` filters PATHS. Measured: on a line reading
        # ``ERROR DEBUG b``, the pipeline drops it and ``exclude`` keeps it.
        "grep -v pat foo.py",
        "grep -rn pat --include='*.py' . | grep -v '^./build/'",
    ],
)
def test_shape_is_deliberately_silent(command: str) -> None:
    assert not _nudging_tools(command), command


def test_grep_counts_lines_and_head_counts_bytes() -> None:
    """``-c`` means COUNT for grep and BYTES for head -- one denylist per exe.

    A shared ``-c`` denylist silently drops ``grep -c``, which the Grep
    tool expresses exactly as ``output_mode="count"``.
    """
    assert "Grep" in _nudging_tools("grep -c pat foo.py")
    assert "Read" not in _nudging_tools("head -c 20 foo.py")


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        # A decoy statement must not change a verdict. Matchers that
        # recover their pipeline neighbour by scanning every invocation
        # for an executable NAME cross statement boundaries, so an
        # unrelated command elsewhere on the line flips the answer.
        ("grep pat foo.py | wc -c; ls | wc -l", set[str]()),
        # ``cat -n`` is a READ: the Read tool numbers every line it
        # returns, so both tools legitimately claim this line.
        ("cat -n a.py; grep -n pat foo.py", {"Grep", "Read"}),
        ("cat -A a.py; grep -n pat foo.py", {"Grep"}),
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


@pytest.mark.parametrize(
    ("command", "tool"),
    [
        # A predicate or flag the tool cannot translate is still a reach
        # for that tool. Gating DETECTION on translatability made every
        # unknown token a silent miss, and there are ~80 per utility.
        ("find /srv -name f.toml -not -path '*/node_modules/*'", "Glob"),
        ("find . -maxdepth 2 -name '*.py'", "Glob"),
        ("find . -newer setup.py -name '*.py'", "Glob"),
        ("grep -w pat foo.py", "Grep"),
        ("grep -o pat foo.py", "Grep"),
        ("grep -rn pat --exclude-dir=.git .", "Grep"),
        ("ls -R /srv", "List"),
        ("ls -lh /srv", "List"),
        ("find /srv -name '*.toml' | head -50", "Glob"),
    ],
)
def test_an_untranslatable_shape_still_nudges(command: str, tool: str) -> None:
    assert tool in _nudging_tools(command), command


@pytest.mark.parametrize(
    "command",
    [
        # A search reading STDIN has no path operand, so the Grep tool --
        # which takes ``path`` -- cannot express it at all. Nudging here
        # sends the caller to a tool that cannot do the job.
        "git log --oneline | grep fix",
        "uv run pytest -q | grep FAILED",
        "journalctl -u foo | grep error",
        "ps aux | grep python",
        "docker ps | grep running",
    ],
)
def test_a_stdin_fed_search_does_not_nudge(command: str) -> None:
    assert not _nudging_tools(command), command


@pytest.mark.parametrize(
    "command",
    [
        # ...but a search whose stdin comes from a FILE reader does have
        # an operand, one hop upstream.
        "cat foo.py | grep pat",
        "head -100 foo.py | grep pat",
        "sed -n '1,50p' foo.py | grep pat",
    ],
)
def test_a_file_fed_search_still_nudges(command: str) -> None:
    assert "Grep" in _nudging_tools(command), command


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (
            "grep -n pat foo.py | head -40 | grep -v skip",
            "grep -n pat foo.py | grep -v skip | head -40",
        ),
        (
            "cat foo.py | head -20 | grep -v skip",
            "cat foo.py | grep -v skip | head -20",
        ),
        ("find . -name '*.py' | head -5", "find . -name '*.py' | head -5 | cat"),
    ],
)
def test_the_verdict_does_not_depend_on_sink_order(first: str, second: str) -> None:
    """Reordering equivalent sinks must not flip the answer.

    Each matcher used to inspect only its IMMEDIATE sink while checking
    redirects across the whole chain, so the same pipeline stages in a
    different order gave opposite verdicts.
    """
    assert _nudging_tools(first) == _nudging_tools(second)


def test_detection_survives_a_translation_failure() -> None:
    """Detection must not depend on rendering a concrete suggestion.

    Fusing the two made an untranslatable flag silently suppress the
    nudge, which is why coverage tracked the flag whitelist rather than
    the shape.
    """
    tool = next(t for t in _MATCHERS if t.name == "List")
    assert isinstance(tool, BashMatcher)
    trees = parse_bash("ls -R /srv")
    assert trees is not None
    hint = tool.bash_match(trees)
    assert hint is not None
    assert "List tool" in hint


@pytest.mark.parametrize(
    ("exe", "tool", "flags"),
    [
        ("grep", "Grep", _GREP_HELP_FLAGS),
        ("find", "Glob", _FIND_HELP_PREDICATES),
    ],
)
def test_every_documented_flag_still_nudges(
    exe: str, tool: str, flags: tuple[str, ...]
) -> None:
    """Sweep each utility's REAL flag alphabet, not a hand-picked sample.

    Example-based cases can only encode misses somebody already noticed,
    which is why an unknown token stayed silent for as long as it did.
    Flags the tool genuinely cannot serve are named in ``_NECESSARY``;
    everything else must still reach its tool.
    """
    silent = [
        flag
        for flag in flags
        if flag not in _NECESSARY
        and tool not in _nudging_tools(f"{exe} {flag} pat /srv")
    ]
    assert not silent, f"{exe} flags silently suppressed the nudge: {silent}"


@pytest.mark.parametrize(
    ("command", "tool", "expected"),
    [
        # A sink `replaceable` ACCEPTS is part of the one tool call it
        # accepted the source for, so dropping it from the example
        # advertises a different search than the one being replaced.
        ("grep -rn pat /src | wc -l", "Grep", 'output_mode="count"'),
        # The operand of a stdin-fed search lives one hop upstream --
        # which is the whole reason `_stdin_operand` accepts the shape.
        ("cat foo.py | grep pat", "Grep", "path='foo.py'"),
        ("head -100 foo.py | grep pat", "Grep", "path='foo.py'"),
        # ``sed`` puts its SCRIPT in operand position, so counting
        # operands alone cannot recover the producer's path.
        ("sed -n '1,50p' foo.py | grep pat", "Grep", "path='foo.py'"),
        # ``find … | xargs grep pat``: the search is the xargs PAYLOAD,
        # so translating the xargs argv reads ``grep`` as the pattern.
        ("find /src -name '*.py' | xargs grep pat", "Grep", "pattern='pat'"),
        # ``tail -n +N`` counts from the START; only the unsigned form
        # is a tail. Measured against coreutils, not inferred.
        ("tail -n +5 foo.py", "Read", "offset=5"),
        ("head -n5 foo.py", "Read", "limit=5"),
        ("head --lines=50 foo.py", "Read", "limit=50"),
        # ``sed -n 'Np'`` prints ONE line.
        ("sed -n '5p' foo.py", "Read", "offset=5 limit=1"),
        # A sink that shows everything imposes no bound.
        ("ls /src | cat", "List", "path='/src'"),
    ],
)
def test_a_worked_example_matches_the_command_it_replaces(
    command: str, tool: str, expected: str
) -> None:
    """A wrong worked example is worse than none -- it gets acted on.

    Each case below was measured against the real utility; the shipped
    translator rendered a call with different semantics.
    """
    trees = parse_bash(command)
    assert trees is not None
    matcher = next(t for t in _MATCHERS if t.name == tool)
    assert isinstance(matcher, BashMatcher)
    assert expected in (matcher.bash_match(trees) or ""), command


# Every way a utility can spell the same request. Sampling ONE per shape
# is what let a defect hide behind its neighbour: ``head -100 f | grep p``
# translated while ``head -n 20 f | grep p`` -- the identical shape --
# returned nothing, because only the separated form leaves a token that a
# naive scan miscounts as a second path.
_COUNT_SPELLINGS: tuple[str, ...] = ("-20", "-n 20", "-n20", "--lines=20", "--lines 20")


@pytest.mark.parametrize("spelling", _COUNT_SPELLINGS)
def test_every_count_spelling_renders_the_same_read(spelling: str) -> None:
    """``head`` has five ways to ask for 20 lines; all are one Read call."""
    trees = parse_bash(f"head {spelling} foo.py")
    assert trees is not None
    matcher = next(t for t in _MATCHERS if t.name == "Read")
    assert isinstance(matcher, BashMatcher)
    hint = matcher.bash_match(trees) or ""
    assert "file_path='foo.py' limit=20" in hint, hint


@pytest.mark.parametrize("spelling", _COUNT_SPELLINGS)
def test_every_count_spelling_yields_the_upstream_path(spelling: str) -> None:
    """A producer's count is never its path, however the count is spelled.

    ``_search_args`` kept a private ``startswith("-")`` scan after
    :func:`operands` was extracted to own that question, so the separated
    forms lost the operand and the example vanished.
    """
    trees = parse_bash(f"head {spelling} foo.py | grep pat")
    assert trees is not None
    matcher = next(t for t in _MATCHERS if t.name == "Grep")
    assert isinstance(matcher, BashMatcher)
    hint = matcher.bash_match(trees) or ""
    assert "pattern='pat' path='foo.py'" in hint, hint


@pytest.mark.parametrize(
    "command",
    [
        # ``-e PATTERN`` is the POSIX spelling, and the only one that
        # works for a pattern beginning with a dash.
        "grep -e pat foo.py",
        "grep -e pat -i foo.py",
    ],
)
def test_an_explicit_pattern_flag_still_renders_the_pattern(command: str) -> None:
    """``-e`` names the pattern; consuming it as a flag value loses it."""
    trees = parse_bash(command)
    assert trees is not None
    matcher = next(t for t in _MATCHERS if t.name == "Grep")
    assert isinstance(matcher, BashMatcher)
    assert "pattern='pat'" in (matcher.bash_match(trees) or ""), command


def test_a_find_without_a_name_predicate_still_renders_its_root() -> None:
    """``find /src -type f | xargs grep pat`` IS ``Grep pattern path``.

    Requiring ``-name`` to render anything dropped the example from a
    shape whose root and pattern are both present.
    """
    trees = parse_bash("find /src -type f | xargs grep pat")
    assert trees is not None
    matcher = next(t for t in _MATCHERS if t.name == "Grep")
    assert isinstance(matcher, BashMatcher)
    hint = matcher.bash_match(trees) or ""
    assert "pattern='pat' path='/src'" in hint, hint


@pytest.mark.parametrize(
    ("command", "tool"),
    [
        # ``head -n -N`` is all-but-the-last-N, which Read cannot window.
        ("head -n -5 foo.py", "Read"),
        # Glob is case-sensitive and returns files AND directories.
        ("find /src -iname '*.PY'", "Glob"),
        ("find /src -type d -name build", "Glob"),
        # Several roots; List and Glob each take one.
        ("ls a b", "List"),
        # A flag VALUE is not an operand.
        ("ls -I '*.pyc' /src", "List"),
    ],
)
def test_an_inexpressible_shape_offers_no_worked_example(
    command: str, tool: str
) -> None:
    """Translation failure must drop the example, never invent one."""
    trees = parse_bash(command)
    assert trees is not None
    matcher = next(t for t in _MATCHERS if t.name == tool)
    assert isinstance(matcher, BashMatcher)
    hint = matcher.bash_match(trees) or ""
    assert hint, command
    assert "Try:" not in hint, hint


@pytest.mark.parametrize(
    "command",
    [
        # ``-c`` bundles: ``-c5`` is the same flag as ``-c 5``, so a deny
        # set matched against whole argv tokens never sees it.
        "head -c5 foo.py",
        "tail -c1 foo.py",
        "tail -fn 5 foo.log",
        "cat -vet foo.py",
        # ``-F`` is ``--follow=name --retry``: still a live stream, which
        # Read cannot return. Only the lowercase spelling was denied.
        "tail -F foo.log",
        "tail --retry -f foo.log",
    ],
)
def test_a_bundled_denied_flag_still_suppresses_the_nudge(command: str) -> None:
    """Deny sets must match the FLAG, not the token that spells it."""
    assert "Read" not in _nudging_tools(command), command


def test_a_bare_line_count_survives_the_bundle_rule() -> None:
    """``head -20 f`` is a count, not a cluster of digit flags.

    The obsolete-but-universal form; treating its digits as bundled
    letters would regress the most common head/tail spelling there is.
    """
    assert "Read" in _nudging_tools("head -20 foo.py")
    assert "limit=20" in (
        next(
            t.bash_match(parse_bash("head -20 foo.py") or ())
            for t in _MATCHERS
            if t.name == "Read" and isinstance(t, BashMatcher)
        )
        or ""
    )


def test_a_nudge_names_the_offending_fragment() -> None:
    """A compound line must say WHICH command to replace.

    Most nudged lines carry several commands, so a fixed string leaves
    the caller guessing which fragment the tool replaces.
    """
    trees = parse_bash("cd /srv && uv run pytest -q && grep -n pat foo.py")
    assert trees is not None
    tool = next(t for t in _MATCHERS if t.name == "Grep")
    assert isinstance(tool, BashMatcher)
    hint = tool.bash_match(trees) or ""
    assert "grep -n pat foo.py" in hint, hint


@pytest.mark.parametrize(
    "command",
    [
        "(grep p f) > out",
        "{ grep p f; } > out",
        "for x in a; do grep p f; done > out",
    ],
)
def test_a_redirect_on_an_enclosing_group_disqualifies_its_commands(
    command: str,
) -> None:
    """A group's redirect writes a file, so its commands are not tool calls.

    bashlex hangs a compound's redirects off its own ``redirects``
    attribute rather than ``parts``, so the walk never saw them and the
    inner command reported ``captures_stdout=False``.
    """
    trees = parse_bash(command)
    assert trees is not None
    matchers = [t for t in _MATCHERS if isinstance(t, BashMatcher)]
    assert all(m.bash_match(trees) is None for m in matchers), command


def test_a_group_without_a_redirect_still_nudges() -> None:
    """The inherited-redirect fix must not silence an ordinary group."""
    trees = parse_bash("(grep p f)")
    assert trees is not None
    matchers = [t for t in _MATCHERS if isinstance(t, BashMatcher)]
    assert any(m.bash_match(trees) for m in matchers)


@pytest.mark.parametrize(
    ("command", "tool", "expected"),
    [
        # A worked example that drops the ``cd`` names a DIFFERENT file
        # than the command it claims to replace -- and the tools resolve
        # relative paths against the agent's cwd, not the shell's.
        ("cd /srv && cat f", "Read", "file_path='/srv/f'"),
        ("cd /srv && head -20 f", "Read", "file_path='/srv/f'"),
        ("cd /srv && sed -n '1,5p' f", "Read", "file_path='/srv/f'"),
        ("cd /srv && grep -n pat f", "Grep", "path='/srv/f'"),
        ("cd /srv && ls", "List", "path='/srv'"),
        ("cd /srv && ls sub", "List", "path='/srv/sub'"),
        ("cd /srv && find . -name '*.py'", "Glob", "path='/srv'"),
        ("cd /srv && find sub -name '*.py'", "Glob", "path='/srv/sub'"),
        # An absolute operand wins over the prefix.
        ("cd /srv && cat /etc/hosts", "Read", "file_path='/etc/hosts'"),
        # ``cd`` composes; the second is relative to the first.
        ("cd /srv && cd sub && cat f", "Read", "file_path='/srv/sub/f'"),
    ],
)
def test_a_cd_prefix_reaches_the_worked_example(
    command: str, tool: str, expected: str
) -> None:
    """``cd`` is tracked by the walk, so a renderer that drops it lies.

    ``cd /srv && cat f`` rendered ``Read file_path='f'``, which resolves
    against the agent's own cwd and reads a different file -- or none.
    """
    trees = parse_bash(command)
    assert trees is not None
    matcher = next(t for t in _MATCHERS if t.name == tool)
    assert isinstance(matcher, BashMatcher)
    assert expected in (matcher.bash_match(trees) or ""), command


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        # ``||`` runs the tail only when the ``cd`` FAILED, so the
        # directory it names is exactly where the command is not.
        ("cd /srv || cat f", "file_path='f'"),
        ("cd /a && cd /b || cat f", "file_path='/a/f'"),
        # ``&`` backgrounds the ``cd`` in a subshell; the parent's
        # directory never changes.
        ("cd /x & cat f", "file_path='f'"),
    ],
)
def test_a_cd_that_does_not_take_effect_is_not_applied(
    command: str, expected: str
) -> None:
    """Only a ``cd`` the shell actually performed may reach the example."""
    trees = parse_bash(command)
    assert trees is not None
    matcher = next(t for t in _MATCHERS if t.name == "Read")
    assert isinstance(matcher, BashMatcher)
    assert expected in (matcher.bash_match(trees) or ""), command


@pytest.mark.parametrize(
    "command",
    [
        # The value of a value-flag is not an operand. Each of these has
        # NO file and NO pattern, so there is nothing for a tool to act
        # on -- ``grep -A 2`` reads stdin and ``head -n 5`` reads stdin.
        "grep -A 2",
        "grep -B 3",
        "grep -C 1",
        "head -n 5",
        "tail -n 3",
        "head --lines=5",
        "grep --include='*.py'",
        "grep --exclude=build",
    ],
)
def test_a_flag_value_is_not_an_operand(command: str) -> None:
    """A command with only flags reads stdin; no tool call replaces it."""
    assert not _nudging_tools(command), command


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        # ...but the same flag WITH an operand still nudges.
        ("grep -A 2 pat f", "Grep"),
        ("head -n 5 f", "Read"),
        ("grep --include='*.py' pat /src", "Grep"),
    ],
)
def test_a_value_flag_does_not_suppress_a_real_operand(
    command: str, expected: str
) -> None:
    assert expected in _nudging_tools(command), command


def test_an_ignore_glob_is_not_a_glob_positional() -> None:
    """``ls -I '*.pyc' /src`` lists a DIRECTORY; the glob is a filter.

    Reading the flag's value as a positional routed a plain listing to
    the Glob tool, which cannot express ``-I`` at all.
    """
    trees = parse_bash("ls -I '*.pyc' /src")
    assert trees is not None
    matcher = next(t for t in _MATCHERS if t.name == "List")
    assert isinstance(matcher, BashMatcher)
    hint = matcher.bash_match(trees) or ""
    assert "List tool" in hint, hint
    assert "Glob" not in hint, hint


def test_a_context_flag_renders_as_a_usable_keyword() -> None:
    """``"-B"=3`` is not syntax any caller can paste.

    The schema names the property ``-B``, and the sibling booleans
    already render ``-i=true``; quoting only this one produced a form
    that parses as nothing.
    """
    trees = parse_bash("grep -B 3 pat f")
    assert trees is not None
    matcher = next(t for t in _MATCHERS if t.name == "Grep")
    assert isinstance(matcher, BashMatcher)
    hint = matcher.bash_match(trees) or ""
    assert "-B=3" in hint, hint
    assert '"-B"' not in hint, hint


def test_a_rendered_fragment_is_shell_safe_and_delimited() -> None:
    """The rendered fragment must survive being read back as shell.

    Two ways it did not: bashlex yields the UNQUOTED word, so re-rendering
    ``find . -name '*.py'`` emitted a bare glob the shell would expand, and a
    path ending in ``.`` ran into the sentence period -- ``grep -r pat ..``
    names the parent directory, not the current one.
    """
    trees = parse_bash("find . -name '*.py'")
    assert trees is not None
    glob_tool = next(t for t in _MATCHERS if t.name == "Glob")
    assert isinstance(glob_tool, BashMatcher)
    hint = glob_tool.bash_match(trees) or ""
    assert "'*.py'" in hint, hint

    trees = parse_bash("grep -r pat .")
    assert trees is not None
    grep_tool = next(t for t in _MATCHERS if t.name == "Grep")
    assert isinstance(grep_tool, BashMatcher)
    hint = grep_tool.bash_match(trees) or ""
    assert "`grep -r pat .`." in hint, hint
    assert " pat .." not in hint, hint


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
