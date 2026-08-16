"""Bash command parsing, classification, and matcher helpers.

Single entry point over ``bashlex`` (no type stubs, third-party).
Everything works on the raw bashlex AST; ``Command`` and the helpers
below are matcher-friendly conveniences over it.

Public surface:

- :func:`parse_bash` / :func:`cached_parse_bash` - parse a command
  string into typed bashlex trees.
- :func:`walk_commands` - every simple command in the line, each with
  its ``cd`` context and pipeline neighbours. The one entry point a
  tool matcher needs.
- :func:`replaceable` - whether one invocation is a shape a dedicated
  tool replaces. The one policy a tool matcher needs.
- :func:`render_command` - re-render an invocation for a nudge message.
- :func:`is_read_only` - classify a parse as side-effect-free.
- :func:`sed_mutates` - whether ``sed`` args request an in-place edit.
- :func:`resolve_cwd_path` - combine a ``cd`` prefix with a positional path.
- :class:`Invocation`, :class:`Command`, :class:`Node` - record types.
- :class:`BashMatcher` - the duck-type a nudging tool satisfies.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast, runtime_checkable

import dataclasses
import re
import shlex
import types


if TYPE_CHECKING:
    from bashlex.ast import (
        node as Node,  # noqa: N812 -- rename to PascalCase for type convention
    )

    import bashlex
    import bashlex.errors
else:
    from wrapt import lazy_import

    bashlex = lazy_import("bashlex")  # 88ms cold
    Node = object

__all__ = [
    "FIND_DENY_FLAGS",
    "BashMatcher",
    "BashParseCache",
    "Command",
    "Invocation",
    "Node",
    "bounding_sink",
    "cached_parse_bash",
    "cwd_is_known",
    "is_read_only",
    "operands",
    "parse_bash",
    "parse_line_count",
    "render_command",
    "replaceable",
    "resolve_cwd_path",
    "sed_mutates",
    "walk_commands",
]

type BashParseCache = dict[str, tuple[Node, ...] | None]


@dataclass(frozen=True, slots=True, kw_only=True)
class Command:
    """One simple command: exe + positional args + leading env prefix.

    ``env_prefix`` captures ``FOO=bar ./cmd`` assignments; they are
    stripped from ``exe``/``args`` so matchers don't see them as
    positional tokens.

    ``captures_stdout`` is True iff a redirect diverts fd 1 (stdout) -
    the only redirect class that changes what the LLM sees on the
    tool-result side. Stderr redirects (``2>&1``, ``2>/dev/null``)
    leave stdout intact, so matchers treat them as cosmetic and still
    fire.
    """

    exe: str
    """Executable name (first word; ``""`` for assignment-only commands)."""

    args: tuple[str, ...]
    """Positional arguments after ``exe`` in argv order."""

    env_prefix: Mapping[str, str]
    """Leading ``KEY=value`` assignments (e.g. ``FOO=bar ./cmd``)."""

    captures_stdout: bool
    """True when a redirect diverts fd 1 (``>``/``>>``/``>|``/``>&``)."""


@runtime_checkable
class BashMatcher(Protocol):
    """A tool that nudges when Bash is doing its job.

    Structural, not inherited: a tool opts in by defining the method,
    and :class:`Bash` discovers its peers through this check.
    """

    def bash_match(self, trees: Sequence[Node]) -> str | None:
        """Return a nudge when ``trees`` contains a replaceable shape."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class Invocation:
    """One simple command found anywhere in a Bash line, in context.

    A Bash line decomposes into four independent, optional axes: a
    leading ``cd``, an enclosing loop or sequence, the command itself,
    and a trailing filter. Matchers that re-derive this structure each
    cover a different subset -- which is why ``grep p f | head`` nudged
    while ``cd X && grep p f | head`` did not. This record is what one
    shared walk yields so a matcher only decides POLICY: whether its own
    executable is replaceable.

    Attributes:
      exe: Executable name (first word).
      args: Positional arguments in argv order.
      cwd: Directory from an enclosing ``cd PATH &&``, or ``""``.
      piped_into: The command this one's stdout feeds, or ``None``.
          Distinguishes ``find . -name '*.py'`` (Glob) from
          ``find . | xargs grep`` (Grep). A REFERENCE, not a name: two
          pipelines on one line can both end in ``wc``, so a matcher
          recovering its neighbour by executable name reads whichever
          came first and an unrelated statement changes the verdict.
      piped_from: The command feeding this one's stdin, or ``None``.
          Some rules are about the SOURCE -- "was I fed by a flagged
          ``cat``" -- which the downstream link cannot answer.
      env_prefix: Leading ``KEY=value`` assignments. A deliberate
          ``LC_ALL=C grep`` is not a Grep call.
      captures_stdout: True when a redirect diverts fd 1. ``grep p f >
          out`` writes a file rather than showing the operator anything.

    """

    exe: str
    args: tuple[str, ...]
    cwd: str = ""
    piped_into: Invocation | None = None
    piped_from: Invocation | None = None
    env_prefix: Mapping[str, str] = types.MappingProxyType({})
    captures_stdout: bool = False

    def downstream(self) -> Iterator[Invocation]:
        """Yield every command this one's output flows through."""
        node = self.piped_into
        while node is not None:
            yield node
            node = node.piped_into


def walk_commands(trees: Sequence[Node]) -> tuple[Invocation, ...]:
    """Yield every simple command in ``trees``, with its context.

    Descends through ``&&``/``;`` lists, pipelines, loop and conditional
    bodies, and command substitutions, so a matcher sees a flat list
    instead of re-deriving shape. Nothing is filtered here: a matcher
    that cares about redirects or an env prefix reads the flags on each
    record, and one that does not is not silently narrowed by the walk.

    Args:
      trees: Top-level bashlex AST nodes.

    Returns:
      invocations: Every simple command found, in source order.

    """
    out: list[Invocation] = []
    for node in trees:
        _walk_node(node, cwd="", out=out)
    return tuple(out)


def _walk_node(
    node: Node,
    *,
    cwd: str,
    out: list[Invocation],
    captured: bool = False,
    sink: Invocation | None = None,
) -> None:
    """Recurse one AST node, appending every simple command found.

    ``captured`` carries an ENCLOSING redirect down to the commands it
    governs. bashlex hangs a compound's redirects off its own
    ``redirects`` attribute, so ``(grep p f) > out`` gave the inner
    command ``captures_stdout=False`` and every matcher nudged a fragment
    that writes a file.
    """
    kind: str = node.kind
    captured = captured or _redirects_stdout(node)
    if kind == "command":
        _walk_command(node, cwd=cwd, out=out, captured=captured, sink=sink)
        return
    if kind == "pipeline":
        _walk_pipeline(node, cwd=cwd, out=out, captured=captured)
        return
    if kind == "list":
        _walk_list(node, cwd=cwd, out=out, captured=captured)
        return
    # ``compound`` (loops, conditionals) and everything else: descend
    # through whatever child collections the node carries. A loop body
    # nudges even when its argument is a loop variable -- a command
    # reaching for ``cat $f`` wants the tool regardless of whether the
    # filename is knowable here.
    for child in _child_nodes(node):
        _walk_node(child, cwd=cwd, out=out, captured=captured, sink=sink)


def _redirects_stdout(node: Node) -> bool:
    """Whether a compound node carries a redirect that diverts fd 1."""
    redirects: object = getattr(node, "redirects", None)
    if not isinstance(redirects, list):
        return False
    return any(
        cast("str", getattr(r, "type", "")) in (">", ">>", ">&", ">|")
        and cast("int | None", getattr(r, "input", None)) in (None, 1)
        for r in cast("list[object]", redirects)
    )


# A ``cd`` whose destination the text does not reveal (``cd``, ``cd -``).
# Distinct from ``""`` (no ``cd`` at all): a worked example must be
# suppressed rather than resolved against the wrong directory.
_UNKNOWN_CWD: Final = "\0unknown-cwd"

# Stands in for a pipeline stage that is not a simple command -- a
# subshell, a group, a loop. No tool claims this name, and no sink rule
# treats it as shaping, so it correctly blocks the stages it separates.
_OPAQUE_STAGE: Final = "\0compound"


def _walk_list(
    node: Node, *, cwd: str, out: list[Invocation], captured: bool = False
) -> None:
    """Walk an ``A && B ; C`` list, threading any ``cd`` through it.

    A ``cd`` binds to everything after it in the same list, so
    ``cd X && ls && cat f`` reports BOTH commands under ``X`` -- the
    two-command-only unwrap missed the chain entirely.

    Three ways the shell's own answer differs from "the last ``cd`` wins":

    - ``cd`` COMPOSES. ``cd /srv && cd sub`` leaves the shell in
      ``/srv/sub``, so the second is resolved against the first.
    - ``||`` runs its tail only when the left side FAILED, so the ``cd``
      immediately before it did not happen -- but any EARLIER one did.
      Rewinding all the way to the incoming cwd discarded those.
    - ``&`` backgrounds the ``cd`` in a subshell, so the parent's
      directory is untouched and the prior one still stands.
    """
    inner_cwd = cwd
    # The cwd in force before the most recent ``cd``, which is where an
    # ``||`` tail runs: that ``cd`` is exactly the command that failed.
    before_last_cd = cwd
    for part in node.parts:
        if part.kind == "operator":
            if part.op == "||":
                inner_cwd = before_last_cd
            elif part.op == "&":
                # The preceding command ran in a subshell; if it was a
                # ``cd``, the parent never moved.
                inner_cwd = before_last_cd
            continue
        if part.kind == "command":
            cmd = _parse_command(part)
            if cmd.exe == "cd" and not cmd.captures_stdout:
                before_last_cd = inner_cwd
                inner_cwd = _cd_target(inner_cwd, cmd.args)
                continue
        before_last_cd = inner_cwd
        _walk_node(part, cwd=inner_cwd, out=out, captured=captured)


def _cd_target(cwd: str, args: tuple[str, ...]) -> str:
    """Where a ``cd`` leaves the shell, or ``_UNKNOWN_CWD`` when unknowable.

    Only a literal path is knowable here. Bare ``cd`` goes to ``$HOME``
    and ``cd -`` to ``$OLDPWD``; both were previously read as "no cd" or,
    worse, as a directory literally named ``-`` -- which rendered
    ``file_path='-/f'``. Option flags (``-P``, ``-L``, ``--``) carry no
    destination and must not be mistaken for one.
    """
    paths = [a for a in args if not a.startswith("-") or a == "-"]
    if not paths or paths[0] == "-":
        # ``$HOME`` / ``$OLDPWD``: real, but not knowable from the text.
        return _UNKNOWN_CWD
    return resolve_cwd_path(cwd, paths[0])


def _walk_pipeline(
    node: Node, *, cwd: str, out: list[Invocation], captured: bool = False
) -> None:
    """Walk ``A | B | C``, linking each stage to its true neighbours.

    Built back-to-front so each stage can hold a reference to the one it
    feeds; the reverse links are then stitched in a second pass. Names
    would not do: two pipelines on one line can both end in ``wc``, and
    a matcher that searches for one by name reads the wrong pipeline.
    """
    stages = [p for p in node.parts if p.kind != "pipe"]
    linked: list[Invocation | None] = [None] * len(stages)
    for i in reversed(range(len(stages))):
        if stages[i].kind != "command":
            # A compound stage -- ``(sort)``, a loop -- is still a stage.
            # Leaving it unlinked let its neighbours join across it as if
            # it were absent, so ``grep p f | (sort)`` looked unpiped and
            # nudged although ``sort`` transforms the output.
            linked[i] = Invocation(
                exe=_OPAQUE_STAGE,
                args=(),
                cwd=cwd,
                piped_into=linked[i + 1] if i + 1 < len(stages) else None,
            )
            continue
        cmd = _parse_command(stages[i])
        if not cmd.exe:
            continue
        # Only the LAST stage's stdout reaches an enclosing redirect; the
        # others feed the next pipe regardless.
        linked[i] = Invocation(
            exe=cmd.exe,
            args=cmd.args,
            cwd=cwd,
            piped_into=linked[i + 1] if i + 1 < len(stages) else None,
            env_prefix=cmd.env_prefix,
            captures_stdout=cmd.captures_stdout or (captured and i == len(stages) - 1),
        )
    # ``Invocation`` is frozen, so the upstream link is stitched by
    # rebuilding each record once its predecessor is known.
    for i, inv in enumerate(linked):
        if inv is None:
            continue
        prev = next((linked[j] for j in reversed(range(i)) if linked[j]), None)
        out.append(dataclasses.replace(inv, piped_from=prev) if prev else inv)
    for i, stage in enumerate(stages):
        if stage.kind == "command":
            _walk_substitutions(stage, cwd=cwd, out=out)
            continue
        # The commands INSIDE a compound stage feed whatever the stage
        # feeds, so they must carry that sink too: ``(grep p f) | sort``
        # is a search whose output is transformed, not a bare search.
        stage_inv = linked[i]
        _walk_node(
            stage,
            cwd=cwd,
            out=out,
            captured=captured,
            sink=stage_inv.piped_into if stage_inv is not None else None,
        )


def _walk_command(
    node: Node,
    *,
    cwd: str,
    out: list[Invocation],
    captured: bool = False,
    sink: Invocation | None = None,
) -> None:
    """Record one un-piped simple command, then its substitutions."""
    cmd = _parse_command(node)
    if cmd.exe:
        out.append(
            Invocation(
                exe=cmd.exe,
                args=cmd.args,
                cwd=cwd,
                piped_into=sink,
                env_prefix=cmd.env_prefix,
                captures_stdout=cmd.captures_stdout or captured,
            )
        )
    _walk_substitutions(node, cwd=cwd, out=out)


def _walk_substitutions(node: Node, *, cwd: str, out: list[Invocation]) -> None:
    """Descend into ``$(...)`` and friends nested in a command's words."""
    for child in _child_nodes(node):
        if child.kind in ("word", "commandsubstitution", "command", "list", "pipeline"):
            _walk_node(child, cwd=cwd, out=out)


# Sinks that only truncate or paginate what the source already produced.
# Piping into one is equivalent to calling the tool directly, since every
# tool bounds its own output.
_SHAPING_SINKS: frozenset[str] = frozenset({"head", "tail", "less", "more", "cat"})

# Executables whose operand is a PATTERN plus an optional path.
_SEARCH_EXES: frozenset[str] = frozenset({"grep", "rg"})

# Executables that turn a path into file CONTENT on stdout. A search fed
# by one of these still has a file operand -- one hop upstream -- so it
# remains a single tool call.
_FILE_PRODUCERS: frozenset[str] = frozenset({"cat", "head", "tail", "sed"})


def _value_flags_for(exe: str) -> frozenset[str]:
    """Flags of ``exe`` whose value is the next word (or the token tail).

    One definition, two readers: :func:`operands` skips the value so it
    is not counted as a path, and :func:`_denied` stops its letter scan
    there so an attached value is not read as more flags. A copy per
    caller is how the two ``find`` denylists drifted apart.
    """
    # ``-n`` is the counterexample in both directions: it takes a value
    # for head/tail and takes NONE for ``cat`` (number lines), ``sed``
    # (quiet), or ``grep`` (show line numbers), where consuming the next
    # word swallows the filename.
    vocabulary: dict[str, frozenset[str]] = {
        "grep": frozenset(
            {"-A", "-B", "-C", "-m", "-e", "-f", "-d", "-D", "--label", "--color"}
        ),
        "rg": frozenset({"-A", "-B", "-C", "-m", "-e", "-f", "-g", "-t", "--color"}),
        "head": frozenset({"-n", "-c", "--lines", "--bytes"}),
        "tail": frozenset({"-n", "-c", "--lines", "--bytes"}),
        "sed": frozenset({"-e", "-f", "--expression", "--file"}),
        "ls": frozenset({"-I", "-w", "-T", "--ignore", "--block-size", "--format"}),
        # ``find``'s operands are its ROOTS. Its predicates are whole
        # words rather than clustered letters, and most take a value, so
        # without them ``find /src -name '*.py'`` reports the glob as a
        # second root.
        "find": frozenset(
            {
                "-name",
                "-iname",
                "-path",
                "-ipath",
                "-regex",
                "-iregex",
                "-type",
                "-maxdepth",
                "-mindepth",
                "-newer",
                "-mtime",
                "-mmin",
                "-size",
                "-perm",
                "-user",
                "-group",
                "-anewer",
                "-cnewer",
            }
        ),
    }
    return cast("frozenset[str]", vocabulary.get(exe, frozenset()))  # pyright: ignore[reportUnnecessaryCast] -- ty needs the cast; pyright resolves the type


def _value_flag_letters(exe: str) -> frozenset[str]:
    """Short-flag letters of ``exe`` that consume the rest of the token.

    Shares :func:`operands`' vocabulary rather than repeating it: a
    second copy is exactly how the two ``find`` denylists drifted apart.
    """
    return frozenset(
        f[1] for f in _value_flags_for(exe) if len(f) == 2 and not f.startswith("--")
    )


def operands(exe: str, args: Sequence[str]) -> tuple[str, ...]:
    """Positional arguments of ``exe``, with flag VALUES excluded.

    Without this, ``grep -A 2`` counts ``2`` as a pattern and
    ``head -n 5`` counts ``5`` as a file, so a command with no operand at
    all -- a stdin filter -- was advertised as a one-call replacement.

    The vocabulary is per-executable and lives here rather than in each
    matcher: ``_stdin_operand`` needs the PRODUCER's spellings to find
    the path on ``head -n 20 f | grep p``, so a copy per tool would be
    the same duplication that let one ``find`` denylist diverge from the
    other. A long flag spelled ``--flag=value`` carries its own value and
    consumes nothing.

    Args:
      exe: Executable name, which selects the vocabulary.
      args: Positional arguments in argv order.

    Returns:
      positional: Arguments that are genuinely operands.

    """
    value_flags = _value_flags_for(exe)
    out: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--":
            out.extend(args[i + 1 :])
            break
        if a.startswith("-") and a != "-":
            i += 2 if a in value_flags and i + 1 < len(args) else 1
            continue
        out.append(a)
        i += 1
    return tuple(out)


def replaceable(
    inv: Invocation,
    *,
    exes: frozenset[str],
    deny: frozenset[str] = frozenset(),
) -> bool:
    """Whether ``inv`` is a shape one dedicated tool call replaces.

    Answers only questions about the SHELL LINE -- does an operand
    exist, does anything downstream write or transform -- and takes the
    caller's own policy as ``exes`` and ``deny``. Splitting it this way
    is the point: matchers previously gated detection on being able to
    TRANSLATE every flag, so any of the ~80 tokens per utility outside
    their whitelist silently suppressed the nudge. Translation now
    happens after this returns, and failing it costs message quality
    rather than detection.

    Args:
      inv: One simple command, with its pipeline context.
      exes: Executables the calling tool claims.
      deny: Arguments that make Bash genuinely necessary for THIS
          executable. Per-executable because the same spelling differs:
          ``-c`` counts bytes for ``head`` but matching lines for
          ``grep``, which the Grep tool expresses directly.

    Returns:
      replaceable: True when one tool call covers the invocation.

    """
    if inv.exe not in exes or inv.env_prefix or inv.captures_stdout:
        return False
    if any(_sink_blocks(inv, d) for d in inv.downstream()):
        return False
    if any(_denied(a, deny, exe=inv.exe) for a in inv.args):
        return False
    if inv.piped_from is not None:
        return _stdin_operand(inv)
    # ``ls``/``find`` default to the current directory, so a bare
    # invocation still names a target.
    if inv.exe in ("ls", "find"):
        return True
    return bool(operands(inv.exe, inv.args))


def render_command(inv: Invocation) -> str:
    """Re-render ``inv`` as the shell fragment it came from.

    Most nudged lines carry several commands, so a fixed nudge string
    leaves the reader to guess which fragment the tool replaces.

    Args:
      inv: Invocation to render.

    Returns:
      text: ``exe arg arg``, each argument shell-quoted.

    """
    # shlex.quote, not a space test: bashlex hands back the UNQUOTED word, so
    # re-rendering `find . -name '*.py'` without quotes produced a fragment the
    # shell would glob-expand, and an argument containing an apostrophe was
    # wrapped in the very character it contains.
    return " ".join([inv.exe, *(shlex.quote(a) for a in inv.args)])


def _denied(arg: str, deny: frozenset[str], *, exe: str = "") -> bool:
    """Whether ``arg`` carries a denied flag in any of its spellings.

    Short flags BUNDLE and may carry an attached value, so ``-c``, ``-c5``
    and ``-fn`` all mean ``-c``/``-f`` to the shell while sharing no token
    with each other. Whole-token equality saw only the first, and
    ``head -c5`` -- a byte window -- was advertised as a line read.

    A bare count (``head -20``) is not a cluster: its digits are the
    argument. Testing that first keeps the most common head/tail spelling
    out of the letter scan.

    A value-taking flag ENDS the cluster: everything after it is that
    flag's argument, not more letters. ``grep -evalue`` is ``-e`` with
    the pattern ``value``, and scanning the whole tail found a ``-v``
    that is not there -- silently suppressing the nudge for any pattern
    containing a denied letter.
    """
    if arg in deny or arg.partition("=")[0] in deny:
        return True
    if not arg.startswith("-") or arg.startswith("--") or arg[1:].isdigit():
        return False
    value_flags = _value_flag_letters(exe)
    for c in arg[1:]:
        if c.isdigit():
            continue
        # Deny FIRST: ``head -c5`` is a denied byte window whose own
        # letter also takes a value, so stopping on the value test would
        # wave through the very flag being denied.
        if f"-{c}" in deny:
            return True
        if c in value_flags:
            # This letter consumes the REST of the token as its value.
            return False
    return False


def bounding_sink(inv: Invocation) -> Invocation | None:
    """The ``head``/``tail`` that bounds ``inv``, anywhere downstream.

    Reading only the ADJACENT stage lost the bound across a pass-through:
    ``ls | cat | head -5`` and ``ls | head -5`` are the same five rows,
    but only the second rendered ``keep_last``/``max_results``. Every
    other shaping sink (``cat``, ``less``, ``more``) shows the whole
    stream and imposes nothing, so it is skipped rather than stopping
    the search.

    Args:
      inv: The invocation whose output bound is wanted.

    Returns:
      sink: The bounding ``head``/``tail``, or ``None`` when unbounded.

    """
    for stage in inv.downstream():
        if stage.exe in ("head", "tail"):
            return stage
        if stage.exe not in _SHAPING_SINKS:
            return None
    return None


def parse_line_count(args: tuple[str, ...]) -> int | None:
    """Line count a ``head``/``tail`` imposes, or ``None`` if unreadable.

    Bare ``head``/``tail`` is 10 lines -- measured against coreutils on
    a 15-entry directory, both printed 10 -- so the default is a real
    bound, not an invented one.

    Args:
      args: The sink's positional arguments.

    Returns:
      count: Lines the sink keeps, or ``None`` when unparseable.

    """
    if not args:
        return 10
    count: int | None = None
    i = 0
    while i < len(args):
        a = args[i]
        if a == "-n":
            if i + 1 >= len(args) or not args[i + 1].isdigit():
                return None
            count = int(args[i + 1])
            i += 2
            continue
        if a.startswith("-n"):
            if not a[2:].isdigit():
                return None
            count = int(a[2:])
            i += 1
            continue
        if len(a) > 1 and a[0] == "-" and a[1:].isdigit():
            count = int(a[1:])
            i += 1
            continue
        return None
    if count is None or count < 1:
        return None
    return count


def _sink_blocks(source: Invocation, sink: Invocation) -> bool:
    """Whether ``sink`` stops ``source`` from being one tool call.

    Asked of every stage in ``downstream()``, not just the adjacent one:
    checking the immediate sink alone made ``a | head | grep -v`` and
    ``a | grep -v | head`` -- the same pipeline -- disagree.
    """
    if sink.captures_stdout:
        return True
    if sink.exe in _SHAPING_SINKS:
        return False
    # ``grep p f | wc -l`` is the search's own ``output_mode="count"``;
    # other ``wc`` flags count bytes or words, which it cannot express.
    return not (
        sink.exe == "wc" and sink.args == ("-l",) and source.exe in _SEARCH_EXES
    )


def _stdin_operand(inv: Invocation) -> bool:
    """Whether a stdin-fed ``inv`` still names an operand one hop up.

    A search reading a pipe has no path of its own, and every dedicated
    tool takes a path. ``git log | grep fix`` is therefore not a Grep
    call at all, while ``cat f.py | grep fix`` is -- the operand is on
    the producer.
    """
    source = inv.piped_from
    assert source is not None
    return (
        inv.exe in _SEARCH_EXES
        # A search still needs its own pattern; only the PATH comes from
        # upstream.
        and bool(operands(inv.exe, inv.args))
        and source.exe in _FILE_PRODUCERS
        # The producer's own vocabulary, not the search's: ``head -n 20``
        # spends its ``20`` on the flag, so it is not the path either.
        and bool(operands(source.exe, source.args))
        and not (source.exe == "sed" and sed_mutates(source.args))
    )


def _child_nodes(node: Node) -> list[Node]:
    """Return every child AST node hanging off ``node``.

    bashlex hangs children off several attributes (``parts``, ``list``,
    ``command``) depending on the construct, so the walk asks for all of
    them rather than special-casing each compound type.
    """
    out: list[Node] = []
    for attr in ("parts", "list", "command"):
        value: object = getattr(node, attr, None)
        if value is None:
            continue
        items: list[object] = (
            cast(list[object], value) if isinstance(value, list) else [value]
        )
        out.extend(
            cast(Node, child)
            for child in items
            if child is not node and hasattr(child, "kind")
        )
    return out


def parse_bash(command: str) -> tuple[Node, ...] | None:
    """Parse ``command`` into bashlex trees, or ``None`` on failure.

    Returns the top-level tree tuple. Empty input or any bashlex
    failure (parse error, unsupported construct) yields ``None``.

    Args:
      command: Shell command string to parse.

    Returns:
      trees: Tuple of top-level AST nodes, or None on failure.

    """
    try:
        trees = cast(
            Sequence[Node],
            bashlex.parse(command),
        )
    except (bashlex.errors.ParsingError, NotImplementedError, AttributeError):
        return None
    except Exception:  # noqa: BLE001 -- bashlex raises various; treat all as parse fail
        return None
    if not trees:
        return None
    return tuple(trees)


def cached_parse_bash(
    command: str,
    cache: BashParseCache,
) -> tuple[Node, ...] | None:
    """Parse ``command`` once per cache lifetime; reuse on subsequent calls.

    Pass a per-request dict (e.g. ``ToolState.bash_parse_cache``). The
    cache stores ``None`` for unparseable inputs as well, so a second
    call with the same string is a single dict lookup.

    Args:
      command: Shell command string to parse.
      cache: Mutable dict for caching parse results.

    Returns:
      trees: Tuple of AST nodes, or None on failure.

    """
    if command in cache:
        return cache[command]
    trees = parse_bash(command)
    cache[command] = trees
    return trees


def cwd_is_known(cwd: str) -> bool:
    """Whether a ``cd`` prefix names a directory the command text reveals.

    ``cd`` (to ``$HOME``) and ``cd -`` (to ``$OLDPWD``) move somewhere
    real that the text does not name, which is distinct both from "no
    ``cd``" and from a literal path. A renderer must drop its worked
    example rather than resolve a relative operand against a directory
    it cannot identify.

    Args:
      cwd: The ``cwd`` field of an :class:`Invocation`.

    Returns:
      known: False only for a destination the text does not reveal.

    """
    return cwd != _UNKNOWN_CWD


def resolve_cwd_path(cwd: str | None, path: str | None) -> str:
    """Combine a ``cd`` prefix with a tool's positional path arg.

    Returns ``""`` when the effective path is just ``"."`` so callers
    can omit ``path=`` from the suggestion entirely.

    ``Invocation.cwd`` spells "no ``cd``" as ``""``, so an empty string is
    treated exactly like ``None``. Without that, every un-prefixed
    invocation rendered ``/f`` for ``cat f`` -- an absolute path at the
    filesystem root.

    Args:
      cwd: Directory from a ``cd`` prefix, or None/"" for none.
      path: Positional path argument, or None.

    Returns:
      resolved: Combined path string, or empty string.

    """
    if cwd == _UNKNOWN_CWD:
        # ``cd``/``cd -`` moved somewhere the command text does not name.
        # A relative operand cannot be resolved against it, and guessing
        # names the wrong file; an absolute one is unaffected.
        return str(path) if path and Path(path).is_absolute() else ""
    if not cwd:
        return "" if path in (None, "", ".") else str(path)
    if path in (None, "", "."):
        return cwd
    assert path is not None
    if Path(path).is_absolute():
        return path
    return f"{cwd.rstrip('/')}/{path}"


def is_read_only(trees: Sequence[Node]) -> bool:
    """Return True iff the parsed command cannot mutate state.

    False on any unrecognized utility, redirection, or control-flow
    construct. Empty input → False (caller should ``parse_bash``
    first; ``None`` from there means unparseable, which is also unsafe).

    Args:
      trees: Top-level bashlex AST nodes.

    Returns:
      safe: True if all commands are classified as read-only.

    """
    if not trees:
        return False
    return all(_is_node_safe(t) for t in trees)


#
# Lets the agent batch consecutive Bash calls in parallel when their
# commands cannot mutate state.
#
# Conservative by design: any unrecognized utility, any redirection,
# any command/process substitution whose inner command is not itself
# read-only - all return False. False positives mean serial dispatch
# (slow), false negatives mean concurrent writes (broken). Bias hard
# toward False.

_READ_ONLY_BASE: frozenset[str] = frozenset(
    {
        # BASH_SEARCH_COMMANDS
        "find",
        "grep",
        "rg",
        "ag",
        "ack",
        "locate",
        "which",
        "whereis",
        # BASH_READ_COMMANDS
        "cat",
        "head",
        "tail",
        "less",
        "more",
        "wc",
        "stat",
        "file",
        "strings",
        "jq",
        "awk",
        "cut",
        "sort",
        "uniq",
        "tr",
        # BASH_LIST_COMMANDS
        "ls",
        "tree",
        "du",
        # BASH_SEMANTIC_NEUTRAL_COMMANDS
        "echo",
        "printf",
        "true",
        "false",
        ":",
    }
)

_READ_ONLY_EXTRA: frozenset[str] = frozenset(
    {
        # Filesystem inspection.
        "pwd",
        "df",
        "realpath",
        "readlink",
        "basename",
        "dirname",
        # Text inspection.
        "column",
        "fold",
        "nl",
        "tac",
        # NOTE: egrep/fgrep/zgrep are deliberately ABSENT. They are wrapper
        # scripts, not binaries: strace shows egrep/fgrep exec grep, and zgrep
        # execs both grep and gzip. An allow-list of executable NAMES cannot
        # see what a wrapper runs.
        # Output / no-op.
        "yes",
        "test",
        "[",
        # System info.
        "env",
        "printenv",
        "date",
        "whoami",
        "id",
        "uname",
        "hostname",
        "uptime",
        "tty",
        "groups",
        "users",
        "w",
        # Lookup.
        "type",
        "command",
        "alias",
        # Stream transforms (output to stdout, never modify input file).
        "rev",
        "paste",
        "join",
        "comm",
        "diff",
        "cmp",
        # sed - read-only iff no in-place flag (see sed_mutates).
        "sed",
        # Hashing.
        "md5sum",
        "sha1sum",
        "sha256sum",
        "sha512sum",
        "cksum",
        "b2sum",
        # Encoding (stdout only).
        "base64",
        "od",  # codespell:ignore od
        "xxd",
        "hexdump",
        # Process inspection.
        "ps",
        "pgrep",
        "top",
        "htop",
        # Static analyzers. Only ``ty`` is here: strace shows mypy writing
        # .mypy_cache/ and basedpyright writing /tmp/pyright-*/ on a bare
        # invocation, with no flag to gate. "Always mutates" is not a shape a
        # flag denylist can express, so they are not read-only at all.
        "ty",
    }
)

_READ_ONLY: frozenset[str] = _READ_ONLY_BASE | _READ_ONLY_EXTRA

# `find` and `xargs` are read-only IFF none of these flags appear.
# `xargs` runs an arbitrary command, so we must inspect its tail -
# but that requires re-classifying the rest as a command, which the
# AST doesn't model. Easiest: deny any xargs.
FIND_DENY_FLAGS: frozenset[str] = frozenset(
    {
        "-delete",
        "-exec",
        "-execdir",
        "-ok",
        "-okdir",
        "-fprint",
        "-fprint0",
        "-fprintf",
        "-fls",
    }
)

# Only ``ty`` remains read-only (see _READ_ONLY_EXTRA); the others write
# unconditionally. Its own write flags still need gating.
_TYPE_CHECKERS: frozenset[str] = frozenset({"ty"})

# Utilities whose argv IS another command. They are read-only only if
# what they run is.
_ARGV_EXECUTORS: frozenset[str] = frozenset({"env", "command", "nice", "stdbuf"})

_ENV_ASSIGNMENT: Final = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

# ``sed`` writes without ``-i`` via the ``w`` command and the ``w`` flag on
# ``s///``, and EXECUTES via the ``e`` command and the ``e`` flag -- ``sed 'e'``
# runs the pattern space as a shell command, so a file's contents become code.
# The write half was modelled and the execute half was not, which is the more
# dangerous of the two.
#
# ``\n`` is a command separator exactly like ``;``: ``sed $'1p\nw out'`` writes
# ``out`` from its second command, which a pattern anchored on ``;`` alone
# never saw.
_SED_WRITE_SCRIPT: Final = re.compile(r"(^|;|\n|\})\s*\d*\s*[we]\b|s/.*/.*/[a-z]*[we]")

# ``awk`` can shell out or redirect from inside its program text. The redirect
# alternative is ANY unparenthesized ``>``/``>>`` in the program, not one
# adjacent to ``print``: `{ print $0 > "out" }` puts the whole record between
# them, and that spelling wrote a file while classifying read-only. Comparison
# operators inside a condition (``$1 > 5``) are the false-positive cost, which
# is the safe direction for a fail-closed classifier.
#
# A pipe is an escape whatever follows it: ``cmd | getline`` runs ``cmd``
# through the shell, and the next token there is a NAME, not the quote a
# ``print | "cmd"`` spelling would show.
_AWK_ESCAPE: Final = re.compile(r"\b(system|close|fflush)\s*\(|>|\|")

# Flags that supply a program from a FILE rather than from argv. The
# gates above analyse program TEXT, and this is the spelling that puts
# the text where no argv regex can look -- so it cannot be analysed at
# all and must fail closed. Same shape as mypy/basedpyright above:
# "unanalysable" is not something a flag denylist can wave through.
_PROGRAM_FILE_FLAGS: frozenset[str] = frozenset(
    {"-f", "--file", "--expression-file", "--source-file"}
)

# Utilities whose behaviour is a PROGRAM, analysed above as text. These
# are exactly the ones for which a program-from-file flag is fatal.
_PROGRAM_TEXT_UTILITIES: frozenset[str] = frozenset({"sed", "awk"})

# Flags that consume the following word as their value. Without
# these, ``git -C /repo log`` skips ``-C`` but leaves ``/repo`` in
# ``tail`` and the subcommand match fails. Per-command because
# value-flag conventions are not portable across tools.
_VALUE_FLAGS: dict[str, frozenset[str]] = {
    "git": frozenset({"-C", "--git-dir", "--work-tree"}),
    "uv": frozenset({"--project", "--directory", "--python", "--with"}),
}

# Exec wrappers: ``<wrapper> [flags] <subcommand> <INNER> [ARGS]``
# delegates classification to ``<INNER> [ARGS]``. This means e.g.
# ``uv run basedpyright --createstubs`` is rejected because
# ``basedpyright``'s own flag-gate catches ``--createstubs``.
_EXEC_WRAPPERS: Final[dict[str, tuple[str, ...]]] = {
    "uv": ("run",),
}

# Commands whose subcommand path gates safety. Each prefix is a
# tuple of words; the command's positional args (post-flag-skip)
# must START with one of these tuples. Anything else
# (``git push``, ``gh pr create``) is rejected.
#
# Use multi-word tuples wherever a bare single-word prefix would
# admit a mutation (e.g. ``uv cache dir`` instead of ``uv cache``,
# since ``uv cache`` alone would match ``uv cache clean``).
_SUBCOMMAND: dict[str, tuple[tuple[str, ...], ...]] = {
    # -- VCS / container CLIs ----------------------------------------
    "git": tuple(
        (w,)
        for w in (
            "status",
            "log",
            "diff",
            "show",
            "blame",
            "branch",  # refined by _subcommand_extra_safe (no positionals)
            "rev-parse",
            "rev-list",
            "ls-files",
            "ls-tree",
            "ls-remote",
            "describe",
            "reflog",  # refined (deny expire/delete/exists)
            "shortlog",
            "cat-file",
            "tag",  # refined (no positionals)
            "for-each-ref",
            "name-rev",
            "merge-base",
            "symbolic-ref",
            "whatchanged",
            "version",
            "help",
            "grep",
        )
    ),
    "gh": (
        ("status",),
        ("version",),
        ("help",),
        ("auth", "status"),
        ("pr", "view"),
        ("pr", "list"),
        ("pr", "status"),
        ("pr", "checks"),
        ("pr", "diff"),
        ("issue", "view"),
        ("issue", "list"),
        ("issue", "status"),
        ("run", "view"),
        ("run", "list"),
        ("run", "watch"),
        ("release", "view"),
        ("release", "list"),
        ("repo", "view"),
        ("repo", "list"),
        ("workflow", "view"),
        ("workflow", "list"),
        ("label", "list"),
        ("search", "code"),
        ("search", "issues"),
        ("search", "prs"),
        ("search", "repos"),
    ),
    "docker": tuple(
        (w,)
        for w in (
            "ps",
            "images",
            "inspect",
            "logs",
            "version",
            "info",
            "history",
            "search",
            "top",
            "diff",
            "stats",
            "events",
        )
    ),
    # -- Our additions -----------------------------------------------
    "kubectl": tuple(
        (w,)
        for w in (
            "get",
            "describe",
            "logs",
            "version",
            "top",
            "explain",
            "api-resources",
            "api-versions",
            "cluster-info",
        )
    ),
    "npm": tuple(
        (w,)
        for w in (
            "list",
            "ls",
            "view",
            "show",
            "outdated",
            "audit",  # refined (deny next "fix")
            "search",
            "doctor",
            "help",
            "ping",
            "root",
            "prefix",
            "explain",
        )
    ),
    "pip": tuple((w,) for w in ("list", "show", "freeze", "check", "search")),
    # uv: `uv run X` goes through _EXEC_WRAPPERS (delegates to X).
    # `uv cache` and `uv python` use multi-word prefixes to admit
    # only their read-only subsubcommands.
    "uv": (
        ("tree",),
        ("version",),
        ("cache", "dir"),
        ("python", "list"),
        ("python", "find"),
    ),
    "cargo": tuple(
        (w,)
        for w in (
            "tree",
            "search",
            "version",
            "metadata",
            "verify-project",
        )
    ),
    "brew": tuple(
        (w,)
        for w in (
            "list",
            "info",
            "search",
            "outdated",
            "leaves",
            "deps",
        )
    ),
    "yarn": tuple((w,) for w in ("list", "info", "outdated", "why")),
    "poetry": tuple((w,) for w in ("show", "check", "search", "version")),
    "go": tuple((w,) for w in ("version", "env", "doc", "list")),
}

# For these (command, subcommand) pairs, reject if the token
# immediately following the matched subcommand is in the deny set.
_SUBCOMMAND_NEXT_DENY: dict[tuple[str, str], frozenset[str]] = {
    ("git", "reflog"): frozenset({"expire", "delete", "exists"}),
    ("npm", "audit"): frozenset({"fix"}),
}

# `git branch|tag` are safe for *listing* but mutate when given a
# ref name positional (create/delete/rename). Flags are fine;
# value-flags (``--contains REF``) consume their value.
_GIT_BRANCH_TAG_VALUE_FLAGS: frozenset[str] = frozenset(
    {
        "--contains",
        "--no-contains",
        "--points-at",
        "--sort",
        "--format",
        "--merged",
        "--no-merged",
    },
)

# Node kinds that always indicate a write or unsafe operation.
_UNSAFE_NODE_KINDS: frozenset[str] = frozenset({"redirect"})

# Syntax that carries no command of its own. Everything NOT listed here
# is recursed into, so a construct bashlex adds later fails closed
# instead of disappearing from an allow-list comprehension.
_GLUE_NODE_KINDS: frozenset[str] = frozenset(
    {"operator", "pipe", "reservedword", "word", "assignment"},
)


def _parse_command(node: Node) -> Command:
    """Convert a bashlex ``command`` node into a :class:`Command` record."""
    env: dict[str, str] = {}
    words: list[str] = []
    captures_stdout = False
    for p in node.parts:
        kind = p.kind
        if kind == "redirect":
            # bashlex: ``.input`` is the source fd (``None`` = default =
            # stdout for ``>``/``>>``, stdin for ``<``). We flag only
            # redirects that divert fd 1 (stdout) - stderr redirects
            # like ``2>&1`` or ``2>/dev/null`` leave stdout untouched.
            input_fd = cast(int | None, getattr(p, "input", None))
            type_ = cast(str, getattr(p, "type", ""))
            if type_ in (">", ">>", ">&", ">|") and input_fd in (None, 1):
                captures_stdout = True
            continue
        if kind == "assignment" and not words:
            # Only leading assignments are env prefix; anything after
            # the first word is a regular positional (argv[n]).
            k, _, v = p.word.partition("=")
            env[k] = v
            continue
        if kind == "word":
            words.append(p.word)
    if not words:
        return Command(exe="", args=(), env_prefix=env, captures_stdout=captures_stdout)
    return Command(
        exe=words[0],
        args=tuple(words[1:]),
        env_prefix=env,
        captures_stdout=captures_stdout,
    )


# Utilities whose SECOND positional operand is the output file. POSIX gives
# ``uniq`` and ``xxd`` an optional ``output`` operand, so no flag exists to
# deny -- a flag-shaped gate can never reach these, which is why they read
# safe while overwriting the named file.
_OPERAND_WRITERS: frozenset[str] = frozenset({"uniq", "xxd"})


def _mutating_flags(exe: str, args: list[str]) -> bool:
    """Whether an otherwise read-only utility was asked to write.

    Each entry is a utility that reads by default but has a documented
    write mode; the allow-list keys on the executable alone, so without
    these gates ``sort -o victim`` and ``sed 'w victim'`` both read safe.

    Verified by tracing every allow-listed utility for writes outside the
    system paths and for child ``execve``. Reading man pages was not enough:
    ``--output-separator`` names no file, while ``uniq``'s output operand
    carries no flag at all.
    """
    # ``-f prog`` supplies the program from a FILE, so every gate below
    # that analyses program TEXT is looking somewhere the program is not.
    # Unanalysable is not read-only.
    if exe in _PROGRAM_TEXT_UTILITIES and any(
        a in _PROGRAM_FILE_FLAGS or a.partition("=")[0] in _PROGRAM_FILE_FLAGS
        for a in args
    ):
        return True
    if exe in _OPERAND_WRITERS:
        return len([a for a in args if not a.startswith("-")]) >= 2
    if exe == "tree":
        # ``-J``/``-X`` select JSON/XML on STDOUT and write nothing;
        # measured leaving the sandbox byte-identical. Only ``-o`` names
        # a file.
        return any(a.startswith("-o") for a in args)
    if exe == "find":
        return any(a in FIND_DENY_FLAGS for a in args)
    if exe == "sed":
        return sed_mutates(args) or any(_SED_WRITE_SCRIPT.search(a) for a in args)
    if exe in _TYPE_CHECKERS:
        return _type_checker_mutates(args)
    if exe == "sort":
        # ``-oFILE`` attaches the value to the flag, so an equality test never
        # matched it: `sort -oout input` classified read-only and wrote `out`.
        return any(a.startswith(("-o", "--output")) for a in args)
    if exe == "awk":
        # ``system()``/``print > file`` make awk a general executor; the
        # program text is not something this classifier can analyse.
        return any(_AWK_ESCAPE.search(a) for a in args)
    return False


def sed_mutates(args: Sequence[str]) -> bool:
    """Return True iff any arg requests in-place editing.

    Catches ``--in-place``/``--in-place=SUFFIX``, the short form ``-i``
    (optionally with a backup suffix like ``-i.bak``), and combined
    short flags that include ``i`` (e.g. ``-ni``, ``-Ei``).
    """
    for a in args:
        if a == "--in-place" or a.startswith("--in-place="):
            return True
        if a.startswith("-") and not a.startswith("--") and "i" in a[1:]:
            return True
    return False


def _type_checker_mutates(args: list[str]) -> bool:
    """True if type-checker args write files or install packages.

    ``--output``/``-o`` and the report formats are here because a checker
    that reads source still writes when told where to put its report:
    ``--junit-xml`` and ``--gitlabcodequality`` were both measured writing
    a file.
    """
    for a in args:
        if a.startswith(("--createstub", "--output", "--junit", "--gitlab")):
            return True
        if a in {"--writebaseline", "--install-types", "-o"}:
            return True
    return False


def _skip_leading_flags(exe: str, args: list[str]) -> int:
    """Index in ``args`` past all leading flags.

    Flags listed in :data:`_VALUE_FLAGS` for ``exe`` consume the
    following token as their value (e.g. ``--project .``).
    """
    value_flags = _VALUE_FLAGS.get(exe, frozenset())
    i = 0
    while i < len(args) and args[i].startswith("-"):
        takes_value = (
            args[i] in value_flags
            and i + 1 < len(args)
            and not args[i + 1].startswith("-")
        )
        i += 1
        if takes_value:
            i += 1
    return i


def _git_branch_or_tag_safe(args_after_sub: list[str]) -> bool:
    """True iff args contain no bare positional (positional → mutation).

    Handles value-flags by skipping the next token; otherwise any
    non-flag token is a bare positional and unsafe.
    """
    i = 0
    while i < len(args_after_sub):
        a = args_after_sub[i]
        if not a.startswith("-"):
            return False
        if a in _GIT_BRANCH_TAG_VALUE_FLAGS:
            i += 2
        else:
            i += 1
    return True


# Flags that make ANY git subcommand unsafe, whatever the allow-list says
# about the subcommand itself. ``--output``/``-o`` write a file; ``-O`` hands
# its argument to the shell as a pager, so `git grep -O"sh -c ..."` is
# arbitrary execution wearing the name of a read-only subcommand.
_GIT_UNSAFE_FLAGS: frozenset[str] = frozenset(
    {"--output", "-o", "-O", "-c", "--exec-path"}
)


def _git_unsafe_flag(args: tuple[str, ...]) -> bool:
    """Whether any git argument writes a file or runs a command.

    Checked over the WHOLE argv, not the post-subcommand tail: ``-c`` and
    ``-O`` are accepted before the subcommand, where a tail-only scan
    never looks.
    """
    return any(
        a in _GIT_UNSAFE_FLAGS
        or a.partition("=")[0] in _GIT_UNSAFE_FLAGS
        # Attached spellings: ``-Osh``, ``-oFILE``.
        or (len(a) > 2 and a[:2] in ("-O", "-o"))
        for a in args
    )


def _go_env_writes(tail: tuple[str, ...]) -> bool:
    """``go env -w K=V`` MUTATES the persistent go environment."""
    return any(a in ("-w", "-u") for a in tail)


def _subcommand_extra_safe(
    exe: str,
    prefix: tuple[str, ...],
    tail: tuple[str, ...],
) -> bool:
    """Apply post-match safety refinement for a matched subcommand.

    Called after the subcommand prefix has matched. Returns False to
    veto the match.
    """
    if exe == "git" and prefix in (("branch",), ("tag",)):
        return _git_branch_or_tag_safe(list(tail[len(prefix) :]))
    if exe == "go" and prefix == ("env",) and _go_env_writes(tail):
        return False
    deny = _SUBCOMMAND_NEXT_DENY.get((exe, prefix[0]))
    return not (
        deny is not None and len(tail) > len(prefix) and tail[len(prefix)] in deny
    )


def _command_words(node: Node) -> list[str]:
    """Return literal words of a command node, skipping assignments."""
    return [p.word for p in node.parts if p.kind == "word"]


def _is_command_safe(node: Node) -> bool:
    """Return True iff this CommandNode is a read-only invocation.

    Recurses into command/process substitutions inside word parts.
    """
    # Reject any redirect at this command level (writes, here-strings
    # that may carry substitutions, etc.).
    if any(p.kind in _UNSAFE_NODE_KINDS for p in node.parts):
        return False

    # Recurse into substitutions nested in word parts. A word like
    # `$(rm foo)` carries a CommandsubstitutionNode under its parts.
    for p in node.parts:
        if p.kind == "word":
            for sub in getattr(p, "parts", []) or []:
                if not _is_node_safe(sub):
                    return False

    words = _command_words(node)
    if not words:
        # Pure assignment (e.g. ``A=1``) - sets a shell var. Accept.
        return True

    return _classify(Path(words[0]).name, words[1:])


def _classify(exe: str, args: list[str]) -> bool:
    """Classify an ``exe args...`` invocation as read-only.

    Split out so exec wrappers (``uv run basedpyright``) can recurse:
    the inner command gets the full safety check, including its own
    flag-gates.
    """
    wrapper_subcmd = _EXEC_WRAPPERS.get(exe)
    if wrapper_subcmd is not None:
        i = _skip_leading_flags(exe, args)
        kw_len = len(wrapper_subcmd)
        if (
            len(args) >= i + kw_len + 1
            and tuple(args[i : i + kw_len]) == wrapper_subcmd
        ):
            inner = args[i + kw_len :]
            return _classify(inner[0], inner[1:])

    # ``env FOO=1 rm x`` / ``command rm x`` EXECUTE their argument, so
    # allow-listing the wrapper without inspecting its payload launders
    # anything through it. Recurse, exactly as ``_EXEC_WRAPPERS`` does.
    if exe in _ARGV_EXECUTORS:
        payload = [a for a in args if not _ENV_ASSIGNMENT.match(a)]
        payload = payload[_skip_leading_flags(exe, payload) :]
        if not payload:
            # Bare ``env`` / ``command`` just prints; nothing runs.
            return True
        return _classify(Path(payload[0]).name, payload[1:])

    if exe in _READ_ONLY:
        return not _mutating_flags(exe, args)

    prefixes = _SUBCOMMAND.get(exe)
    if prefixes is not None and args:
        # Whole argv, not the post-subcommand tail: ``git -c core.pager=CMD``
        # and ``git grep -O CMD`` both run a command, and the first appears
        # BEFORE the subcommand where a tail-only scan never looks.
        if exe == "git" and _git_unsafe_flag(tuple(args)):
            return False
        i = _skip_leading_flags(exe, args)
        tail = tuple(args[i:])
        for pfx in prefixes:
            if len(tail) >= len(pfx) and tail[: len(pfx)] == pfx:
                return _subcommand_extra_safe(exe, pfx, tail)
        return False

    return False


def _is_node_safe(node: Node) -> bool:
    """Recursively classify a bashlex AST node as read-only."""
    kind = node.kind
    if kind == "command":
        return _is_command_safe(node)
    if kind in ("pipeline", "list"):
        # Operator/pipe/reservedword children are pure glue, and are
        # named explicitly rather than filtered by an allow-list of the
        # kinds we DO understand: an unlisted kind must reach the
        # fail-closed branch below, not vanish from the comprehension.
        return all(
            _is_node_safe(p) for p in node.parts if p.kind not in _GLUE_NODE_KINDS
        )
    if kind == "compound":
        # Subshells and group commands: recurse into the embedded list.
        # EVERY child is examined, not just the kinds we recognise --
        # filtering meant a ``for``/``while`` child was dropped and
        # ``all([])`` returned True, so an entire loop body went
        # unexamined and ``for f in victim; do rm $f; done`` read safe.
        return all(_is_node_safe(p) for p in node.list)
    if kind in ("commandsubstitution", "processsubstitution"):
        return _is_node_safe(node.command)
    # ``tilde`` / ``parameter`` are pure word expansions (``~/path``,
    # ``$VAR``, ``${VAR:-default}``) - no execution, no writes.
    # Everything else (function, for, while, if, case, select, and any
    # kind bashlex grows later) may hide arbitrary code. Fail CLOSED:
    # a classifier whose default is "safe" is not a classifier.
    return kind in ("tilde", "parameter")
