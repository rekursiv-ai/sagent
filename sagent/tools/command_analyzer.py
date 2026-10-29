"""Command analyzer: what a Bash command does, read before it runs.

``invocations`` splits a command into the simple commands it runs.
``ReadOnly`` says whether they only read, so the call may run beside others.
``rewrite`` says whether they are another tool's job, as ``cat`` is Read's;
a tool takes on such commands by defining ``from_shell``.

Typical usage example:

  match rewrite("cat -n a.py | head -20", peers=(Read(),)):
      case Rewrite(tool=tool, arguments=arguments):
          ...  # Call ``tool`` with ``arguments`` instead.
      case str() as message:
          ...  # Refuse the command; ``message`` names the right calls.
      case None:
          ...  # Run the command.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from importlib.resources import files
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Protocol

import itertools
import posixpath
import re

from treekle import PlainTree, from_plain, json_dumps

import yaml

from sagent.tools.types import Tool


if TYPE_CHECKING:
    from collections.abc import Sequence

    from bashlex.ast import (
        node as Node,  # noqa: N812 -- bashlex spells its class lowercase.
    )

    import bashlex

else:
    from wrapt import lazy_import

    bashlex = lazy_import("bashlex")  # 88 ms cold.


__all__ = [
    "Invocation",
    "ReadOnly",
    "Rewrite",
    "ShellTool",
    "bounding_sink",
    "drop_cd",
    "invocations",
    "line_count",
    "moves",
    "operands",
    "replaceable",
    "resolve",
    "rewrite",
    "sed_mutates",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class Invocation:
    """One simple command in a Bash line, with what surrounds it."""

    exe: str
    args: tuple[str, ...]
    cwd: str | None = ""
    """Where an earlier ``cd`` went: ``""`` for none, ``None`` if the text does not say."""

    pipeline: int = 0
    """Which pipeline on the line this is a stage of."""

    piped_into: Invocation | None = None
    """The stage this one's output feeds."""

    piped_from: Invocation | None = None
    """The stage feeding this one's input."""

    env_prefix: Mapping[str, str] = field(default_factory=dict[str, str])
    """Leading ``KEY=value`` words, as in ``LC_ALL=C grep``."""

    captures_stdout: bool = False
    """Whether a redirect sends its output to a file."""

    expands: bool = False
    """Whether bash fills in a word first, as ``$x`` or ``$(...)``; ``~`` is no expansion."""

    def downstream(self) -> Iterator[Invocation]:
        """Yield each stage this one's output flows through, in order."""
        stage = self.piped_into
        while stage is not None:
            yield stage
            stage = stage.piped_into


class ShellTool(Tool, Protocol):
    """A tool that can stand in for a shell command."""

    def from_shell(
        self,
        invocation: Invocation,
    ) -> Mapping[str, PlainTree] | str | None:
        """Return this tool's arguments for ``invocation``.

        Args:
          invocation: One simple command, with its pipeline.

        Returns:
          arguments: The call that does exactly what the command does; a
            message, when the command is this tool's job but no call matches
            it exactly; or ``None``, when it is not this tool's job.

        """
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class Rewrite:
    """A shell command, as a call of the tool whose job it is."""

    tool: ShellTool
    arguments: Mapping[str, PlainTree]


def rewrite(
    command: str,
    *,
    peers: Sequence[ShellTool],
    reads: bool = False,
) -> Rewrite | tuple[Rewrite, ...] | str | None:
    """Return what ``command`` should be instead of a shell command, if anything.

    Every simple command must be some peer's job, else the line does real
    shell work and runs. Exactly one, translated exactly and with no ``cd``,
    becomes that peer's call. Several become one call each when every one is
    translated exactly with no ``cd``, the line only ``reads``, and no ``||``
    makes one a fallback: run apart and at once, reads lose nothing. Anything
    else is refused with a message naming the calls to make.

    Args:
      command: The Bash command the model wrote.
      peers: The tools that may take it on.
      reads: Whether every command in it only reads.

    Returns:
      result: The call, or calls, to make instead; a refusal; or ``None`` to
        run it.

    """
    trees = _parse(command)
    if trees is None:
        return None
    found = _invocations(trees)
    claims: list[tuple[Invocation, ShellTool, Mapping[str, PlainTree] | str]] = []
    for invocation in found:
        for peer in peers:
            outcome = peer.from_shell(invocation)
            if outcome is not None:
                claims.append((invocation, peer, outcome))
                break
    claimed = {invocation.pipeline for invocation, _, _ in claims}
    if not claims or any(i.pipeline not in claimed for i in found):
        return None
    exact = [
        Rewrite(tool=peer, arguments=outcome)
        for invocation, peer, outcome in claims
        if not isinstance(outcome, str) and invocation.cwd == ""
    ]
    if len(exact) == len(claims) == 1:
        return exact[0]
    if len(exact) == len(claims) and reads and not _falls_back(trees):
        unique = {json_dumps({"t": r.tool.name, "a": r.arguments}): r for r in exact}
        return tuple(unique.values())
    lines = [
        outcome if isinstance(outcome, str) else f"{peer.name} {json_dumps(outcome)}"
        for _, peer, outcome in claims
    ]
    return (
        "This command is the work of other tools; call them instead, in one"
        " block:\n" + "\n".join(f"- {line}" for line in dict.fromkeys(lines))
    )


def invocations(command: str) -> tuple[Invocation, ...] | None:
    """Return every simple command in ``command``, in order, with its context.

    Lists (``&&``, ``;``), pipelines, loop and conditional bodies, and command
    substitutions are all descended; a ``cd`` is folded into the ``cwd`` of
    what follows it rather than listed.

    Args:
      command: A Bash command.

    Returns:
      invocations: Each simple command, or ``None`` when ``command`` does not
        parse.

    """
    trees = _parse(command)
    return None if trees is None else _invocations(trees)


@dataclass(frozen=True, slots=True, kw_only=True)
class ReadOnly:
    """Which commands only read, so a call running only them may run beside others.

    An unknown command, a redirect, a loop, or an unparsed line counts as
    writing: a wrong ``False`` only serializes, a wrong ``True`` races writers.
    """

    utilities: frozenset[str]
    """Commands that only read, unless a ``write_flags`` entry says otherwise."""

    executors: frozenset[str]
    """Commands that run their arguments as a command: ``env FOO=1 rm x``."""

    wrappers: Mapping[str, str]
    """Commands whose subcommand runs the rest as a command: ``uv run X``."""

    leading_flags: Mapping[str, frozenset[str]]
    """Flags before a subcommand that take the next word: ``git -C dir log``."""

    subcommands: Mapping[str, tuple[str, ...]]
    """Subcommands that only read, as their words: ``gh`` has ``pr view``."""

    write_flags: Mapping[str, tuple[str, ...]]
    """Flag prefixes that make a command, or ``"command subcommand"``, write."""

    write_words: Mapping[str, tuple[str, ...]]
    """Words that make a subcommand write when they follow it: ``reflog expire``."""

    listings: Mapping[str, frozenset[str]]
    """Subcommands that list, but write given a name; with their value flags."""

    output_operands: frozenset[str]
    """Commands whose second operand is a file they write: ``uniq in out``."""

    @classmethod
    def default(cls) -> ReadOnly:
        """Return the policy in ``assets/bash.yaml``.

        Returns:
          policy: The commands traced as only reading.

        Raises:
          ReadError: The asset does not fit this class.

        """
        text = (files(__package__) / "assets" / "bash.yaml").read_text(encoding="utf-8")
        return from_plain(yaml.safe_load(text), cls)

    def reads(self, command: str) -> bool:
        """Return whether ``command`` only reads.

        Args:
          command: A Bash command.

        Returns:
          reads: ``True`` only when every command in it is known to only read.

        """
        trees = _parse(command)
        return trees is not None and all(self._safe(tree) for tree in trees)

    def _safe(self, node: Node) -> bool:
        """Return whether ``node`` runs only commands that read."""
        if node.kind == "command":
            return self._safe_command(node)
        if node.kind in {"pipeline", "list"}:
            # Glue kinds are listed, so a kind bashlex adds later fails closed.
            glue = {"operator", "pipe", "reservedword", "word", "assignment"}
            return all(self._safe(part) for part in node.parts if part.kind not in glue)
        if node.kind == "compound":
            return all(self._safe(part) for part in node.list)
        if node.kind in {"commandsubstitution", "processsubstitution"}:
            return self._safe(node.command)
        # `bashlex` leaves an expansion's text unparsed, so the ``rm`` that
        # ``${x:-$(rm f)}`` runs is in no node of its own.
        if node.kind == "parameter":
            return not re.search(r"\$\(|`", node.value)
        # A tilde runs nothing; a loop, conditional, or function may.
        return node.kind == "tilde"

    def _safe_command(self, node: Node) -> bool:
        """Return whether one simple command only reads."""
        if any(part.kind == "redirect" for part in node.parts):
            return False
        for part in node.parts:
            if part.kind in {"word", "assignment"} and not all(
                self._safe(sub)
                for sub in (part.parts if hasattr(part, "parts") else ())
            ):
                return False
        words = [part.word for part in node.parts if part.kind == "word"]
        # A bare assignment sets a shell variable.
        return not words or self._reads(PurePosixPath(words[0]).name, words[1:])

    def _reads(self, exe: str, args: list[str]) -> bool:
        """Return whether ``exe args`` only reads."""
        start = self._skip_flags(exe, args)
        wrapper = self.wrappers.get(exe)
        if wrapper is not None and args[start : start + 1] == [wrapper]:
            inner = args[start + 1 :]
            return bool(inner) and self._reads(PurePosixPath(inner[0]).name, inner[1:])
        if exe in self.executors:
            payload = [arg for arg in args if not re.match(r"[A-Za-z_]\w*=", arg)]
            payload = payload[self._skip_flags(exe, payload) :]
            return not payload or self._reads(
                PurePosixPath(payload[0]).name,
                payload[1:],
            )
        if self._writes(exe, args):
            return False
        if exe in self.utilities:
            return True
        rest = args[start:]
        for subcommand in self.subcommands.get(exe, ()):
            words = subcommand.split()
            if rest[: len(words)] == words:
                return self._subcommand_reads(f"{exe} {subcommand}", rest[len(words) :])
        return False

    def _subcommand_reads(self, path: str, rest: list[str]) -> bool:
        """Return whether a read-only subcommand ``path`` stays so given ``rest``."""
        if self._flagged(path, rest) or (
            any(word in self.write_words.get(path, ()) for word in rest[:1])
        ):
            return False
        takes_value = self.listings.get(path)
        i = 0
        while takes_value is not None and i < len(rest):
            if not rest[i].startswith("-"):
                return False
            i += 2 if rest[i] in takes_value else 1
        return True

    def _writes(self, exe: str, args: list[str]) -> bool:
        """Return whether an otherwise read-only command was asked to write."""
        if self._flagged(exe, args):
            return True
        if exe in self.output_operands:
            return len([a for a in args if not a.startswith("-")]) >= 2
        if exe not in {"sed", "awk"}:
            return False
        # ``sed`` and ``awk`` write from their program text; one in a file
        # cannot be checked.
        program_file = {"-f", "--file", "--expression-file", "--source-file"}
        if any(a.partition("=")[0] in program_file for a in args):
            return True
        if exe == "sed":
            # ``w`` writes and ``e`` executes, as commands or as ``s///``
            # flags; a newline separates commands as ``;`` does.
            script = re.compile(r"(^|;|\n|\})\s*\d*\s*[we]\b|s/.*/.*/[a-z]*[we]")
            return sed_mutates(args) or any(script.search(a) for a in args)
        # ``system()``, ``print > f``, and ``cmd | getline`` reach outside.
        return any(re.search(r"\b(system|close|fflush)\s*\(|>|\|", a) for a in args)

    def _flagged(self, key: str, args: list[str]) -> bool:
        """Return whether any of ``args`` starts with a write flag of ``key``."""
        prefixes = self.write_flags.get(key, ())
        return bool(prefixes) and any(arg.startswith(prefixes) for arg in args)

    def _skip_flags(self, exe: str, args: Sequence[str]) -> int:
        """Return the index of the first argument past ``exe``'s leading flags."""
        takes_value = self.leading_flags.get(exe, frozenset[str]())
        i = 0
        while i < len(args) and args[i].startswith("-"):
            valued = (
                args[i] in takes_value
                and i + 1 < len(args)
                and not args[i + 1].startswith("-")
            )
            i += 2 if valued else 1
        return i


def operands(
    args: Sequence[str],
    *,
    takes_value: frozenset[str] = frozenset(),
) -> tuple[str, ...]:
    """Return the positional arguments in ``args``, without flags or their values.

    ``head -n 5 f`` has one operand, ``f``: the ``5`` belongs to ``-n``.

    Args:
      args: A command's arguments.
      takes_value: Its flags that take the next word, such as ``head``'s ``-n``.

    Returns:
      operands: The arguments that are operands.

    """
    out: list[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--":
            out.extend(args[i + 1 :])
            break
        if arg.startswith("-") and arg != "-":
            i += 2 if arg in takes_value and i + 1 < len(args) else 1
            continue
        out.append(arg)
        i += 1
    return tuple(out)


def replaceable(
    invocation: Invocation,
    *,
    exes: frozenset[str],
    deny: frozenset[str] = frozenset(),
    takes_value: frozenset[str] = frozenset(),
    bare: bool = False,
    fed: bool = False,
    counted: bool = False,
) -> bool:
    """Return whether one tool call could do what ``invocation`` does.

    The command must be one of ``exes``, without an environment prefix or a
    redirect, and its output may pass only through stages that cut it short:
    ``head``, ``tail``, ``cat``, ``less``, ``more``.

    Args:
      invocation: One simple command, with its pipeline.
      exes: The commands the caller's tool takes on.
      deny: Flags that need the shell, such as ``head -c``, which counts bytes.
      takes_value: The command's flags that take the next word.
      bare: Whether the command names a target with no operand, as ``ls`` does.
      fed: Whether it may read a pipe; the caller then checks what feeds it.
      counted: Whether ``| wc -l`` after it is something the tool can say.

    Returns:
      replaceable: ``True`` when one call covers it.

    """
    if (
        invocation.exe not in exes
        or invocation.env_prefix
        or invocation.captures_stdout
    ):
        return False
    if any(_sink_blocks(sink, counted=counted) for sink in invocation.downstream()):
        return False
    if any(_denied(arg, deny, takes_value=takes_value) for arg in invocation.args):
        return False
    if invocation.piped_from is not None and not fed:
        return False
    return bare or bool(operands(invocation.args, takes_value=takes_value))


def bounding_sink(invocation: Invocation) -> Invocation | None:
    """Return the ``head`` or ``tail`` that cuts ``invocation``'s output, if any.

    ``ls | cat | head -5`` is cut as ``ls | head -5`` is: ``cat``, ``less``,
    and ``more`` pass everything through.

    Args:
      invocation: The command whose output is cut.

    Returns:
      sink: The first ``head`` or ``tail`` downstream, past pass-through stages.

    """
    for stage in invocation.downstream():
        if stage.exe in {"head", "tail"}:
            return stage
        if not _passes_through(stage.exe):
            return None
    return None


def line_count(args: Sequence[str]) -> int | None:
    """Return how many lines a ``head`` or ``tail`` with ``args`` keeps.

    Args:
      args: Its arguments: none, ``-n N``, ``-nN``, or ``-N``.

    Returns:
      count: The lines kept, 10 when none is given; ``None`` for any other
        spelling, such as ``-n +5``.

    """
    count = 10
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "-n" and i + 1 < len(args) and args[i + 1].isdigit():
            count, i = int(args[i + 1]), i + 2
        elif arg.startswith("-n") and arg[2:].isdigit():
            count, i = int(arg[2:]), i + 1
        elif arg[:1] == "-" and arg[1:].isdigit():
            count, i = int(arg[1:]), i + 1
        else:
            return None
    return count or None


def resolve(cwd: str | None, path: str) -> str | None:
    """Return ``path`` as the shell would find it after ``cd cwd``.

    Args:
      cwd: An ``Invocation.cwd``.
      path: An operand; ``""`` for none.

    Returns:
      path: The path to name in a tool call, ``""`` for the current directory,
        or ``None`` when the ``cd`` went somewhere the text does not say.

    """
    # ``~`` names a home directory wherever the shell stands; the tools expand it.
    if path and (PurePosixPath(path).is_absolute() or path.startswith("~")):
        return path
    if cwd is None:
        return None
    if path in {"", "."}:
        return cwd
    return str(PurePosixPath(cwd) / path) if cwd else path


def drop_cd(command: str, *, cwd: str) -> str:
    """Return ``command`` without a leading ``cd`` to ``cwd``, where it starts anyway.

    Only ``cd DIR && ...``, ``cd DIR; ...``, or ``cd DIR`` on a line of its own
    before more, with ``DIR`` a plain word naming ``cwd``; anything else is
    returned unchanged.

    Args:
      command: A Bash command.
      cwd: Where it will start.

    Returns:
      command: What follows the ``cd``, or ``command`` itself.

    """
    trees = _parse(command)
    if trees is None:
        return command
    first = trees[0]
    if first.kind == "list":
        parts = first.parts
        if (
            len(parts) < 3
            or parts[1].kind != "operator"
            or parts[1].op not in {"&&", ";"}
        ):
            return command
        cd, rest = parts[0], parts[2]
    elif first.kind == "command" and len(trees) > 1:
        cd, rest = first, trees[1]
    else:
        return command
    if not _cd_to(cd, cwd):
        return command
    return command[rest.pos[0] :]


def moves(command: str) -> bool:
    """Return whether ``command`` may change the directory the shell ends in.

    Args:
      command: A Bash command.

    Returns:
      moves: ``True`` for any command that moves the shell or runs text as
        shell (``cd``, ``pushd``, ``popd``, ``source``, ``.``, ``eval``, ``exec``,
        through ``builtin`` or ``command``), one whose name is expanded, or a
        line that does not parse.

    """
    trees = _parse(command)
    if trees is None:
        return True
    pending = list(trees)
    while pending:
        node = pending.pop()
        if node.kind == "command":
            words = [p.word for p in node.parts if p.kind == "word"]
            while words and words[0] in {"builtin", "command"} and len(words) > 1:
                words = words[1:]
            if words and (
                words[0] in {"cd", "pushd", "popd", "source", ".", "eval", "exec"}
                or not _literal(words[0])
            ):
                return True
        pending.extend(_children(node))
    return False


def sed_mutates(args: Sequence[str]) -> bool:
    """Return whether ``sed`` with ``args`` edits a file in place.

    Args:
      args: Its arguments.

    Returns:
      mutates: ``True`` for ``-i``, ``-i.bak``, ``--in-place``, or a cluster
        such as ``-ni``.

    """
    return any(
        arg == "--in-place"
        or arg.startswith("--in-place=")
        or (arg.startswith("-") and not arg.startswith("--") and "i" in arg[1:])
        for arg in args
    )


def _parse(command: str) -> tuple[Node, ...] | None:
    """Return ``command``'s syntax trees, or ``None`` when bashlex cannot read it."""
    try:
        trees = bashlex.parse(command)
    # `bashlex` raises more than ``ParsingError`` on what it cannot read.
    except Exception:  # noqa: BLE001 -- Any parser failure means "unknown", which fails closed.
        return None
    return tuple(trees) or None


# `bashlex` hangs ``(grep p f) > out``'s redirect on the compound; ``captured``
# carries it down to the commands inside.
def _walk(
    node: Node,
    *,
    cwd: str | None,
    out: list[Invocation],
    pipelines: Iterator[int],
    captured: bool = False,
    sink: Invocation | None = None,
) -> None:
    """Append every simple command under ``node`` to ``out``."""
    captured = captured or _redirects_stdout(node)
    if node.kind == "command":
        command = _command(node, cwd=cwd, pipeline=next(pipelines))
        if command.exe:
            out.append(
                replace(
                    command,
                    piped_into=sink,
                    captures_stdout=command.captures_stdout or captured,
                ),
            )
        _walk_substitutions(node, cwd=cwd, out=out, pipelines=pipelines)
    elif node.kind == "pipeline":
        _walk_pipeline(node, cwd=cwd, out=out, pipelines=pipelines, captured=captured)
    elif node.kind == "list":
        _walk_list(node, cwd=cwd, out=out, pipelines=pipelines, captured=captured)
    else:
        # Loop bodies too: ``for f in *; do cat $f; done`` is still Read's job.
        for child in _children(node):
            _walk(
                child,
                cwd=cwd,
                out=out,
                pipelines=pipelines,
                captured=captured,
                sink=sink,
            )


# A ``cd`` holds for the rest of the list, except after ``||`` (it failed) or
# ``&`` (it ran in a subshell).
def _walk_list(
    node: Node,
    *,
    cwd: str | None,
    out: list[Invocation],
    pipelines: Iterator[int],
    captured: bool,
) -> None:
    """Walk ``A && B ; C``, carrying each ``cd`` into what follows it."""
    here = before = cwd
    for part in node.parts:
        if part.kind == "operator":
            if part.op in {"||", "&"}:
                here = before
            continue
        if part.kind == "command":
            command = _command(part, cwd=here, pipeline=-1)
            if command.exe == "cd" and not command.captures_stdout:
                before = here
                here = _cd(here, command.args)
                continue
        before = here
        _walk(part, cwd=here, out=out, pipelines=pipelines, captured=captured)


def _walk_pipeline(
    node: Node,
    *,
    cwd: str | None,
    out: list[Invocation],
    pipelines: Iterator[int],
    captured: bool,
) -> None:
    """Walk ``A | B | C``, linking each stage to its neighbours."""
    pipeline = next(pipelines)
    stages = [part for part in node.parts if part.kind != "pipe"]
    linked: list[Invocation | None] = [None] * len(stages)
    for i in reversed(range(len(stages))):
        after = linked[i + 1] if i + 1 < len(stages) else None
        if stages[i].kind != "command":
            # ``(sort)`` transforms what passes; no tool claims this name.
            linked[i] = Invocation(
                exe="\0compound",
                args=(),
                cwd=cwd,
                pipeline=pipeline,
                piped_into=after,
            )
            continue
        command = _command(stages[i], cwd=cwd, pipeline=pipeline)
        if command.exe:
            # Only the last stage's output reaches an enclosing redirect.
            last = i == len(stages) - 1
            linked[i] = replace(
                command,
                piped_into=after,
                captures_stdout=command.captures_stdout or (captured and last),
            )
    previous: Invocation | None = None
    for i, stage in enumerate(stages):
        invocation = linked[i]
        if invocation is not None:
            # The copy, not ``invocation``, so each stage's source has its own.
            previous = replace(invocation, piped_from=previous)
            out.append(previous)
        if stage.kind == "command":
            _walk_substitutions(stage, cwd=cwd, out=out, pipelines=pipelines)
        else:
            # What runs inside a compound stage feeds what the stage feeds.
            _walk(
                stage,
                cwd=cwd,
                out=out,
                pipelines=pipelines,
                captured=captured,
                sink=linked[i + 1] if i + 1 < len(stages) else None,
            )


def _walk_substitutions(
    node: Node,
    *,
    cwd: str | None,
    out: list[Invocation],
    pipelines: Iterator[int],
) -> None:
    """Walk the ``$(...)`` nested in a command's words and assignments."""
    # A command's children are its words, assignments, and redirects; the
    # substitutions hang under the first two.
    for child in _children(node):
        if child.kind in {"word", "assignment"}:
            _walk(child, cwd=cwd, out=out, pipelines=pipelines)


def _command(node: Node, *, cwd: str | None, pipeline: int) -> Invocation:
    """Return a ``command`` node as an ``Invocation``, not yet linked."""
    env: dict[str, str] = {}
    words: list[str] = []
    captures = expands = False
    for part in node.parts:
        if part.kind == "redirect":
            # Only a redirect of stdout hides output; ``2>&1`` does not.
            captures |= part.type in {">", ">>", ">&", ">|"} and part.input in {None, 1}
        elif part.kind == "assignment" and not words:
            key, _, value = part.word.partition("=")
            env[key] = value
        elif part.kind == "word":
            words.append(part.word)
            expands |= any(
                sub.kind != "tilde"
                for sub in (part.parts if hasattr(part, "parts") else ())
            )
    return Invocation(
        exe=words[0] if words else "",
        args=tuple(words[1:]),
        cwd=cwd,
        pipeline=pipeline,
        env_prefix=env,
        captures_stdout=captures,
        expands=expands,
    )


# Bare ``cd`` goes to ``$HOME`` and ``cd -`` to ``$OLDPWD``, neither in the text.
def _cd(cwd: str | None, args: tuple[str, ...]) -> str | None:
    """Return where a ``cd`` with ``args`` leaves the shell."""
    paths = [arg for arg in args if not arg.startswith("-") or arg == "-"]
    if not paths or paths[0] == "-":
        return None
    return resolve(cwd, paths[0])


def _redirects_stdout(node: Node) -> bool:
    """Return whether a compound node sends its output to a file."""
    return hasattr(node, "redirects") and any(
        r.type in {">", ">>", ">&", ">|"} and r.input in {None, 1}
        for r in node.redirects
    )


# `bashlex` hangs children on ``parts``, ``list``, or ``command``, by construct.
def _children(node: Node) -> list[Node]:
    """Return every node directly under ``node``."""
    out = [
        *(node.parts if hasattr(node, "parts") else ()),
        *(node.list if hasattr(node, "list") else ()),
        *((node.command,) if hasattr(node, "command") else ()),
    ]
    return [child for child in out if child is not node and hasattr(child, "kind")]


def _sink_blocks(sink: Invocation, *, counted: bool) -> bool:
    """Return whether ``sink`` does more to what flows into it than cut it."""
    if sink.captures_stdout:
        return True
    if _passes_through(sink.exe):
        return False
    return not (counted and sink.exe == "wc" and sink.args == ("-l",))


def _passes_through(exe: str) -> bool:
    """Return whether a stage only cuts short or pages what flows through it."""
    return exe in {"head", "tail", "cat", "less", "more"}


# Short flags cluster: ``-c5`` and ``-nc`` hold ``-c``. A flag taking a value
# ends the cluster: ``grep -evalue`` is ``-e value``, not ``-v``.
def _denied(arg: str, deny: frozenset[str], *, takes_value: frozenset[str]) -> bool:
    """Return whether ``arg`` spells a flag in ``deny``."""
    if arg in deny or arg.partition("=")[0] in deny:
        return True
    if not arg.startswith("-") or arg.startswith("--") or arg[1:].isdigit():
        return False
    valued = {flag[1] for flag in takes_value if len(flag) == 2}
    for letter in arg[1:]:
        if f"-{letter}" in deny:
            return True
        if letter in valued:
            return False
    return False


def _falls_back(trees: Sequence[Node]) -> bool:
    """Return whether ``trees`` run anything only when what came before failed."""
    pending = list(trees)
    while pending:
        node = pending.pop()
        if node.kind == "operator" and node.op == "||":
            return True
        pending.extend(_children(node))
    return False


def _cd_to(node: Node, cwd: str) -> bool:
    """Return whether ``node`` is exactly ``cd DIR``, with ``DIR`` naming ``cwd``."""
    if node.kind != "command" or any(p.kind != "word" for p in node.parts):
        return False
    words = node.parts
    if len(words) != 2 or words[0].word != "cd":
        return False
    # A word bash fills in, ``~`` included, names a place the text does not say.
    target = words[1]
    if getattr(target, "parts", ()) or target.word.startswith("~"):
        return False
    here = posixpath.normpath(cwd)
    return posixpath.normpath(PurePosixPath(here) / target.word) == here


def _literal(word: str) -> bool:
    """Return whether ``word`` names itself: no expansion, quote, or escape in it."""
    return not any(c in word for c in "$`\"'\\*?[~")


def _invocations(trees: Sequence[Node]) -> tuple[Invocation, ...]:
    """Return every simple command in ``trees``, in order, with its context."""
    out: list[Invocation] = []
    pipelines = itertools.count()
    for tree in trees:
        _walk(tree, cwd="", out=out, pipelines=pipelines)
    return tuple(out)
