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
    "BashMatcher",
    "BashParseCache",
    "Command",
    "Invocation",
    "Node",
    "cached_parse_bash",
    "is_read_only",
    "parse_bash",
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


def _walk_node(node: Node, *, cwd: str, out: list[Invocation]) -> None:
    """Recurse one AST node, appending every simple command found."""
    kind: str = node.kind
    if kind == "command":
        _walk_command(node, cwd=cwd, out=out)
        return
    if kind == "pipeline":
        _walk_pipeline(node, cwd=cwd, out=out)
        return
    if kind == "list":
        _walk_list(node, cwd=cwd, out=out)
        return
    # ``compound`` (loops, conditionals) and everything else: descend
    # through whatever child collections the node carries. A loop body
    # nudges even when its argument is a loop variable -- a command
    # reaching for ``cat $f`` wants the tool regardless of whether the
    # filename is knowable here.
    for child in _child_nodes(node):
        _walk_node(child, cwd=cwd, out=out)


def _walk_list(node: Node, *, cwd: str, out: list[Invocation]) -> None:
    """Walk an ``A && B ; C`` list, threading any ``cd`` through it.

    A ``cd`` binds to everything after it in the same list, so
    ``cd X && ls && cat f`` reports BOTH commands under ``X`` -- the
    two-command-only unwrap missed the chain entirely.
    """
    inner_cwd = cwd
    for part in node.parts:
        if part.kind == "operator":
            # ``cd X || CMD`` runs CMD only when the cd FAILED, so the
            # directory it names is exactly where CMD is not.
            if part.op == "||":
                inner_cwd = cwd
            continue
        if part.kind == "command":
            cmd = _parse_command(part)
            if cmd.exe == "cd" and len(cmd.args) == 1 and not cmd.captures_stdout:
                inner_cwd = cmd.args[0]
                continue
        _walk_node(part, cwd=inner_cwd, out=out)


def _walk_pipeline(node: Node, *, cwd: str, out: list[Invocation]) -> None:
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
            continue
        cmd = _parse_command(stages[i])
        if not cmd.exe:
            continue
        linked[i] = Invocation(
            exe=cmd.exe,
            args=cmd.args,
            cwd=cwd,
            piped_into=linked[i + 1] if i + 1 < len(stages) else None,
            env_prefix=cmd.env_prefix,
            captures_stdout=cmd.captures_stdout,
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
        elif linked[i] is None:
            _walk_node(stage, cwd=cwd, out=out)


def _walk_command(node: Node, *, cwd: str, out: list[Invocation]) -> None:
    """Record one un-piped simple command, then its substitutions."""
    cmd = _parse_command(node)
    if cmd.exe:
        out.append(
            Invocation(
                exe=cmd.exe,
                args=cmd.args,
                cwd=cwd,
                env_prefix=cmd.env_prefix,
                captures_stdout=cmd.captures_stdout,
            )
        )
    _walk_substitutions(node, cwd=cwd, out=out)


def _walk_substitutions(node: Node, *, cwd: str, out: list[Invocation]) -> None:
    """Descend into ``$(...)`` and friends nested in a command's words."""
    for child in _child_nodes(node):
        if child.kind in ("word", "commandsubstitution", "command", "list", "pipeline"):
            _walk_node(child, cwd=cwd, out=out)


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
    except Exception:
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


def resolve_cwd_path(cwd: str | None, path: str | None) -> str:
    """Combine a ``cd`` prefix with a tool's positional path arg.

    Returns ``""`` when the effective path is just ``"."`` so callers
    can omit ``path=`` from the suggestion entirely.

    Args:
      cwd: Directory from a ``cd`` prefix, or None.
      path: Positional path argument, or None.

    Returns:
      resolved: Combined path string, or empty string.

    """
    if cwd is None:
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
        # Text search variants.
        "egrep",
        "fgrep",
        "zgrep",
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
        # Static analyzers - read source, emit diagnostics (flag-gated
        # against stub/baseline/install commands; see _type_checker_mutates).
        "pyright",
        "basedpyright",
        "mypy",
        "ty",
    }
)

_READ_ONLY: frozenset[str] = _READ_ONLY_BASE | _READ_ONLY_EXTRA

# `find` and `xargs` are read-only IFF none of these flags appear.
# `xargs` runs an arbitrary command, so we must inspect its tail -
# but that requires re-classifying the rest as a command, which the
# AST doesn't model. Easiest: deny any xargs.
_FIND_DENY_FLAGS: frozenset[str] = frozenset(
    {
        "-delete",
        "-exec",
        "-execdir",
        "-ok",
        "-okdir",
        "-fprint",
        "-fprintf",
        "-fls",
    }
)

_TYPE_CHECKERS: frozenset[str] = frozenset(
    {"pyright", "basedpyright", "mypy", "ty"},
)

# Utilities whose argv IS another command. They are read-only only if
# what they run is.
_ARGV_EXECUTORS: frozenset[str] = frozenset({"env", "command", "nice", "stdbuf"})

_ENV_ASSIGNMENT: Final = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

# ``sed`` writes without ``-i`` via the ``w`` command and the ``w``
# flag on ``s///``.
_SED_WRITE_SCRIPT: Final = re.compile(r"(^|;|\})\s*\d*\s*w\s|s/.*/.*/[a-z]*w")

# ``awk`` can shell out or redirect from inside its program text.
_AWK_ESCAPE: Final = re.compile(r"\b(system|print\s*>|printf\s*>)|\|\s*\"")

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


def _mutating_flags(exe: str, args: list[str]) -> bool:
    """Whether an otherwise read-only utility was asked to write.

    Each entry is a utility that reads by default but has a documented
    write mode; the allow-list keys on the executable alone, so without
    these gates ``sort -o victim`` and ``sed 'w victim'`` both read safe.
    """
    if exe == "find":
        return any(a in _FIND_DENY_FLAGS for a in args)
    if exe == "sed":
        return sed_mutates(args) or any(_SED_WRITE_SCRIPT.search(a) for a in args)
    if exe in _TYPE_CHECKERS:
        return _type_checker_mutates(args)
    if exe == "sort":
        return any(a == "-o" or a.startswith("--output") for a in args)
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

    - ``--createstub`` / ``--createstubs`` - pyright/basedpyright stub gen
    - ``--writebaseline`` - basedpyright baseline file
    - ``--install-types`` - mypy stub installation via pip
    """
    for a in args:
        if a.startswith("--createstub"):
            return True
        if a in {"--writebaseline", "--install-types"}:
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
