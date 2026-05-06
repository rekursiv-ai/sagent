"""Bash command parsing, classification, and matcher helpers.

Single entry point over ``bashlex`` (no type stubs, third-party).
Everything works on the raw bashlex AST; ``Command`` and the helpers
below are matcher-friendly conveniences over it.

Public surface:

- :func:`parse_bash` - parse a command string into typed bashlex trees.
- :func:`unwrap_cd_prefix` - normalize ``cd X && CMD`` into ``(cwd, CMD)``.
- :func:`match_pipeline` - extract a clean two-command pipeline.
- :func:`resolve_cwd_path` - combine a ``cd`` prefix with a positional path.
- :func:`is_read_only` - classify a parse as side-effect-free (concurrency gate).
- :class:`Command`, :class:`Node` - record / protocol types.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast


if TYPE_CHECKING:
    from bashlex.ast import (
        node as Node,  # noqa: N812 -- rename to PascalCase for type convention
    )

    import bashlex
    import bashlex.errors
else:
    from sagent.lib.lazy_import import lazy_import

    bashlex = lazy_import("bashlex")  # 88ms cold
    Node = object


__all__ = [
    "BashParseCache",
    "Command",
    "Node",
    "cached_parse_bash",
    "is_read_only",
    "match_pipeline",
    "parse_bash",
    "resolve_cwd_path",
    "unwrap_cd_prefix",
    "unwrap_cd_subtree",
]


type BashParseCache = dict[str, tuple[Node, ...] | None]


# -- Records ---------------------------------------------------------


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
    args: tuple[str, ...]
    env_prefix: Mapping[str, str]
    captures_stdout: bool


# -- Parse + matcher helpers -----------------------------------------


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


def unwrap_cd_prefix(
    trees: Sequence[Node],
) -> tuple[str | None, Command] | None:
    """Normalize a ``cd X && CMD`` pattern into ``(cwd, CMD)``.

    Returns ``(None, cmd)`` for a plain simple command, or
    ``(cd_path, cmd)`` for ``cd PATH && CMD`` with a single positional
    ``cd`` arg and no other adornment (extra operators, redirects,
    env prefix). Returns ``None`` for everything else.

    Matchers call this so ``cd src && grep foo .`` and ``grep foo src``
    produce the same suggestion.

    Args:
      trees: Top-level bashlex AST nodes.

    Returns:
      result: ``(cd_path, command)`` tuple, or None if not matched.

    """
    if len(trees) != 1:
        return None
    top = trees[0]
    if top.kind == "command":
        cmd = _parse_command(top)
        if cmd.captures_stdout:
            return None
        return None, cmd
    if top.kind != "list":
        return None
    parts = top.parts
    cmd_nodes = [p for p in parts if p.kind == "command"]
    op_nodes = [p for p in parts if p.kind == "operator"]
    if len(cmd_nodes) != 2 or len(op_nodes) != 1 or op_nodes[0].op != "&&":
        return None
    cd_cmd = _parse_command(cmd_nodes[0])
    inner = _parse_command(cmd_nodes[1])
    if (
        cd_cmd.exe != "cd"
        or len(cd_cmd.args) != 1
        or cd_cmd.env_prefix
        or cd_cmd.captures_stdout
        or inner.env_prefix
        or inner.captures_stdout
    ):
        return None
    return cd_cmd.args[0], inner


def unwrap_cd_subtree(trees: Sequence[Node]) -> Sequence[Node] | None:
    """Strip a leading ``cd PATH &&`` and return the remaining subtree.

    For input ``cd X && SUBTREE`` (where ``SUBTREE`` is any single AST
    node — command, pipeline, etc.), returns ``[SUBTREE]`` so callers
    can re-dispatch through other matchers (``match_pipeline``,
    ``unwrap_cd_prefix`` won't recurse into a pipeline). Returns
    ``None`` if the prefix doesn't match the cd-clean shape.

    Args:
      trees: Top-level bashlex AST nodes.

    Returns:
      subtree: Remaining AST nodes after the ``cd`` prefix, or None.

    """
    if len(trees) != 1 or trees[0].kind != "list":
        return None
    parts = trees[0].parts
    if len(parts) != 3:
        return None
    cd_node, op_node, rest = parts[0], parts[1], parts[2]
    if cd_node.kind != "command":
        return None
    if op_node.kind != "operator" or op_node.op != "&&":
        return None
    cd_cmd = _parse_command(cd_node)
    if (
        cd_cmd.exe != "cd"
        or len(cd_cmd.args) != 1
        or cd_cmd.env_prefix
        or cd_cmd.captures_stdout
    ):
        return None
    return [rest]


def match_pipeline(trees: Sequence[Node]) -> tuple[Command, Command] | None:
    """Match a clean two-command pipeline (``A | B``).

    Returns ``(first, second)`` only if exactly two simple commands
    are piped together with no env prefix, no redirect, and no
    enclosing subshell. ``None`` otherwise.

    Args:
      trees: Top-level bashlex AST nodes.

    Returns:
      result: ``(first, second)`` command pair, or None.

    """
    if len(trees) != 1:
        return None
    top = trees[0]
    if top.kind != "pipeline":
        return None
    cmd_nodes = [p for p in top.parts if p.kind == "command"]
    if len(cmd_nodes) != 2:
        return None
    first = _parse_command(cmd_nodes[0])
    second = _parse_command(cmd_nodes[1])
    if (
        first.env_prefix
        or second.env_prefix
        or first.captures_stdout
        or second.captures_stdout
    ):
        return None
    return first, second


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


# -- Read-only command classifier --------------------------------------
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
        # sed - read-only iff no in-place flag (see _sed_mutates).
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
_EXEC_WRAPPERS: dict[str, tuple[str, ...]] = {
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


def _sed_mutates(args: list[str]) -> bool:
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

    if exe in _READ_ONLY:
        mutates = (
            (exe == "find" and any(a in _FIND_DENY_FLAGS for a in args))
            or (exe == "sed" and _sed_mutates(args))
            or (exe in _TYPE_CHECKERS and _type_checker_mutates(args))
        )
        return not mutates

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
        # Operator/pipe/reservedword children are pure glue - only
        # actual command-bearing children matter.
        return all(
            _is_node_safe(p)
            for p in node.parts
            if p.kind == "command"
            or p.kind
            in (
                "pipeline",
                "list",
                "compound",
                "commandsubstitution",
                "processsubstitution",
            )
        )
    if kind == "compound":
        # Subshells/group commands: recurse into the embedded list.
        return all(
            _is_node_safe(p)
            for p in node.list
            if p.kind
            in (
                "command",
                "pipeline",
                "list",
                "compound",
                "commandsubstitution",
                "processsubstitution",
            )
        )
    if kind in ("commandsubstitution", "processsubstitution"):
        return _is_node_safe(node.command)
    # ``tilde`` / ``parameter`` are pure word expansions (``~/path``,
    # ``$VAR``, ``${VAR:-default}``) - no execution, no writes.
    # Everything else (reservedword, operator, pipe, function, for,
    # while, if, case, select) may hide arbitrary code and is rejected.
    return kind in ("tilde", "parameter")
