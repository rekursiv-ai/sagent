"""Tests for lib.bash.

Covers the parse+matcher helpers and the read-only classifier. The
classifier's large parametrize lists are inline for easy cross-ref
with the implementation.
"""

from __future__ import annotations

import pytest

from sagent.tools.lib import bash as bash_mod
from sagent.tools.lib.bash import (
    BashParseCache,
    Node,
    cached_parse_bash,
    is_read_only,
    match_pipeline,
    parse_bash,
    resolve_cwd_path,
    unwrap_cd_prefix,
    unwrap_cd_subtree,
)


def _parse(command: str) -> tuple[Node, ...]:
    trees = parse_bash(command)
    assert trees is not None, command
    return trees


# -- parse_bash --------------------------------------------------------


class TestParseBash:
    def test_empty_returns_none(self) -> None:
        assert parse_bash("") is None

    def test_parse_error_returns_none(self) -> None:
        assert parse_bash("not valid bash )") is None

    def test_simple_command(self) -> None:
        trees = parse_bash("grep foo file")
        assert trees is not None
        assert len(trees) == 1
        assert trees[0].kind == "command"


# -- cached_parse_bash -------------------------------------------------


class TestCachedParseBash:
    def test_first_call_populates_cache(self) -> None:
        cache: BashParseCache = {}
        trees = cached_parse_bash("grep foo .", cache)
        assert trees is not None
        assert "grep foo ." in cache
        assert cache["grep foo ."] is trees

    def test_repeat_call_returns_cached(self) -> None:
        cache: BashParseCache = {}
        first = cached_parse_bash("ls -la", cache)
        second = cached_parse_bash("ls -la", cache)
        assert first is second  # identity, not just equality

    def test_unparseable_input_cached_as_none(self) -> None:
        """Repeated bad input doesn't repeatedly invoke bashlex."""
        cache: BashParseCache = {}
        first = cached_parse_bash("not valid bash )", cache)
        assert first is None
        assert "not valid bash )" in cache
        # Sentinel: re-call returns None without re-parsing.
        second = cached_parse_bash("not valid bash )", cache)
        assert second is None

    def test_cache_skips_reparse(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify the cache short-circuits parse_bash on hit."""
        call_count = 0
        real_parse = bash_mod.parse_bash

        def counting_parse(cmd: str) -> tuple[Node, ...] | None:
            nonlocal call_count
            call_count += 1
            return real_parse(cmd)

        monkeypatch.setattr(bash_mod, "parse_bash", counting_parse)
        cache: BashParseCache = {}
        _ = cached_parse_bash("grep foo .", cache)
        _ = cached_parse_bash("grep foo .", cache)
        _ = cached_parse_bash("grep foo .", cache)
        assert call_count == 1


# -- unwrap_cd_prefix --------------------------------------------------


class TestUnwrapCdPrefix:
    def test_bare_command(self) -> None:
        result = unwrap_cd_prefix(_parse("grep foo file"))
        assert result is not None
        cwd, cmd = result
        assert cwd is None
        assert cmd.exe == "grep"
        assert cmd.args == ("foo", "file")
        assert dict(cmd.env_prefix) == {}

    def test_bundled_flags(self) -> None:
        result = unwrap_cd_prefix(_parse("grep -rln pattern src/"))
        assert result is not None
        _, cmd = result
        assert cmd.args == ("-rln", "pattern", "src/")

    def test_env_prefix(self) -> None:
        result = unwrap_cd_prefix(_parse("FOO=bar BAZ=qux grep foo"))
        assert result is not None
        _, cmd = result
        assert dict(cmd.env_prefix) == {"FOO": "bar", "BAZ": "qux"}
        assert cmd.args == ("foo",)

    def test_cd_then_cmd(self) -> None:
        result = unwrap_cd_prefix(_parse("cd src && grep foo ."))
        assert result is not None
        cwd, cmd = result
        assert cwd == "src"
        assert cmd.exe == "grep"
        assert cmd.args == ("foo", ".")

    def test_redirect_bails(self) -> None:
        assert unwrap_cd_prefix(_parse("grep foo > out")) is None

    def test_pipeline_bails(self) -> None:
        assert unwrap_cd_prefix(_parse("grep foo | wc -l")) is None

    def test_cd_or_pattern_bails(self) -> None:
        assert unwrap_cd_prefix(_parse("cd src || grep foo")) is None

    def test_cd_multiple_args_bails(self) -> None:
        assert unwrap_cd_prefix(_parse("cd a b && grep foo")) is None

    def test_three_command_chain_bails(self) -> None:
        assert unwrap_cd_prefix(_parse("cd src && cd sub && grep foo")) is None

    def test_subshell_bails(self) -> None:
        assert unwrap_cd_prefix(_parse("(cd src && grep foo)")) is None


# -- unwrap_cd_subtree ------------------------------------------------


class TestUnwrapCdSubtree:
    def test_no_cd_returns_none(self) -> None:
        assert unwrap_cd_subtree(_parse("ls -la")) is None

    def test_cd_then_command(self) -> None:
        sub = unwrap_cd_subtree(_parse("cd src && ls -la"))
        assert sub is not None
        assert len(sub) == 1
        assert sub[0].kind == "command"

    def test_cd_then_pipeline(self) -> None:
        sub = unwrap_cd_subtree(_parse("cd src && ls | head -5"))
        assert sub is not None
        assert len(sub) == 1
        assert sub[0].kind == "pipeline"
        # Re-dispatching through match_pipeline works on the slice.
        pair = match_pipeline(sub)
        assert pair is not None
        first, second = pair
        assert first.exe == "ls"
        assert second.exe == "head"

    def test_non_cd_prefix_bails(self) -> None:
        assert unwrap_cd_subtree(_parse("echo hi && ls")) is None

    def test_or_operator_bails(self) -> None:
        assert unwrap_cd_subtree(_parse("cd src || ls")) is None

    def test_cd_no_arg_bails(self) -> None:
        assert unwrap_cd_subtree(_parse("cd && ls")) is None


# -- match_pipeline ----------------------------------------------------


class TestMatchPipeline:
    def test_two_commands(self) -> None:
        result = match_pipeline(_parse("find . | xargs grep foo"))
        assert result is not None
        first, second = result
        assert first.exe == "find"
        assert second.exe == "xargs"

    def test_single_command_bails(self) -> None:
        assert match_pipeline(_parse("grep foo file")) is None

    def test_three_commands_bails(self) -> None:
        assert match_pipeline(_parse("a | b | c")) is None

    def test_env_prefix_bails(self) -> None:
        assert match_pipeline(_parse("FOO=x cat f | wc")) is None


# -- resolve_cwd_path --------------------------------------------------


@pytest.mark.parametrize(
    ("cwd", "path", "expected"),
    [
        (None, None, ""),
        (None, "", ""),
        (None, ".", ""),
        (None, "src", "src"),
        ("src", None, "src"),
        ("src", "", "src"),
        ("src", ".", "src"),
        ("src", "sub", "src/sub"),
        ("src/", "sub", "src/sub"),
        ("src", "/abs/path", "/abs/path"),
    ],
)
def test_resolve_cwd_path(
    cwd: str | None,
    path: str | None,
    expected: str,
) -> None:
    assert resolve_cwd_path(cwd, path) == expected


# -- is_read_only ------------------------------------------------------


def _classify(command: str) -> bool:
    trees = parse_bash(command)
    if trees is None:
        return False
    return is_read_only(trees)


@pytest.mark.parametrize(
    "cmd",
    [
        # Basic single utilities.
        "ls -la",
        "pwd",
        "cat foo.txt",
        "head -100 foo",
        "wc -l foo",
        "echo hello",
        "printf '%s\\n' hi",
        "true",
        # Search / inspection.
        "grep -r foo .",
        "rg -n hello src/",
        "find . -name '*.py'",
        "find /tmp -type f",
        # sed without in-place is read-only.
        "sed 's/foo/bar/' file.txt",
        "sed -n '1,10p' file.txt",
        "sed -E 's/[0-9]+//' file.txt",
        "cat file.txt | sed 's/x/y/'",
        # Type-checkers (no mutating flags).
        "basedpyright",
        "basedpyright src/",
        "pyright --outputjson",
        "mypy foo.py",
        "ty check src/",
        # uv run wrapper delegates to inner command.
        "uv run basedpyright",
        "uv run mypy .",
        "uv --quiet --project . run basedpyright",
        "uv --project /tmp run pyright --outputjson",
        "uv tree",
        # git with -C value-flag handling.
        "git -C /repo status",
        "git -C /repo --no-pager log",
        # git branch/tag for listing (no positionals).
        "git branch",
        "git branch -a",
        "git branch -v",
        "git branch --list",
        "git branch --contains HEAD",
        "git branch --merged main",
        "git tag",
        "git tag -l",
        "git tag -n",
        # git reflog default (show).
        "git reflog",
        "git reflog show",
        # npm audit without fix.
        "npm audit",
        "npm audit --json",
        # uv cache/python read-only subsubcommands.
        "uv cache dir",
        "uv python list",
        "uv python find 3.12",
        # Subcommand allowlist.
        "git status",
        "git log --oneline -10",
        "git diff HEAD~1",
        "git --no-pager log",
        "gh pr list",
        "gh issue view 42",
        "docker ps",
        "docker inspect foo",
        "kubectl get pods",
        "npm list",
        "pip show numpy",
        # Pipelines.
        "cat foo | head -10",
        "ls | grep py | head",
        "find . -type f | wc -l",
        # Compound.
        "pwd && ls",
        "ls; pwd",
        "(ls; pwd)",
        # Substitutions of read-only inner commands.
        "echo $(pwd)",
        "diff <(ls a) <(ls b)",
        # Pure assignment.
        "A=1",
        "FOO=bar BAZ=qux ls",
    ],
)
def test_is_read_only_safe(cmd: str) -> None:
    assert _classify(cmd), cmd


@pytest.mark.parametrize(
    "cmd",
    [
        # Direct mutators.
        "rm foo",
        "mkdir foo",
        "mv a b",
        "cp a b",
        "touch foo",
        "chmod 755 foo",
        # Redirection (write).
        "echo hi > out.txt",
        "echo hi >> out.txt",
        "cat foo > /etc/passwd",
        # Heredoc / herestring (redirect node - reject conservatively).
        "cat <<< hello",
        # Mutating subcommands.
        "git push",
        "git commit -m foo",
        "git config user.email foo",
        "git fetch",
        "gh pr create",
        "docker rm xyz",
        "docker run alpine",
        "kubectl delete pod x",
        "npm install",
        "pip install foo",
        # find with mutating actions.
        "find . -delete",
        "find . -exec rm {} \\;",
        # sed with in-place edit (all short/long/combined forms).
        "sed -i 's/foo/bar/' file.txt",
        "sed -i.bak 's/foo/bar/' file.txt",
        "sed -i '' 's/foo/bar/' file.txt",
        "sed --in-place 's/foo/bar/' file.txt",
        "sed --in-place=.bak 's/foo/bar/' file.txt",
        "sed -ni 's/foo/bar/p' file.txt",
        "sed -Ei 's/foo/bar/' file.txt",
        # Type-checker mutating flags.
        "pyright --createstub requests",
        "basedpyright --createstubs requests",
        "basedpyright --writebaseline",
        "mypy --install-types",
        # uv run delegates: inner command's gates still apply.
        "uv run basedpyright --createstubs requests",
        "uv run sed -i 's/a/b/' foo",
        # uv run with non-readonly inner (pytest executes arbitrary code).
        "uv run pytest",
        "uv --project . run pytest tests/",
        # uv subcommand not in allowlist.
        "uv add requests",
        "uv pip install foo",
        # git branch/tag with positional → create/delete/rename.
        "git branch newbranch",
        "git branch -d oldbranch",
        "git branch -D oldbranch",
        "git branch -m old new",
        "git tag v1.0",
        "git tag -d v1.0",
        # git reflog with mutating subsubcommand.
        "git reflog expire --expire=now --all",
        "git reflog delete HEAD@{0}",
        "git reflog exists HEAD",
        # npm audit fix mutates.
        "npm audit fix",
        "npm audit fix --force",
        # uv cache/python mutations.
        "uv cache clean",
        "uv cache prune",
        "uv python install 3.13",
        "uv python uninstall 3.12",
        "uv python pin 3.13",
        # cargo/go subcommands we dropped.
        "cargo check",
        "cargo fetch",
        "go vet",
        # Dangerous nested substitution.
        "cat $(rm foo)",
        "echo `rm bar`",
        # Compound with one bad member.
        "cd /tmp && rm foo",
        "ls; rm foo",
        "(ls; rm foo)",
        # Pipeline with one bad member.
        "cat foo | tee out.txt",
        "ls | xargs rm",
        # Privilege escalation - would re-evaluate.
        "sudo ls",
        "su -c ls",
        # Unknown utilities.
        "foo bar",
        "make build",
        # Parse errors.
        "not valid bash )",
        "",
    ],
)
def test_is_read_only_unsafe(cmd: str) -> None:
    assert not _classify(cmd), cmd
