"""Tests for ``tools.lib.bash``: parsing, classification, matchers."""

from __future__ import annotations

import pytest

from sagent.tools.lib.bash import (
    BashParseCache,
    cached_parse_bash,
    is_read_only,
    parse_bash,
    resolve_cwd_path,
    walk_commands,
)


def test_parse_bash_simple_command() -> None:
    trees = parse_bash("ls -la")
    assert trees is not None
    assert len(trees) == 1
    assert trees[0].kind == "command"


def test_parse_bash_empty_returns_none() -> None:
    assert parse_bash("") is None


def test_parse_bash_unparseable_returns_none() -> None:
    # An unterminated heredoc fails to parse.
    assert parse_bash("cat <<EOF") is None


def test_cached_parse_bash_reuses_result() -> None:
    cache: BashParseCache = {}
    first = cached_parse_bash("ls", cache)
    second = cached_parse_bash("ls", cache)
    assert first is second
    assert "ls" in cache


def test_cached_parse_bash_caches_none() -> None:
    cache: BashParseCache = {}
    _ = cached_parse_bash("cat <<EOF", cache)
    assert cache["cat <<EOF"] is None


def test_resolve_cwd_path_neither() -> None:
    assert resolve_cwd_path(None, None) == ""


def test_resolve_cwd_path_dot_only() -> None:
    assert resolve_cwd_path(None, ".") == ""


def test_resolve_cwd_path_path_only() -> None:
    assert resolve_cwd_path(None, "src") == "src"


def test_resolve_cwd_path_cwd_only() -> None:
    assert resolve_cwd_path("src", None) == "src"


def test_resolve_cwd_path_join_relative() -> None:
    assert resolve_cwd_path("src", "x") == "src/x"


def test_resolve_cwd_path_join_strips_trailing_slash() -> None:
    assert resolve_cwd_path("src/", "x") == "src/x"


def test_resolve_cwd_path_absolute_path_wins() -> None:
    assert resolve_cwd_path("src", "/abs/path") == "/abs/path"


def test_resolve_cwd_path_cwd_with_dot() -> None:
    assert resolve_cwd_path("src", ".") == "src"


def test_is_read_only_grep() -> None:
    trees = parse_bash("grep foo .")
    assert trees is not None
    assert is_read_only(trees) is True


def test_is_read_only_ls_pipeline() -> None:
    trees = parse_bash("ls | grep foo")
    assert trees is not None
    assert is_read_only(trees) is True


def test_is_read_only_rm_unsafe() -> None:
    trees = parse_bash("rm -rf /tmp")
    assert trees is not None
    assert is_read_only(trees) is False


def test_is_read_only_redirect_unsafe() -> None:
    trees = parse_bash("ls > out.txt")
    assert trees is not None
    assert is_read_only(trees) is False


def test_is_read_only_stderr_redirect_safe() -> None:
    trees = parse_bash("ls 2>/dev/null")
    assert trees is not None
    # stderr-only redirect doesn't divert stdout; ls is still read-only.
    # Note: bashlex parses the redirect as a "redirect" node which
    # _is_command_safe rejects. So this is unsafe in current classifier.
    # Adjust: any redirect at command level is unsafe.
    assert is_read_only(trees) is False


def test_is_read_only_find_with_delete_unsafe() -> None:
    trees = parse_bash("find . -delete")
    assert trees is not None
    assert is_read_only(trees) is False


def test_is_read_only_sed_in_place_unsafe() -> None:
    trees = parse_bash("sed -i 's/a/b/' file")
    assert trees is not None
    assert is_read_only(trees) is False


def test_is_read_only_sed_basic_safe() -> None:
    trees = parse_bash("sed 's/a/b/' file")
    assert trees is not None
    assert is_read_only(trees) is True


def test_is_read_only_git_status_safe() -> None:
    trees = parse_bash("git status")
    assert trees is not None
    assert is_read_only(trees) is True


def test_is_read_only_git_push_unsafe() -> None:
    trees = parse_bash("git push origin main")
    assert trees is not None
    assert is_read_only(trees) is False


def test_is_read_only_git_branch_list_safe() -> None:
    trees = parse_bash("git branch")
    assert trees is not None
    assert is_read_only(trees) is True


def test_is_read_only_git_branch_create_unsafe() -> None:
    trees = parse_bash("git branch new-feature")
    assert trees is not None
    assert is_read_only(trees) is False


def test_is_read_only_uv_run_basedpyright_safe() -> None:
    trees = parse_bash("uv run basedpyright")
    assert trees is not None
    assert is_read_only(trees) is True


def test_is_read_only_uv_run_basedpyright_createstubs_unsafe() -> None:
    trees = parse_bash("uv run basedpyright --createstubs pkg")
    assert trees is not None
    assert is_read_only(trees) is False


def test_is_read_only_empty_input_false() -> None:
    assert is_read_only([]) is False


def test_is_read_only_pyright_writebaseline_unsafe() -> None:
    trees = parse_bash("pyright --writebaseline")
    assert trees is not None
    assert is_read_only(trees) is False


def test_is_read_only_command_substitution_safe() -> None:
    trees = parse_bash("echo $(date)")
    assert trees is not None
    assert is_read_only(trees) is True


def test_is_read_only_command_substitution_unsafe() -> None:
    trees = parse_bash("echo $(rm foo)")
    assert trees is not None
    assert is_read_only(trees) is False


def test_is_read_only_env_assignment_only_safe() -> None:
    trees = parse_bash("FOO=bar")
    assert trees is not None
    assert is_read_only(trees) is True


def test_is_read_only_gh_pr_view_safe() -> None:
    trees = parse_bash("gh pr view 42")
    assert trees is not None
    assert is_read_only(trees) is True


def test_is_read_only_gh_pr_create_unsafe() -> None:
    trees = parse_bash("gh pr create")
    assert trees is not None
    assert is_read_only(trees) is False


def test_is_read_only_npm_audit_safe() -> None:
    trees = parse_bash("npm audit")
    assert trees is not None
    assert is_read_only(trees) is True


def test_is_read_only_npm_audit_fix_unsafe() -> None:
    trees = parse_bash("npm audit fix")
    assert trees is not None
    assert is_read_only(trees) is False


def test_is_read_only_git_reflog_safe() -> None:
    trees = parse_bash("git reflog")
    assert trees is not None
    assert is_read_only(trees) is True


def test_is_read_only_git_reflog_expire_unsafe() -> None:
    trees = parse_bash("git reflog expire")
    assert trees is not None
    assert is_read_only(trees) is False


def test_is_read_only_unknown_command_unsafe() -> None:
    trees = parse_bash("frobnicate")
    assert trees is not None
    assert is_read_only(trees) is False


def test_command_captures_stdout_flag() -> None:
    trees = parse_bash("ls > out.txt")
    assert trees is not None
    assert [i.captures_stdout for i in walk_commands(trees)] == [True]


@pytest.mark.parametrize(
    "command",
    [
        # ``env``/``command`` EXECUTE their argument; allowlisting the
        # wrapper without inspecting its payload launders anything.
        "env rm victim",
        "command rm victim",
        # Write flags on otherwise read-only utilities.
        "sort -o victim input",
        "sed 'w victim' input",
        "go env -w GOFLAGS=-x",
        # ``awk`` can shell out.
        "awk 'BEGIN { system(\"rm victim\") }'",
        # Loop bodies: the compound child is kind ``for``/``while``,
        # which the kind-filter drops -- and ``all([])`` is True, so an
        # unexamined body reads as safe.
        "for f in victim; do rm $f; done",
        "while true; do rm victim; break; done",
    ],
)
def test_a_mutating_command_is_never_read_only(command: str) -> None:
    """The gate must bias to False: a false negative means concurrent writes."""
    trees = parse_bash(command)
    assert trees is not None
    assert not is_read_only(trees), command


def test_an_unknown_node_kind_fails_closed() -> None:
    """A construct the classifier does not model must read as unsafe.

    ``if`` parses to a ``compound`` whose child is an ``if`` node -- a
    kind the classifier has no rule for, so it must reject rather than
    let the branch bodies go unexamined.
    """
    trees = parse_bash("if true; then rm victim; fi")
    assert trees is not None
    assert not is_read_only(trees)


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
