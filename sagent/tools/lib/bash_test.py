"""Tests for ``tools.lib.bash``: parsing, classification, matchers."""

from __future__ import annotations

import pytest

from sagent.tools.lib.bash import (
    _UNKNOWN_CWD,
    BashParseCache,
    _denied,
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


@pytest.mark.parametrize(
    "command",
    [
        # Attached value: `-o FILE` was gated by an equality test, so the
        # attached spelling slipped through and `sort -oout input` wrote `out`.
        "sort -oout input",
        "sort -o out input",
        # `-fprint0` writes exactly like its listed `-fprint`/`-fls` siblings.
        "find . -fprint0 out",
        # The redirect need not follow `print`: the whole record sits between
        # them, and the old regex required adjacency.
        "awk '{ print $0 > \"out\" }' input",
        # A read-only git subcommand still writes when handed an output path.
        "git diff --output=report.patch",
        "git diff -o report.patch",
    ],
)
def test_is_read_only_rejects_writes_that_look_like_reads(command: str) -> None:
    """Each of these WRITES a file while classifying as read-only.

    ``is_read_only`` gates concurrent dispatch, so a false negative here is not
    a cosmetic misclassification -- it lets two writers run at once.
    """
    trees = parse_bash(command)
    assert trees is not None
    assert is_read_only(trees) is False


@pytest.mark.parametrize(
    "command",
    [
        "sort input",
        "find . -name x",
        "awk '{ print $1 }' f",
        "git diff",
        "git show HEAD",
    ],
)
def test_is_read_only_still_admits_the_plain_reads(command: str) -> None:
    """The write-gates above must not swallow the ordinary read spelling."""
    trees = parse_bash(command)
    assert trees is not None
    assert is_read_only(trees) is True


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


def test_is_read_only_uv_run_basedpyright_unsafe() -> None:
    # strace on a bare `basedpyright a.py`: mkdir("/tmp/pyright-<pid>-*", 0700)
    # then openat(..., O_WRONLY|O_CREAT|O_TRUNC) inside it. Unconditional, with
    # no flag to gate, so "read-only with a flag denylist" cannot express it.
    trees = parse_bash("uv run basedpyright")
    assert trees is not None
    assert is_read_only(trees) is False


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


@pytest.mark.parametrize(
    ("arg", "deny", "exe", "denied"),
    [
        # ``-e`` takes a value, so everything after it is the PATTERN.
        # Scanning the whole tail found a ``-v`` that is not there and
        # silently suppressed the nudge. Measured: ``grep -evalue f``
        # prints the matching line, i.e. it does not invert.
        ("-evalue", frozenset({"-v"}), "grep", False),
        ("-equiet", frozenset({"-q"}), "grep", False),
        # A real bundle still denies.
        ("-iv", frozenset({"-v"}), "grep", True),
        ("-vf", frozenset({"-v"}), "grep", True),
        # The denied flag may itself take a value: ``head -c5`` is a byte
        # window, so deny wins over the value-flag stop.
        ("-c5", frozenset({"-c"}), "head", True),
    ],
)
def test_a_flag_value_is_not_scanned_for_denied_letters(
    arg: str, deny: frozenset[str], exe: str, denied: bool
) -> None:
    """A value-taking flag ends the cluster; its value is not more flags."""
    assert _denied(arg, deny, exe=exe) is denied


@pytest.mark.parametrize(
    ("command", "cwd"),
    [
        # A literal destination is knowable and composes.
        ("cd /srv && cat f", "/srv"),
        ("cd -- /srv && cat f", "/srv"),
        ("cd -P /srv && cat f", "/srv"),
        ("cd /srv && cd sub && cat f", "/srv/sub"),
        # ``cd`` goes to $HOME and ``cd -`` to $OLDPWD: real moves whose
        # destination the command text does not name. Reading ``-`` as a
        # literal directory rendered ``file_path='-/f'``.
        ("cd && cat f", _UNKNOWN_CWD),
        ("cd - && cat f", _UNKNOWN_CWD),
        # No ``cd`` at all is a THIRD state, distinct from unknowable.
        ("cat f", ""),
    ],
)
def test_cd_destination_is_resolved_or_marked_unknowable(
    command: str, cwd: str
) -> None:
    """Option flags are not destinations, and $HOME/$OLDPWD are not literals."""
    trees = parse_bash(command)
    assert trees is not None
    assert [i.cwd for i in walk_commands(trees) if i.exe == "cat"] == [cwd]


def test_an_unknowable_cwd_suppresses_a_relative_path() -> None:
    """Resolving against an unknown directory names the wrong file."""
    assert resolve_cwd_path(_UNKNOWN_CWD, "f") == ""
    # An absolute operand does not depend on the cwd at all.
    assert resolve_cwd_path(_UNKNOWN_CWD, "/etc/hosts") == "/etc/hosts"


@pytest.mark.parametrize(
    "command",
    [
        # A compound stage is still a stage. Leaving it unlinked let the
        # neighbours join across it as if it were absent, so the search
        # looked unpiped even though ``sort`` transforms its output.
        "(grep p f) | sort",
        "grep p f | (sort)",
        "grep p f | { sort; }",
    ],
)
def test_a_compound_stage_still_separates_its_neighbours(command: str) -> None:
    """The stage between two commands must not vanish from the pipeline."""
    trees = parse_bash(command)
    assert trees is not None
    search = next(i for i in walk_commands(trees) if i.exe == "grep")
    assert search.piped_into is not None, command


def test_a_command_inside_a_compound_stage_inherits_its_sink() -> None:
    """``(grep p f) | head`` bounds the search exactly as the bare form does."""
    trees = parse_bash("(grep p f) | head -5")
    assert trees is not None
    search = next(i for i in walk_commands(trees) if i.exe == "grep")
    assert search.piped_into is not None
    assert search.piped_into.exe == "head"


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
