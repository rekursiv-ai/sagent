"""Tests for reading a Bash command before it runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from treekle import PlainTree

import bashlex
import pytest

from sagent.session.types import UncategorizedToolResult
from sagent.tools.command_analyzer import (
    Invocation,
    ReadOnly,
    Rewrite,
    _command,
    bounding_sink,
    drop_cd,
    invocations,
    line_count,
    moves,
    operands,
    replaceable,
    resolve,
    rewrite,
    sed_mutates,
)
from sagent.tools.display import ToolDisplay


if TYPE_CHECKING:
    from sagent.tools.types import ToolContext


@dataclass(frozen=True, slots=True, kw_only=True)
class _Peer:
    """A tool that takes on ``exe FILE`` as ``{"file": FILE}``, or refuses many files."""

    name: str = "Cat"
    exe: str = "cat"
    description: str = "Show a file."
    schema: Mapping[str, PlainTree] = field(default_factory=dict[str, "PlainTree"])
    display: ToolDisplay = field(default_factory=ToolDisplay)

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        del arguments
        return self.name

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        del context, arguments
        return None

    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> UncategorizedToolResult:
        del context, arguments
        return UncategorizedToolResult(call_id="")

    def from_shell(
        self,
        invocation: Invocation,
    ) -> Mapping[str, PlainTree] | str | None:
        if invocation.exe != self.exe:
            return None
        if len(invocation.args) != 1:
            return f"Show one file per {self.name} call."
        path = resolve(invocation.cwd, invocation.args[0])
        return {"file": path} if path is not None else f"{self.name} needs a path."


def _found(command: str) -> tuple[Invocation, ...]:
    found = invocations(command)
    assert found is not None, command
    return found


def _one(command: str, exe: str) -> Invocation:
    (invocation,) = [i for i in _found(command) if i.exe == exe]
    return invocation


def test_a_bare_command_is_one_invocation() -> None:
    assert _found("cat -n a.py") == (
        Invocation(exe="cat", args=("-n", "a.py"), cwd="", pipeline=0),
    )


@pytest.mark.parametrize("command", ["", "echo (", "if"])
def test_a_command_bashlex_cannot_read_has_no_invocations(command: str) -> None:
    assert invocations(command) is None


def test_a_bare_assignment_runs_no_command() -> None:
    assert _found("FOO=bar") == ()


def test_lists_yield_every_command_in_order() -> None:
    assert [i.exe for i in _found("cat a; head -3 b && ls || wc c")] == [
        "cat",
        "head",
        "ls",
        "wc",
    ]


@pytest.mark.parametrize(
    ("command", "cwd"),
    [
        ("cat f", ""),
        ("cd /srv && cat f", "/srv"),
        ("cd -- /srv && cat f", "/srv"),
        ("cd -P /srv && cat f", "/srv"),
        ("cd /srv && cd sub && cat f", "/srv/sub"),
        ("cd sub; cat f", "sub"),
        ("cd /srv && cd /etc && cat f", "/etc"),
        ("cd && cat f", None),
        ("cd - && cat f", None),
        ("cd /srv || cat f", ""),
        ("cd /srv & cat f", ""),
        ("cd /srv > log && cat f", ""),
        ("for f in *; do cd /srv && cat $f; done", "/srv"),
        ("true || cat f", ""),
    ],
)
def test_a_cd_moves_what_follows_it(command: str, cwd: str | None) -> None:
    assert _one(command, "cat").cwd == cwd


def test_a_cd_is_folded_into_the_cwd_not_listed() -> None:
    assert [i.exe for i in _found("cd /srv && cat f")] == ["cat"]


def test_a_cd_holds_for_the_rest_of_the_list() -> None:
    found = _found("cd /a; cat f; cd /b; head g")
    assert [(i.exe, i.cwd) for i in found] == [("cat", "/a"), ("head", "/b")]


def test_a_failed_cd_leaves_the_directory_it_started_from() -> None:
    found = _found("cd /a && cd /b || cat f")
    assert [(i.exe, i.cwd) for i in found] == [("cat", "/a")]


def test_pipeline_stages_link_to_their_neighbours() -> None:
    cat, grep, head = _found("cat f | grep p | head -5")
    assert cat.piped_from is None
    assert cat.piped_into is not None
    assert cat.piped_into.exe == "grep"
    assert grep.piped_from is not None
    assert grep.piped_from.exe == "cat"
    assert [s.exe for s in cat.downstream()] == ["grep", "head"]
    assert head.piped_into is None
    assert cat.pipeline == grep.pipeline == head.pipeline


def test_a_stage_reaches_every_stage_upstream() -> None:
    """``find | xargs grep`` asks what feeds ``find``; a cut chain says nothing."""
    *_, head = _found("cat f | grep p | head -5")
    assert head.piped_from is not None
    assert head.piped_from.piped_from is not None
    assert head.piped_from.piped_from.exe == "cat"
    assert head.piped_from.piped_from.piped_from is None


def test_each_pipeline_on_a_line_is_numbered_and_linked_apart() -> None:
    grep, wc_c, ls, wc_l = _found("grep p f | wc -c; ls | wc -l")
    assert grep.pipeline != ls.pipeline
    assert grep.piped_into is not None
    assert grep.piped_into.args == ("-c",)
    assert ls.piped_into is not None
    assert ls.piped_into.args == ("-l",)
    assert (wc_c.exe, wc_l.exe) == ("wc", "wc")
    assert ls.piped_from is None


def test_separate_commands_have_no_neighbours() -> None:
    for invocation in _found("cat -n a.py; grep -n p b.py"):
        assert invocation.piped_into is None
        assert invocation.piped_from is None


def test_loop_and_conditional_bodies_are_descended() -> None:
    assert [i.exe for i in _found("for f in *.py; do cat $f; done")] == ["cat"]
    assert [i.exe for i in _found("if true; then rm x; fi")] == ["true", "rm"]


def test_command_substitutions_are_descended() -> None:
    assert {i.exe for i in _found("cat $(ls *.py)")} == {"cat", "ls"}
    assert _one("cd /srv && echo $(cat f)", "cat").cwd == "/srv"


@pytest.mark.parametrize(
    ("command", "expands"),
    [
        ('sed -i "s/a/$HOME/" f', True),
        ("cat ${x}", True),
        ("cat $(ls)", True),
        ("cat `ls`", True),
        ('grep "a$" f', True),
        ("grep 'x$' f", False),
        ("ls ~/x", False),
        ("cat f", False),
    ],
)
def test_an_invocation_says_whether_bash_fills_in_a_word(
    command: str,
    expands: bool,
) -> None:
    """A word bash fills in is no literal a tool call could repeat."""
    assert _found(command)[0].expands is expands


def test_a_substitution_in_an_assignment_is_descended() -> None:
    """``X=$(rm f); cat a`` runs ``rm``: a walk that skipped it would hand the
    line to Read, and the removal would silently never happen.
    """
    assert [i.exe for i in _found("X=$(rm f); cat a")] == ["rm", "cat"]
    assert rewrite("X=$(rm f); cat a", peers=(_Peer(),)) is None


def test_an_env_prefix_is_kept_apart_from_the_words() -> None:
    grep = _one("LC_ALL=C grep p f", "grep")
    assert dict(grep.env_prefix) == {"LC_ALL": "C"}
    assert grep.args == ("p", "f")


def test_an_assignment_after_the_name_sets_no_environment() -> None:
    """Bashlex spells ``cat X=1`` with a word; an assignment node there is no prefix."""
    command = bashlex.ast.node(
        kind="command",
        parts=[
            bashlex.ast.node(kind="word", word="cat"),
            bashlex.ast.node(kind="assignment", word="X=1"),
            bashlex.ast.node(kind="word", word="f"),
        ],
    )
    invocation = _command(command, cwd="", pipeline=0)
    assert (invocation.exe, invocation.args) == ("cat", ("f",))
    assert dict(invocation.env_prefix) == {}


def test_a_pipeline_stage_that_runs_nothing_feeds_nothing() -> None:
    (cat,) = _found("X=1 | cat f")
    assert cat.exe == "cat"
    assert cat.piped_from is None
    assert dict(cat.env_prefix) == {}


@pytest.mark.parametrize(
    ("command", "captures"),
    [
        ("ls > out", True),
        ("ls >> out", True),
        ("ls >| out", True),
        ("ls 1> out", True),
        ("ls 2> err", False),
        ("ls 2>&1", False),
        ("ls < in", False),
        ("(ls) > out", True),
    ],
)
def test_only_a_redirect_of_stdout_captures_it(command: str, captures: bool) -> None:
    assert _one(command, "ls").captures_stdout is captures


def test_only_the_last_stage_reaches_an_enclosing_redirect() -> None:
    found = _found("(cat f | grep p) > out")
    cat, grep = (next(i for i in found if i.exe == exe) for exe in ("cat", "grep"))
    assert not cat.captures_stdout
    assert grep.captures_stdout


def test_every_command_of_a_redirected_group_is_captured() -> None:
    found = _found("(cat f; ls) > out")
    assert [(i.exe, i.captures_stdout) for i in found] == [("cat", True), ("ls", True)]


def test_a_redirect_downstream_is_seen_from_the_source() -> None:
    grep = _one("grep p f | head -20 > out", "grep")
    assert not grep.captures_stdout
    assert any(stage.captures_stdout for stage in grep.downstream())


@pytest.mark.parametrize(
    "command",
    ["(grep p f) | sort", "grep p f | (sort)", "grep p f | { sort; }"],
)
def test_a_compound_stage_still_separates_its_neighbours(command: str) -> None:
    grep = _one(command, "grep")
    assert grep.piped_into is not None
    assert grep.piped_into.exe in {"sort", "\0compound"}


def test_a_command_inside_a_compound_stage_feeds_what_the_stage_feeds() -> None:
    grep = _one("(grep p f) | head -5", "grep")
    assert grep.piped_into is not None
    assert grep.piped_into.exe == "head"


@pytest.mark.parametrize(
    "command",
    [
        "cat foo.py",
        "/usr/bin/cat foo.py",
        "grep -n pat foo.py",
        "ls -la",
        "ls ~",
        "echo $HOME",
        "cat f | head -5",
        "sort input",
        "find . -name x",
        "awk '{ print $1 }' f",
        "sed 's/a/b/' file",
        "sed -n '1,2p' f",
        "git status --short",
        "git diff",
        "git show HEAD",
        "git -C dir log",
        "git --no-pager -C dir log",
        "! cat f",
        "diff <(ls) <(ls a)",
        "env foo=1 cat f",
        "git --git-dir .git log",
        "git branch",
        "git branch --contains abc",
        "git branch --contains abc -a",
        "git branch -a --contains abc",
        "git reflog show expire",
        "git branch -a",
        "git tag --sort v",
        "git reflog",
        "gh pr view 42",
        "npm audit",
        "uv cache dir",
        "uv run ty check",
        "uv --project p run ty check",
        "echo $(date)",
        "FOO=bar",
        "env",
        "env FOO=1 cat f",
        "nice cat a.txt",
        "X=$(date) cat a",
        "echo ${x:-default}",
        'grep "a$" f',
        "stdbuf -oL cat f",
        "command -v ls",
        "cat a; ls && wc -l b",
        "uniq in",
    ],
)
def test_a_command_that_only_reads_may_run_beside_others(command: str) -> None:
    assert ReadOnly.default().reads(command)


@pytest.mark.parametrize(
    "command",
    [
        "rm victim",
        "frobnicate",
        "echo hi > victim",
        "cat f 2>/dev/null",
        "sed -i 's/a/b/' foo.py",
        "sed -ni '1p' f",
        "git checkout .",
        "git push origin main",
        "for f in victim; do rm $f; done",
        "while true; do rm x; break; done",
        "if true; then rm x; fi",
        "f() { rm x; }",
        "env rm victim",
        "env FOO=1 rm victim",
        "command rm victim",
        "nice rm a.txt",
        "sort -o out input",
        "sort -oout input",
        "sort --output=out input",
        "find . -delete",
        "find . -exec rm {} ;",
        "find . -fprint0 out",
        "tree -o out .",
        "uniq a out",
        "xxd a out",
        "awk '{ print $0 > \"out\" }' input",
        "awk 'BEGIN { system(\"rm x\") }'",
        "awk 'BEGIN { cmd = \"ls\"; cmd | getline }'",
        "awk -f prog.awk f",
        "awk --file=prog.awk f",
        "sed 'w victim' input",
        "sed 's/x/rm a/e' f",
        "sed 's/x/y/w out' f",
        "sed '1p\nw out' f",
        "sed -f script.sed f",
        "git diff --output=report.patch",
        "git diff -o report.patch",
        "git -c core.pager=sh log",
        "git grep -O sh x",
        "git branch new-feature",
        "git branch -a new",
        "git branch --contains abc new",
        "env sort -o out in",
        "uv run sort -o out in",
        "sed --expression-file=s f",
        "awk --source-file=p f",
        "git tag v1",
        "git reflog expire",
        "gh pr create",
        "npm audit fix",
        "go env -w GOFLAGS=-x",
        "uv run basedpyright",
        "uv run",
        "uv cache clean",
        "echo $(rm foo)",
        "cat <(rm x)",
        "echo ${x:-$(rm f)}",
        "echo ${x:-`rm f`}",
        "X=$(rm f)",
        "X=$(rm f) cat a",
        "cat foo; rm foo",
        "cat foo | tee out",
        "cat <<'EOF'\nhi\nEOF",
        "echo (",
        "",
    ],
)
def test_a_command_that_may_write_runs_alone(command: str) -> None:
    assert not ReadOnly.default().reads(command)


def test_the_policy_reads_from_its_asset() -> None:
    policy = ReadOnly.default()
    assert "cat" in policy.utilities
    assert "env" in policy.executors
    assert policy.wrappers["uv"] == "run"
    assert "rm" not in policy.utilities


def test_rewrite_hands_a_peers_command_to_the_peer() -> None:
    peer = _Peer()
    assert rewrite("cat a.py", peers=(peer,)) == Rewrite(
        tool=peer,
        arguments={"file": "a.py"},
    )


def test_rewrite_counts_a_stage_of_a_claimed_pipeline_as_claimed() -> None:
    peer = _Peer()
    assert rewrite("cat a.py | head -3", peers=(peer,)) == Rewrite(
        tool=peer,
        arguments={"file": "a.py"},
    )


def test_the_first_peer_to_claim_a_command_takes_it() -> None:
    first, second = _Peer(name="First"), _Peer(name="Second")
    result = rewrite("cat a.py", peers=(first, second))
    assert isinstance(result, Rewrite)
    assert result.tool is first


@pytest.mark.parametrize(
    "command",
    ["ls", "cat a.py && make", "make | cat a.py; ls", "echo ("],
)
def test_rewrite_runs_a_line_that_does_shell_work(command: str) -> None:
    assert rewrite(command, peers=(_Peer(),)) is None


def test_rewrite_runs_anything_without_peers() -> None:
    assert rewrite("cat a.py", peers=()) is None


def test_rewrite_refuses_a_cd_and_names_the_call_to_make() -> None:
    assert rewrite("cd /srv && cat a.py", peers=(_Peer(),)) == (
        "This command is the work of other tools; call them instead, in one"
        ' block:\n- Cat {"file": "/srv/a.py"}'
    )


def test_rewrite_refuses_many_calls_and_lists_each_once() -> None:
    cat, head = _Peer(), _Peer(name="Head", exe="head")
    assert rewrite("cat a; head b; cat a", peers=(cat, head)) == (
        "This command is the work of other tools; call them instead, in one"
        ' block:\n- Cat {"file": "a"}\n- Head {"file": "b"}'
    )


def test_a_line_that_only_reads_becomes_each_call_once() -> None:
    cat, head = _Peer(), _Peer(name="Head", exe="head")
    assert rewrite("cat a; head b && cat a", peers=(cat, head), reads=True) == (
        Rewrite(tool=cat, arguments={"file": "a"}),
        Rewrite(tool=head, arguments={"file": "b"}),
    )


@pytest.mark.parametrize(
    "command",
    [
        "cat a || head b",
        "cd /srv && cat a; head b",
        "cat a b; head c",
    ],
    ids=["fallback", "cd", "inexact"],
)
def test_a_line_that_only_reads_is_refused_when_its_calls_would_differ(
    command: str,
) -> None:
    cat, head = _Peer(), _Peer(name="Head", exe="head")
    refused = rewrite(command, peers=(cat, head), reads=True)
    assert isinstance(refused, str)
    assert refused.startswith("This command is the work of other tools")


def test_rewrite_passes_on_a_peers_refusal() -> None:
    assert rewrite("cat a b", peers=(_Peer(),)) == (
        "This command is the work of other tools; call them instead, in one"
        " block:\n- Show one file per Cat call."
    )


@pytest.mark.parametrize(
    ("args", "takes_value", "expected"),
    [
        (("-n", "5", "f"), frozenset({"-n"}), ("f",)),
        (("-n", "5", "f"), frozenset[str](), ("5", "f")),
        (("-", "f"), frozenset[str](), ("-", "f")),
        (("-a", "--", "-b", "c"), frozenset[str](), ("-b", "c")),
        (("f", "-n"), frozenset({"-n"}), ("f",)),
    ],
)
def test_operands_skip_flags_and_their_values(
    args: tuple[str, ...],
    takes_value: frozenset[str],
    expected: tuple[str, ...],
) -> None:
    assert operands(args, takes_value=takes_value) == expected


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("cat f", True),
        ("cat", False),
        ("ls", False),
        ("LC_ALL=C cat f", False),
        ("cat f > out", False),
        ("cat f | head -3", True),
        ("cat f | cat | less | more | tail", True),
        ("cat f | sort", False),
        ("cat f | head -3 > out", False),
        ("cat f | wc -l", False),
        ("cat -A f", False),
        ("cat -nA f", False),
        ("cat --show-all=x f", False),
        ("echo x | cat f", False),
        ("rm f", False),
    ],
)
def test_replaceable_admits_one_plain_command_cut_short_at_most(
    command: str,
    expected: bool,
) -> None:
    assert (
        replaceable(
            next(i for i in _found(command) if i.exe != "echo"),
            exes=frozenset({"cat", "ls"}),
            deny=frozenset({"-A", "--show-all"}),
        )
        is expected
    )


def test_replaceable_options_widen_what_it_admits() -> None:
    ls = _found("ls")[0]
    assert replaceable(ls, exes=frozenset({"ls"}), bare=True)
    fed = _one("echo x | grep p f", "grep")
    assert replaceable(fed, exes=frozenset({"grep"}), fed=True)
    counted = _found("grep p f | wc -l")[0]
    assert replaceable(counted, exes=frozenset({"grep"}), counted=True)
    words = _found("grep p f | wc -w")[0]
    assert not replaceable(words, exes=frozenset({"grep"}), counted=True)


@pytest.mark.parametrize(
    ("arg", "denied"),
    [
        ("-evalue", False),
        ("-ie", False),
        ("-iv", True),
        ("-vf", True),
        ("-c5", True),
        ("-5", False),
        ("--invert", False),
        ("-in", False),
    ],
)
def test_a_flags_value_is_not_read_as_more_flags(arg: str, denied: bool) -> None:
    grep = _found(f"grep {arg} p f")[0]
    admitted = replaceable(
        grep,
        exes=frozenset({"grep"}),
        deny=frozenset({"-v", "-c"}),
        takes_value=frozenset({"-e", "--regexp"}),
    )
    assert admitted is not denied


def test_a_count_flag_is_no_cluster_of_letters() -> None:
    """``head -15`` keeps 15 lines; it does not hold a ``-1`` flag."""
    head = _found("head -15 f")[0]
    assert replaceable(head, exes=frozenset({"head"}), deny=frozenset({"-1"}))


@pytest.mark.parametrize(
    ("command", "sink"),
    [
        ("ls | head -5", ("head", ("-5",))),
        ("ls | cat | less | tail -3", ("tail", ("-3",))),
        ("ls | more | head", ("head", ())),
        ("ls | sort | head -5", None),
        ("ls", None),
    ],
)
def test_bounding_sink_finds_the_cut_past_pass_through_stages(
    command: str,
    sink: tuple[str, tuple[str, ...]] | None,
) -> None:
    found = bounding_sink(_found(command)[0])
    assert (None if found is None else (found.exe, found.args)) == sink


@pytest.mark.parametrize(
    ("args", "count"),
    [
        ((), 10),
        (("-n", "5"), 5),
        (("-n5",), 5),
        (("-5",), 5),
        (("-n", "3", "-7"), 7),
        (("-5", "-n", "3"), 3),
        (("-n5", "-3"), 3),
        (("-3", "-n", "5"), 5),
        (("-n", "+5"), None),
        (("-n",), None),
        (("-n", "0"), None),
        (("-c", "5"), None),
        (("f",), None),
    ],
)
def test_line_count_reads_the_spellings_of_a_count(
    args: tuple[str, ...],
    count: int | None,
) -> None:
    assert line_count(args) == count


@pytest.mark.parametrize(
    ("cwd", "path", "expected"),
    [
        ("", "", ""),
        ("", "f", "f"),
        ("/srv", "f", "/srv/f"),
        ("/srv/", "f", "/srv/f"),
        ("sub", "f", "sub/f"),
        ("/srv", "", "/srv"),
        ("", ".", ""),
        ("/srv", ".", "/srv"),
        ("/srv", "/etc/hosts", "/etc/hosts"),
        (None, "/etc/hosts", "/etc/hosts"),
        ("/srv", "~/f", "~/f"),
        ("/srv", "~", "~"),
        (None, "~user/f", "~user/f"),
        (None, "f", None),
        (None, "", None),
    ],
)
def test_resolve_names_a_path_as_the_shell_finds_it(
    cwd: str | None,
    path: str,
    expected: str | None,
) -> None:
    assert resolve(cwd, path) == expected


@pytest.mark.parametrize(
    ("command", "kept"),
    [
        ("cd /srv/app && git status", "git status"),
        ("cd /srv/app/ && git status", "git status"),
        ("cd /srv/app; ls", "ls"),
        ("cd /srv/app\nls -la", "ls -la"),
        ("cd . && ls", "ls"),
        ("cd ~/app && ls", "cd ~/app && ls"),
        ("cd /srv/app && a && b", "a && b"),
        ("cd /srv/app && ls | head", "ls | head"),
        ("cd /srv && ls", "cd /srv && ls"),
        ("cd /srv/app || exit 1", "cd /srv/app || exit 1"),
        ("cd /srv/app", "cd /srv/app"),
        ('cd "$HOME/app" && ls', 'cd "$HOME/app" && ls'),
        ("cd -P /srv/app && ls", "cd -P /srv/app && ls"),
        ("cd /srv/app > log && ls", "cd /srv/app > log && ls"),
        ("ls && cd /srv/app", "ls && cd /srv/app"),
        ("cd /srv/app && (", "cd /srv/app && ("),
    ],
)
def test_drop_cd_drops_only_a_leading_cd_to_where_the_shell_starts(
    command: str,
    kept: str,
) -> None:
    assert drop_cd(command, cwd="/srv/app") == kept


@pytest.mark.parametrize(
    ("command", "moved"),
    [
        ("ls && git status", False),
        ("echo cd", False),
        ("cd sub && ls", True),
        ("ls; (cd sub)", True),
        ("pushd sub", True),
        ("popd", True),
        ("if true; then cd /; fi", True),
        ("(", True),
        ("source env.sh", True),
        (". env.sh", True),
        ("builtin cd /x", True),
        ("command cd /x", True),
        ("eval 'cd /x'", True),
        ("X=1 cd /x", True),
        ('"$go" /x', True),
        ("command -v git", False),
        ("builtin echo hi", False),
        ("Xvfb :1", False),
    ],
)
def test_moves_says_whether_a_command_may_change_directory(
    command: str,
    moved: bool,
) -> None:
    assert moves(command) is moved


@pytest.mark.parametrize(
    ("args", "mutates"),
    [
        (("-i", "s/a/b/", "f"), True),
        (("-i.bak", "s/a/b/", "f"), True),
        (("-ni", "1p", "f"), True),
        (("--in-place", "s/a/b/", "f"), True),
        (("--in-place=.bak", "s/a/b/", "f"), True),
        (("-n", "1p", "f"), False),
        (("--expression=s/i/j/", "f"), False),
        (("s/i/j/", "f"), False),
    ],
)
def test_sed_mutates_only_in_place(args: tuple[str, ...], mutates: bool) -> None:
    assert sed_mutates(args) is mutates


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
