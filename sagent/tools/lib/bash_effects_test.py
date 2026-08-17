"""Differential test: ``is_read_only`` versus what commands actually do.

Every other test of the classifier asserts what a reader believed the
command would do. This one asks the filesystem: run each command in a
sandbox, snapshot the tree before and after, and compare reality to the
verdict. A reading can be wrong; a changed inode cannot.

The asymmetry the classifier is built around is asserted directly. A
false negative -- blessing a writer -- lets two Bash calls run
concurrently and one silently clobbers the other's edit, so it fails the
test. A false positive costs only serial dispatch, so it is reported as
information rather than failure.

Marked ``slow``: 47 real subprocesses, ~0.3s total.
"""

from __future__ import annotations

from pathlib import Path

import hashlib
import shutil
import subprocess
import tempfile

import pytest

from sagent.tools.lib.bash import is_read_only, parse_bash


# Commands that must NOT touch the sandbox. A miss here costs
# parallelism only, so these are the soft half of the matrix.
_READS = [
    "cat a.txt",
    "grep x a.txt",
    "ls",
    "head -2 a.txt",
    "wc -l a.txt",
    "sort a.txt",
    "sed -n '1,2p' a.txt",
    "find . -name '*.txt'",
    "echo hi",
    "cut -c1 a.txt",
    "awk '{print $1}' a.txt",
    "diff a.txt b.txt",
    "env",
    "command -v ls",
    "nice cat a.txt",
]

# Commands that DO touch the sandbox. Every one of these must classify
# as a writer: blessing one is the failure this whole gate exists to
# prevent.
_WRITES = [
    "rm a.txt",
    "touch new.txt",
    "echo hi > a.txt",
    "echo hi >> a.txt",
    "cp a.txt c.txt",
    "mv a.txt d.txt",
    "sed -i s/x/y/ a.txt",
    "sort -o out.txt a.txt",
    "sed 'w out.txt' a.txt",
    "awk '{print > \"out.txt\"}' a.txt",
    "tee out.txt < a.txt",
    # Wrappers that EXECUTE their argument: allow-listing the wrapper
    # without inspecting its payload launders anything through it.
    "env rm a.txt",
    "command rm a.txt",
    "nice rm a.txt",
    # Control flow: the body is where the damage is, and a classifier
    # that filters unknown node kinds never looks at it.
    "for f in a.txt; do rm $f; done",
    "while true; do rm a.txt; break; done",
    "sh -c 'rm a.txt'",
    "bash -c 'rm a.txt'",
    "cat a.txt | tee out.txt",
    "mkdir sub",
    "ln -s a.txt link.txt",
    "truncate -s 0 a.txt",
    "xargs rm < list.txt",
    "find . -name a.txt -delete",
    'python3 -c \'open("z","w")\'',
    # The program text is not something a regex can analyse, which is
    # why ``awk`` needs its escape hatches gated rather than trusted.
    'awk \'BEGIN{s="sys" "tem"; system("rm a.txt")}\'',
    # -- Attached option values ---------------------------------------
    # ``-oFILE`` carries its value, so a gate testing ``a == "-o"`` never
    # sees it. Measured: this wrote ``out.txt``.
    "sort -oout.txt a.txt",
    # -- Output as a POSITIONAL, with no flag to gate ------------------
    # POSIX gives ``uniq`` and ``xxd`` an optional second operand that is
    # the OUTPUT file. No flag exists to deny, so a flag-shaped gate can
    # never reach these.
    "uniq a.txt out.txt",
    "xxd a.txt out.txt",
    # ``tree -o`` plus its attached spellings.
    "tree -o out.txt .",
    # ``find`` writes through four predicates; ``-fprint0`` was the one
    # missing from a deny list that already held its three siblings.
    "find . -fprint0 out.txt",
    "find . -fprint out.txt",
    "find . -fls out.txt",
    # -- Utilities that mutate with NO flag and NO operand -------------
    # A flag-shaped gate cannot express "always". Each was allow-listed.
    "mypy a.txt",
    # -- Execution laundered through an allow-listed reader ------------
    # ``sed`` already has a gate for the ``w`` command; ``e`` executes the
    # pattern space as a shell command, which is strictly worse.
    "sed 's/x/rm a.txt/e' a.txt",
    # ``git grep -O`` runs its argument as a pager: arbitrary execution
    # inside a subcommand the allow-list calls read-only.
    "git grep -O\"sh -c 'rm a.txt'\" x",
    # A read-only git subcommand still writes when handed an output path.
    "git diff --output=out.txt",
    "git log --output=out.txt",
    # -- Program text loaded from a FILE ------------------------------
    # ``-f`` puts the program where no argv regex looks. Both gates scan
    # the ARGS, so a script that writes or shells out reads clean.
    "sed -f script.sed a.txt",
    "awk -f prog.awk a.txt",
    # ``sed`` separates commands by NEWLINE as well as ``;``, which the
    # write-script pattern did not treat as a boundary.
    "sed '1p\nw out.txt' a.txt",
]


def _snapshot(root: Path) -> dict[str, str]:
    """Content hash of every path under ``root``.

    ``.git`` is excluded: git rewrites its index and reflog as a side
    effect of reading, which would report every ``git`` command as a
    mutator and tell us nothing about the working tree the concurrent
    Bash calls actually share.
    """
    out: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        rel = str(p.relative_to(root))
        if rel == ".git" or rel.startswith(".git/"):
            continue
        if p.is_dir():
            out[rel] = "<dir>"
            continue
        try:
            out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        except OSError:
            # A dangling symlink is itself a change worth recording.
            out[rel] = "<unreadable>"
    return out


def _git_init(root: Path) -> None:
    """Make ``root`` a committed git repo, so ``git`` subcommands run.

    Outside a repo every ``git`` invocation exits before doing anything,
    so ``git log --output=FILE`` looked harmless for the same reason a
    misspelled command does -- the fixture, not the classifier, was
    reporting.
    """
    git = shutil.which("git")
    if git is None:
        return
    ident = ["-c", "user.email=t@t", "-c", "user.name=t"]
    for argv in (["init", "-q"], ["add", "."], [*ident, "commit", "-qm", "x"]):
        _ = subprocess.run(  # noqa: S603 -- fixed argv
            [git, *argv], cwd=root, capture_output=True, check=True
        )


def _mutation_result(command: str) -> tuple[bool, int]:
    """Run ``command``; return whether the tree changed and its exit code."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _ = (root / "a.txt").write_text("x\ny\n")
        _ = (root / "b.txt").write_text("x\nz\n")
        _ = (root / "list.txt").write_text("a.txt\n")
        # Program text held in a FILE, which no argv regex can see.
        _ = (root / "prog.awk").write_text('BEGIN{system("rm a.txt")}')
        _ = (root / "script.sed").write_text("w out.txt")
        _git_init(root)
        before = _snapshot(root)
        result = subprocess.run(  # noqa: S603 -- fixed argv over a fixed command table
            ["/bin/bash", "-c", command],
            cwd=td,
            timeout=10,
            capture_output=True,
            check=False,
        )
        return _snapshot(root) != before, result.returncode


def _mutates(command: str) -> bool:
    """Run ``command`` in a sandbox; report whether the tree changed."""
    changed, _ = _mutation_result(command)
    return changed


def _skip_if_missing(command: str) -> None:
    """Skip when the utility under test is not installed."""
    exe = command.split(maxsplit=1)[0]
    if exe in ("for", "while", "echo", "cat") or shutil.which(exe):
        return
    pytest.skip(f"{exe} not available")


@pytest.mark.slow
@pytest.mark.parametrize("command", _WRITES)
def test_a_command_that_writes_is_never_classified_read_only(command: str) -> None:
    """The failure that costs data, asserted against the filesystem."""
    _skip_if_missing(command)
    trees = parse_bash(command)
    verdict = trees is not None and is_read_only(trees)
    assert not verdict, (
        f"{command!r} was blessed as read-only but CHANGED the sandbox;"
        " two such calls run concurrently and one clobbers the other"
    )
    changed, returncode = _mutation_result(command)
    if returncode != 0 and not changed:
        pytest.skip(f"utility syntax unsupported on this platform: {command!r}")
    assert changed, f"fixture drift: {command!r} no longer writes"


@pytest.mark.slow
@pytest.mark.parametrize("command", _READS)
def test_a_read_only_verdict_is_never_given_to_a_writer(command: str) -> None:
    """A blessed command must leave the tree byte-identical.

    The converse -- a read classified as a writer -- is deliberately
    NOT asserted: it costs serial dispatch, not correctness, and the
    classifier is designed to err that way.
    """
    _skip_if_missing(command)
    trees = parse_bash(command)
    if trees is None or not is_read_only(trees):
        pytest.skip(f"{command!r} classifies as a writer; only speed is lost")
    assert not _mutates(command), (
        f"{command!r} was blessed as read-only but CHANGED the sandbox"
    )


# Commands that EXECUTE an arbitrary child while wearing the name of an
# allow-listed reader. Distinct from the write table: the harm is not a
# changed file but a spawned process, which a tree snapshot only notices
# when the child happens to write.
_EXECUTORS = [
    # ``sed`` executes the pattern space with ``e``; the module already
    # models sed's ``w`` (write) command, so the script language was read
    # and this half was missed.
    "sed 'e' script.txt",
    "sed 's/x/echo hi/e' a.txt",
    # ``git grep -O`` passes its argument to the shell as a pager.
    "git grep -Oecho x",
    # ``awk`` can shell out from inside the program text.
    "awk 'BEGIN{system(\"echo hi\")}'",
    # ``cmd | getline`` runs ``cmd`` through the shell. The pipe is
    # followed by a NAME, not a quote, so a pattern keyed on `|"` missed it.
    "awk 'BEGIN{ cmd=\"echo hi\"; cmd | getline }'",
    # These three are wrapper SCRIPTS: they exec grep, and zgrep also
    # execs gzip. An allow-list of executables cannot see that.
    "zgrep x a.txt",
    "egrep x a.txt",
    "fgrep x a.txt",
]


@pytest.mark.slow
@pytest.mark.parametrize("command", _EXECUTORS)
def test_a_command_that_executes_is_never_classified_read_only(
    command: str,
) -> None:
    """An allow-listed reader that spawns a child is not a reader.

    A tree snapshot cannot see this: the child may only read, or write
    somewhere the sandbox does not cover. The verdict is asserted
    directly because the mechanism -- not its visible effect -- is what
    makes the command unsafe to run concurrently.
    """
    _skip_if_missing(command)
    trees = parse_bash(command)
    verdict = trees is not None and is_read_only(trees)
    assert not verdict, (
        f"{command!r} was blessed as read-only but EXECUTES an arbitrary"
        " child; the allow-list names the wrapper, not what it runs"
    )


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
