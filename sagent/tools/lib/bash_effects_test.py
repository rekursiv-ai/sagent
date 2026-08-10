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
]


def _snapshot(root: Path) -> dict[str, str]:
    """Content hash of every path under ``root``."""
    out: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        rel = str(p.relative_to(root))
        if p.is_dir():
            out[rel] = "<dir>"
            continue
        try:
            out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        except OSError:
            # A dangling symlink is itself a change worth recording.
            out[rel] = "<unreadable>"
    return out


def _mutates(command: str) -> bool:
    """Run ``command`` in a sandbox; report whether the tree changed."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _ = (root / "a.txt").write_text("x\ny\n")
        _ = (root / "b.txt").write_text("x\nz\n")
        _ = (root / "list.txt").write_text("a.txt\n")
        before = _snapshot(root)
        _ = subprocess.run(  # noqa: S603 -- fixed argv over a fixed command table
            ["/bin/bash", "-c", command],
            cwd=td,
            timeout=10,
            capture_output=True,
            check=False,
        )
        return _snapshot(root) != before


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
    assert _mutates(command), f"fixture drift: {command!r} no longer writes"
    trees = parse_bash(command)
    verdict = trees is not None and is_read_only(trees)
    assert not verdict, (
        f"{command!r} was blessed as read-only but CHANGED the sandbox;"
        " two such calls run concurrently and one clobbers the other"
    )


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


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
