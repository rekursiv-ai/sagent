"""Tests for the ``code`` mode, on scripted models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import asyncio
import re

import pytest

from sagent.journal.types import UserMessage
from sagent.model.exceptions import RetryableError
from sagent.model.scripted import ScriptedProvider, call, reply
from sagent.scripts.prompt_debug.harness import Case, Run
from sagent.scripts.prompt_debug.modes.code import CodeMode


if TYPE_CHECKING:
    from collections.abc import Callable

    from sagent.model.types import Request, Response


_SECTIONS = {"file: widget.py": "def f():\n    pass\n"}


def _parse(meta: dict[str, object]) -> CodeMode:
    return CodeMode.parse(meta, _SECTIONS, where="c.md")


def test_the_mode_reads_its_regex_and_file() -> None:
    mode = _parse({"flag_regex": "return _", "file": "widget.py"})

    assert mode == CodeMode(flag_regex="return _", file="widget.py")


@pytest.mark.parametrize(
    ("meta", "error", "said"),
    [
        ({"flag_regex": 1, "file": "widget.py"}, TypeError, "flag_regex and file must"),
        ({"flag_regex": "x", "file": 1}, TypeError, "flag_regex and file must"),
        ({"file": "widget.py"}, ValueError, "code mode requires flag_regex and file."),
        ({"flag_regex": "x"}, ValueError, "code mode requires flag_regex and file."),
        ({"flag_regex": "x", "file": "a.py"}, ValueError, "no '## file: a.py' section"),
        (
            {"flag_regex": "(", "file": "widget.py"},
            ValueError,
            "invalid flag_regex '\\('",
        ),
    ],
)
def test_the_mode_refuses_what_it_cannot_score(
    meta: dict[str, object],
    error: type[Exception],
    said: str,
) -> None:
    with pytest.raises(error, match=f"^c\\.md: {said}"):
        _ = _parse(meta)


def _editing(new: str) -> Callable[[Request], Response]:
    """Return a reply that rewrites the body of the file the task names."""

    def edit(request: Request) -> Response:
        (task,) = [r.content for r in request.records if isinstance(r, UserMessage)]
        match = re.search(r"\S*widget\.py", task or "")
        assert match is not None
        return reply(
            calls=[
                call("Edit", file_path=match[0], old_string="    pass", new_string=new),
            ],
        )

    return edit


@pytest.mark.parametrize(
    ("new", "flagged"),
    [
        ("    return _helper()", True),
        ("    return Widget(size=2)", False),
    ],
)
def test_a_trial_matches_the_edited_file(new: str, flagged: bool) -> None:
    provider = ScriptedProvider(replies=[_editing(new), reply("done")])
    model = provider["m"]
    mode = CodeMode(flag_regex=r"return _[A-Za-z_]+\(", file="widget.py")
    case = Case(
        name="c",
        n=1,
        max_flag_rate=0.0,
        mode=mode,
        producer_task="Edit {DIR}/widget.py.",
        files={"widget.py": "def f():\n    pass\n"},
    )

    run = Run(model=model, judge_model=model, system="S", cli_context=False)

    assert asyncio.run(mode.trial(case, run)) is flagged
    tools = {t.name for t in provider.requests[0].tools}
    assert tools == {"Read", "Edit", "Glob", "Grep", "Skill"}


def test_a_trial_that_edits_nothing_is_void_and_retried() -> None:
    model = ScriptedProvider(replies=[reply("done")])["m"]
    mode = CodeMode(flag_regex="x", file="widget.py")
    case = Case(
        name="c",
        n=1,
        max_flag_rate=0.0,
        mode=mode,
        producer_task="Edit {DIR}/widget.py.",
        files={"widget.py": "x\n"},
    )
    run = Run(model=model, judge_model=model, system="S", cli_context=False)

    with pytest.raises(
        RetryableError,
        match=r"^void trial: producer left widget\.py unchanged$",
    ):
        _ = asyncio.run(mode.trial(case, run))


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
