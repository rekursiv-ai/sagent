"""Tests for converting old sagent session logs into today's journals."""

from __future__ import annotations

from typing import TYPE_CHECKING

import asyncio

from treekle import json_dumps

from sagent.lib.userdirs import data_dir
from sagent.scripts.convert_sessions import main
from sagent.session.jsonl import JsonlJournal
from sagent.session.types import ContextClear, UserMessage


if TYPE_CHECKING:
    from pathlib import Path

    from treekle import PlainTree

    import pytest

    from sagent.session.types import SessionRecord


def _old(directory: Path, *records: PlainTree) -> Path:
    path = directory / "session.jsonl"
    path.parent.mkdir(parents=True)
    _ = path.write_text("".join(json_dumps(r) + "\n" for r in records))
    return path


async def _window(path: Path) -> list[SessionRecord]:
    async with JsonlJournal(path) as journal:
        return list(journal)


def _said(text: str) -> dict[str, PlainTree]:
    return {"kind": "history", "type": "user", "text": text}


def test_an_old_session_is_converted_in_place(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    log = _old(
        tmp_path / "_p" / "abc",
        {"kind": "meta", "session_id": "abc"},
        _said("hi"),
    )

    assert main([str(tmp_path)]) == 0

    clear, said = asyncio.run(_window(log))
    assert isinstance(clear, ContextClear)
    assert said == UserMessage(content="hi")
    assert not log.with_name("journal.jsonl").exists()
    assert capsys.readouterr().out == f"converted: {log}\n1 converted\n"


def test_a_converted_session_is_never_converted_again(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _old(tmp_path / "x" / "abc", _said("hi"))
    _old(tmp_path / "z" / "empty", {"kind": "meta", "session_id": "empty"})

    assert main([str(tmp_path)]) == 0
    assert main([str(tmp_path)]) == 0

    out = capsys.readouterr().out.splitlines()
    assert out[-2] == "1 converted, 1 skipped: nothing to resume"
    assert out[-1] == "1 skipped: not old, 1 skipped: nothing to resume"


def test_a_log_of_todays_records_is_not_old(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    log = _old(tmp_path / "s1", {"py/object": "ContextClear"})

    assert main([str(log)]) == 0

    assert capsys.readouterr().out == "1 skipped: not old\n"


def test_a_dry_run_writes_nothing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    log = _old(tmp_path / "s1", _said("hi"))
    before = log.read_bytes()
    _old(tmp_path / "s2", {"kind": "meta", "session_id": "s2"})

    assert main([str(tmp_path), "--dry-run"]) == 0

    assert capsys.readouterr().out == (
        f"would convert: {log}\n1 skipped: nothing to resume, 1 would convert\n"
    )
    assert log.read_bytes() == before


def test_the_projects_root_is_the_users_by_default(
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = data_dir() / "rekursiv-ai" / "sagent" / "projects"

    assert main([]) == 0
    log = _old(root / "_p" / "s2", _said("hi"))
    assert main([]) == 0

    assert capsys.readouterr().out.splitlines()[0] == "no logs"
    assert log.read_text().startswith('{"py/object": "ContextClear"')


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
