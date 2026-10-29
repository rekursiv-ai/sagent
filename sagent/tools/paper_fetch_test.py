"""Tests for PaperFetch: downloads, the cache on disk, and how they fail.

``wesearch`` owns every download; each is replaced where the tool imports
it, so nothing reaches the network.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING

import importlib

from wesearch.fetch import FetchSession
from wesearch.paper.errors import PaperError

import pytest

from sagent.lib.userdirs import cache_dir
from sagent.tools import paper_fetch
from sagent.tools.paper_fetch import PaperFetch
from sagent.tools.types import ToolContext


if TYPE_CHECKING:
    from treekle import PlainTree


_PDF = b"%PDF-1.7\n" + b"x" * 200


def test_a_download_needs_the_session_it_shares() -> None:
    (session,) = [f for f in fields(PaperFetch) if f.name == "session"]
    assert (session.default, session.default_factory) == (MISSING, MISSING)


def test_pdfs_are_kept_in_the_users_cache_by_default() -> None:
    assert _tool().directory == cache_dir() / "rekursiv-ai" / "wesearch" / "papers"


def test_the_description_is_read_as_utf8_whatever_the_locale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The asset holds an em dash, which a Latin-1 locale would read as three letters."""
    real = Path.read_text

    def latin1_default(
        self: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        return real(
            self,
            encoding=encoding or "latin-1",
            errors=errors,
        )

    monkeypatch.setattr(Path, "read_text", latin1_default)
    try:
        importlib.reload(paper_fetch)
        description = paper_fetch.PaperFetch.description
    finally:
        monkeypatch.undo()
        importlib.reload(paper_fetch)
    assert "\u2014" in description
    assert "â€" not in description


def test_summary_names_the_ids() -> None:
    assert _tool().summary({"ids": ["2106.15928", "10.1000/x"]}) == (
        "PaperFetch 2106.15928 (+1 more)"
    )
    assert _tool().summary({}) == "PaperFetch ?"


def test_downloads_run_beside_anything(tmp_path: Path) -> None:
    assert _tool().serialize_key(ToolContext(cwd=tmp_path), {"ids": "a"}) is None


@pytest.mark.asyncio
async def test_one_paper_is_downloaded_and_kept(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _downloads(monkeypatch, {"2106.15928": (_PDF, "arXiv")})
    batches = _batches(monkeypatch)
    tool = _tool(tmp_path)
    result = await tool.call(ToolContext(cwd=tmp_path), {"ids": "arXiv:2106.15928"})
    path = tmp_path / "arxiv_2106.15928.pdf"
    assert result.text() == f"Downloaded via arXiv: {path}"
    assert result.is_error is False
    assert result.call_id == ""
    assert path.read_bytes() == _PDF
    assert batches == []
    assert calls == [
        _Call(
            args=("arxiv", "2106.15928"),
            kwargs={"oa_url": None, "oa_looked_up": False, "session": tool.session},
        ),
    ]


@pytest.mark.asyncio
async def test_a_kept_pdf_is_returned_without_a_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "doi_10.1000_x.pdf").write_bytes(_PDF)
    calls = _downloads(monkeypatch)
    result = await _tool(tmp_path).call(ToolContext(cwd=tmp_path), {"ids": "10.1000/x"})
    assert result.text() == f"Cached: {tmp_path / 'doi_10.1000_x.pdf'}"
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kept",
    [b"%PDF-1.7", b"<html>" + b"x" * 200],
    ids=["short", "html"],
)
async def test_a_kept_file_that_is_no_pdf_is_downloaded_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kept: bytes,
) -> None:
    path = tmp_path / "doi_10.1000_x.pdf"
    path.write_bytes(kept)
    _downloads(monkeypatch, {"10.1000/x": (_PDF, "open access")})
    result = await _tool(tmp_path).call(ToolContext(cwd=tmp_path), {"ids": "10.1000/x"})
    assert result.text() == f"Downloaded via open access: {path}"
    assert path.read_bytes() == _PDF


@pytest.mark.asyncio
async def test_many_papers_share_one_lookup_and_report_in_input_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batches = _batches(monkeypatch, ["https://oa/a.pdf", None])
    calls = _downloads(
        monkeypatch,
        {
            "10.1000/a": (_PDF, "open access"),
            "10.1000/b": PaperError("no copy of 10.1000/b"),
        },
    )
    tool = _tool(tmp_path)
    result = await tool.call(
        ToolContext(cwd=tmp_path),
        {"ids": ["10.1000/a", "10.1000/b"]},
    )
    assert result.text() == (
        f"Downloaded via open access: {tmp_path / 'doi_10.1000_a.pdf'}\n"
        "no copy of 10.1000/b"
    )
    assert result.is_error is True
    assert batches == [
        _Call(
            args=(["DOI:10.1000/a", "DOI:10.1000/b"],),
            kwargs={"session": tool.session},
        ),
    ]
    assert [c.kwargs["oa_url"] for c in calls] == ["https://oa/a.pdf", None]
    assert [c.kwargs["oa_looked_up"] for c in calls] == [True, True]


@pytest.mark.asyncio
async def test_a_failed_lookup_leaves_each_download_to_find_its_own(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _batches(monkeypatch, None)
    calls = _downloads(
        monkeypatch,
        {"2106.15928": (_PDF, "arXiv"), "2106.15929": (_PDF, "arXiv")},
    )
    result = await _tool(tmp_path).call(
        ToolContext(cwd=tmp_path),
        {"ids": "2106.15928, 2106.15929"},
    )
    assert result.is_error is False
    assert [(c.kwargs["oa_url"], c.kwargs["oa_looked_up"]) for c in calls] == [
        (None, False),
        (None, False),
    ]


@pytest.mark.asyncio
async def test_every_paper_failing_is_one_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _batches(monkeypatch, [None, None])
    _downloads(
        monkeypatch,
        {"10.1000/a": PaperError("no a"), "10.1000/b": PaperError("no b")},
    )
    result = await _tool(tmp_path).call(
        ToolContext(cwd=tmp_path),
        {"ids": ["10.1000/a", "10.1000/b"]},
    )
    assert (result.is_error, result.text()) == (True, "no a\nno b")
    assert _names(tmp_path) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"ids": ""}, "'ids' is empty."),
        ({"ids": []}, "'ids' is empty."),
        ({"ids": "not-an-id"}, "not-an-id"),
        ({"ids": ["10.1000/a", "nope"]}, "nope"),
        ({"ids": 3}, "'ids' must be"),
        ({}, "Missing argument 'ids'"),
    ],
)
async def test_ids_that_name_no_paper_are_refused_before_any_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arguments: dict[str, PlainTree],
    message: str,
) -> None:
    calls = _downloads(monkeypatch)
    batches = _batches(monkeypatch)
    with pytest.raises(ValueError, match=message):
        await _tool(tmp_path).call(ToolContext(cwd=tmp_path), arguments)
    assert (calls, batches) == ([], [])


@dataclass(frozen=True, slots=True, kw_only=True)
class _Call:
    """One lookup or download the tool asked for."""

    args: tuple[object, ...]
    kwargs: dict[str, object]


def _tool(directory: Path | None = None) -> PaperFetch:
    session = FetchSession()
    return (
        PaperFetch(session=session)
        if directory is None
        else PaperFetch(
            session=session,
            directory=directory,
        )
    )


# The downloads run at once, so the order they arrive in is no order at all.
def _downloads(
    monkeypatch: pytest.MonkeyPatch,
    answers: dict[str, tuple[bytes, str] | Exception] | None = None,
) -> list[_Call]:
    """Answer each download by its paper's id; return what was asked, in id order."""
    calls: list[_Call] = []

    def download(*args: object, **kwargs: object) -> tuple[bytes, str]:
        calls.append(_Call(args=args, kwargs=kwargs))
        calls.sort(key=lambda c: str(c.args))
        answer = (answers or {})[str(args[1])]
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(paper_fetch, "download", download)
    return calls


def _batches(
    monkeypatch: pytest.MonkeyPatch,
    *answers: list[str | None] | None,
) -> list[_Call]:
    """Answer each batched lookup with the next of ``answers``; return what was asked."""
    calls: list[_Call] = []
    pending = list(answers)

    def batch_oa_urls(*args: object, **kwargs: object) -> list[str | None] | None:
        calls.append(_Call(args=args, kwargs=kwargs))
        return pending.pop(0)

    monkeypatch.setattr(paper_fetch, "batch_oa_urls", batch_oa_urls)
    return calls


def _names(directory: Path) -> list[str]:
    """Return the names in ``directory``, sorted."""
    return sorted(p.name for p in directory.iterdir())


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
