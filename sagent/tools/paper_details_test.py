"""Tests for tools.paper_details. All HTTP calls are mocked."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import json

import pytest

from sagent.custom_types import (
    JsonMessage,
    Message,
    MultipartMessage,
    TextMessage,
)
from sagent.lib.json import JSON, json_freeze
from sagent.lib.web.fetch import FetchError
from sagent.tools import paper_details as pb_mod


def _msg(directive: JSON) -> Message:
    return MultipartMessage(
        (JsonMessage(directive, "application/x-tool-paperdetails"),),
        "multipart/x-tool-call",
    )


def _txt(msg: Message) -> str:
    if isinstance(msg, TextMessage):
        return msg.content
    if isinstance(msg, MultipartMessage):
        for p in msg.content:
            if isinstance(p, TextMessage):
                return p.content
    return ""


class _Response:
    def __init__(
        self,
        *,
        status_code: int = 200,
        json_data: Any = None,
    ) -> None:
        self.status_code = status_code
        self.is_success = 200 <= status_code < 300
        self.text = ""
        self._json = json_data

    def json(self) -> Any:
        return self._json


class _Calls:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, str]]] = []


def _patch_client(
    monkeypatch: pytest.MonkeyPatch,
    responses: dict[str, _Response],
) -> _Calls:
    tracker = _Calls()

    def _mock_fetch(url: str, **kwargs: object) -> bytes:
        raw = kwargs.get("params")
        d = cast(dict[str, Any], raw) if isinstance(raw, dict) else {}
        params: dict[str, str] = {str(k): str(v) for k, v in d.items()}
        tracker.calls.append(("GET", url, params))
        for pattern, resp in responses.items():
            if pattern in url:
                if resp.status_code >= 400:
                    raise FetchError(url, resp.status_code, {}, b"")
                return json.dumps(resp._json).encode()
        raise FetchError(url, 404, {}, b"")

    monkeypatch.setattr("sagent.tools.paper_common.fetch", _mock_fetch)
    return tracker


@pytest.fixture(autouse=True)
def _clear_cache() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction] -- pytest fixture used via decorator
    pb_mod._cache.clear()
    yield
    pb_mod._cache.clear()


def _paper(
    *,
    paper_id: str = "abc123",
    doi: str | None = None,
    arxiv: str | None = None,
    title: str = "Test Paper",
    year: int | None = 2020,
) -> dict[str, Any]:
    ext: dict[str, Any] = {}
    if doi:
        ext["DOI"] = doi
    if arxiv:
        ext["ArXiv"] = arxiv
    return {
        "paperId": paper_id,
        "externalIds": ext,
        "title": title,
        "abstract": "some abstract text",
        "authors": [{"name": "Alice"}, {"name": "Bob"}],
        "year": year,
        "venue": "TestConf",
        "citationCount": 10,
        "referenceCount": 5,
        "openAccessPdf": {"url": "https://example.com/p.pdf"},
    }


class TestValidation:
    @pytest.mark.anyio
    async def test_missing_id(self) -> None:
        tool = pb_mod.PaperDetails()
        resp = await tool.run(_msg(json_freeze({"id": ""})))
        assert resp.descriptor == "text/x-error"
        assert "required" in str(resp.content)

    @pytest.mark.anyio
    async def test_bad_id_shape(self) -> None:
        tool = pb_mod.PaperDetails()
        resp = await tool.run(_msg(json_freeze({"id": "not-an-id"})))
        assert resp.descriptor == "text/x-error"
        assert "Unrecognized" in str(resp.content)

    @pytest.mark.anyio
    async def test_bad_operation(self) -> None:
        tool = pb_mod.PaperDetails()
        resp = await tool.run(
            _msg(json_freeze({"id": "10.1234/2", "operation": "search"}))
        )
        assert resp.descriptor == "text/x-error"
        assert "Unknown operation" in str(resp.content)

    @pytest.mark.anyio
    async def test_filter_without_citations_op(self) -> None:
        tool = pb_mod.PaperDetails()
        resp = await tool.run(
            _msg(
                json_freeze(
                    {
                        "id": "10.1234/2",
                        "operation": "references",
                        "influential_only": True,
                    }
                )
            )
        )
        assert resp.descriptor == "text/x-error"
        assert "influential_only" in str(resp.content)


class TestMetadata:
    @pytest.mark.anyio
    async def test_lookup_by_doi(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock = _patch_client(
            monkeypatch,
            {
                "/paper/DOI:10.1234/2": _Response(
                    json_data=_paper(doi="10.1234/2", arxiv="2001.0001"),
                ),
            },
        )
        tool = pb_mod.PaperDetails()
        resp = await tool.run(_msg(json_freeze({"id": "10.1234/2"})))
        assert "title: Test Paper" in _txt(resp)
        assert "doi: 10.1234/2" in _txt(resp)
        assert "id: arXiv:2001.0001" in _txt(resp)
        assert "abstract: some abstract text" in _txt(resp)
        # Correct endpoint / wire id used.
        assert any(
            "paper/DOI:10.1234/2" in call[1] and "references" not in call[1]
            for call in mock.calls
        )

    @pytest.mark.anyio
    async def test_lookup_by_arxiv(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock = _patch_client(
            monkeypatch,
            {
                "/paper/ARXIV:2106.15928": _Response(
                    json_data=_paper(arxiv="2106.15928"),
                ),
            },
        )
        tool = pb_mod.PaperDetails()
        resp = await tool.run(_msg(json_freeze({"id": "arXiv:2106.15928"})))
        assert "id: arXiv:2106.15928" in _txt(resp)
        assert any("ARXIV:2106.15928" in call[1] for call in mock.calls)

    @pytest.mark.anyio
    async def test_abstract_truncation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        long_paper = _paper()
        # 'q' avoids contamination from 'x' in "example.com/p.pdf".
        long_paper["abstract"] = "q" * 500
        _patch_client(
            monkeypatch,
            {"/paper/DOI:10.1234/2": _Response(json_data=long_paper)},
        )
        tool = pb_mod.PaperDetails()
        resp = await tool.run(
            _msg(json_freeze({"id": "10.1234/2", "abstract_chars": 50}))
        )
        assert "..." in _txt(resp)
        assert _txt(resp).count("q") == 50

    @pytest.mark.anyio
    async def test_404_surfaces_cleanly(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_client(
            monkeypatch,
            {"/paper/": _Response(status_code=404)},
        )
        tool = pb_mod.PaperDetails()
        resp = await tool.run(_msg(json_freeze({"id": "10.1234/2"})))
        assert resp.descriptor == "text/x-error"
        assert "Not found" in str(resp.content)

    @pytest.mark.anyio
    async def test_429_surfaces_with_key_hint(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_client(
            monkeypatch,
            {"/paper/": _Response(status_code=429)},
        )
        tool = pb_mod.PaperDetails()
        resp = await tool.run(_msg(json_freeze({"id": "10.1234/2"})))
        assert resp.descriptor == "text/x-error"
        assert "rate limit" in str(resp.content).lower()
        assert "retry" in str(resp.content).lower()

    @pytest.mark.anyio
    async def test_api_key_header(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The mock client doesn't inspect headers, but we can ensure the
        # env var doesn't crash the call.
        monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "test-key")
        _patch_client(
            monkeypatch,
            {"/paper/DOI:10.1234/2": _Response(json_data=_paper(doi="10.1234/2"))},
        )
        tool = pb_mod.PaperDetails()
        resp = await tool.run(_msg(json_freeze({"id": "10.1234/2"})))
        assert "title: Test Paper" in _txt(resp)


class TestReferences:
    @pytest.mark.anyio
    async def test_list(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock = _patch_client(
            monkeypatch,
            {
                "/references": _Response(
                    json_data={
                        "data": [
                            {
                                "isInfluential": True,
                                "citedPaper": _paper(
                                    paper_id="r1",
                                    doi="10.1234/ref1",
                                    title="Reference One",
                                ),
                            },
                            {
                                "isInfluential": False,
                                "citedPaper": _paper(
                                    paper_id="r2",
                                    doi="10.1234/ref2",
                                    title="Reference Two",
                                ),
                            },
                        ],
                    },
                ),
            },
        )
        tool = pb_mod.PaperDetails()
        resp = await tool.run(
            _msg(json_freeze({"id": "10.1234/root", "operation": "references"}))
        )
        assert "Reference One" in _txt(resp)
        assert "Reference Two" in _txt(resp)
        # isInfluential tag on first only
        lines = _txt(resp).split("\n")
        first_line = next(ln for ln in lines if "Reference One" in ln)
        second_line = next(ln for ln in lines if "Reference Two" in ln)
        assert "influential" in first_line
        assert "influential" not in second_line
        # Asked S2 for references endpoint
        assert any("/references" in call[1] for call in mock.calls)

    @pytest.mark.anyio
    async def test_empty(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_client(
            monkeypatch,
            {"/references": _Response(json_data={"data": []})},
        )
        tool = pb_mod.PaperDetails()
        resp = await tool.run(
            _msg(json_freeze({"id": "10.1234/root", "operation": "references"}))
        )
        assert "(no results)" in _txt(resp)


class TestCitations:
    @pytest.mark.anyio
    async def test_list(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_client(
            monkeypatch,
            {
                "/citations": _Response(
                    json_data={
                        "data": [
                            {
                                "isInfluential": True,
                                "citingPaper": _paper(
                                    paper_id="c1",
                                    doi="10.1234/cite1",
                                    title="Citer One",
                                    year=2023,
                                ),
                            },
                        ],
                    },
                ),
            },
        )
        tool = pb_mod.PaperDetails()
        resp = await tool.run(
            _msg(json_freeze({"id": "10.1234/root", "operation": "citations"}))
        )
        assert "Citer One" in _txt(resp)
        assert "influential" in _txt(resp)

    @pytest.mark.anyio
    async def test_influential_only_filter(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_client(
            monkeypatch,
            {
                "/citations": _Response(
                    json_data={
                        "data": [
                            {
                                "isInfluential": True,
                                "citingPaper": _paper(
                                    paper_id="c1",
                                    title="Important Citer",
                                ),
                            },
                            {
                                "isInfluential": False,
                                "citingPaper": _paper(
                                    paper_id="c2",
                                    title="Incidental Citer",
                                ),
                            },
                        ],
                    },
                ),
            },
        )
        tool = pb_mod.PaperDetails()
        resp = await tool.run(
            _msg(
                json_freeze(
                    {
                        "id": "10.1234/root",
                        "operation": "citations",
                        "influential_only": True,
                    }
                )
            )
        )
        assert "Important Citer" in _txt(resp)
        assert "Incidental Citer" not in _txt(resp)

    @pytest.mark.anyio
    async def test_year_from_filter(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_client(
            monkeypatch,
            {
                "/citations": _Response(
                    json_data={
                        "data": [
                            {
                                "isInfluential": False,
                                "citingPaper": _paper(
                                    paper_id="c1",
                                    title="Old Citer",
                                    year=2015,
                                ),
                            },
                            {
                                "isInfluential": False,
                                "citingPaper": _paper(
                                    paper_id="c2",
                                    title="New Citer",
                                    year=2023,
                                ),
                            },
                        ],
                    },
                ),
            },
        )
        tool = pb_mod.PaperDetails()
        resp = await tool.run(
            _msg(
                json_freeze(
                    {"id": "10.1234/root", "operation": "citations", "year_from": 2020}
                )
            )
        )
        assert "New Citer" in _txt(resp)
        assert "Old Citer" not in _txt(resp)

    @pytest.mark.anyio
    async def test_filter_fetches_larger_batch(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock = _patch_client(
            monkeypatch,
            {"/citations": _Response(json_data={"data": []})},
        )
        tool = pb_mod.PaperDetails()
        _ = await tool.run(
            _msg(
                json_freeze(
                    {
                        "id": "10.1234/root",
                        "operation": "citations",
                        "influential_only": True,
                        "limit": 10,
                    }
                )
            )
        )
        # When filter is active, we fetch up to _FILTER_FETCH_CAP.
        _, _, params = mock.calls[0]
        assert params["limit"] == str(pb_mod._FILTER_FETCH_CAP)

    @pytest.mark.anyio
    async def test_no_filter_uses_limit(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock = _patch_client(
            monkeypatch,
            {"/citations": _Response(json_data={"data": []})},
        )
        tool = pb_mod.PaperDetails()
        _ = await tool.run(
            _msg(
                json_freeze(
                    {"id": "10.1234/root", "operation": "citations", "limit": 25}
                )
            )
        )
        _, _, params = mock.calls[0]
        assert params["limit"] == "25"


class TestCache:
    @pytest.mark.anyio
    async def test_hits_cache_on_repeat(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock = _patch_client(
            monkeypatch,
            {"/paper/DOI:10.1234/2": _Response(json_data=_paper(doi="10.1234/2"))},
        )
        tool = pb_mod.PaperDetails()
        _ = await tool.run(_msg(json_freeze({"id": "10.1234/2"})))
        _ = await tool.run(_msg(json_freeze({"id": "10.1234/2"})))
        assert len(mock.calls) == 1

    @pytest.mark.anyio
    async def test_miss_on_different_params(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock = _patch_client(
            monkeypatch,
            {
                "/paper/DOI:10.1234/2": _Response(
                    json_data=_paper(doi="10.1234/2", title="P"),
                ),
            },
        )
        tool = pb_mod.PaperDetails()
        _ = await tool.run(_msg(json_freeze({"id": "10.1234/2"})))
        _ = await tool.run(_msg(json_freeze({"id": "10.1234/2", "abstract_chars": 50})))
        assert len(mock.calls) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
