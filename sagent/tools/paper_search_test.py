"""Tests for tools.paper_search. All HTTP calls are mocked."""

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
from sagent.tools import paper_search as ps_mod


def _msg(directive: JSON) -> Message:
    return MultipartMessage(
        (JsonMessage(directive, "application/x-tool-papersearch"),),
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
    *,
    s2: _Response | None = None,
    openalex: _Response | None = None,
) -> _Calls:
    tracker = _Calls()

    def _mock_fetch(url: str, **kwargs: object) -> bytes:
        raw = kwargs.get("params")
        d = cast(dict[str, Any], raw) if isinstance(raw, dict) else {}
        params: dict[str, str] = {str(k): str(v) for k, v in d.items()}
        tracker.calls.append(("GET", url, params))
        if "semanticscholar.org" in url:
            if s2 is None:
                raise FetchError(url, 503, {}, b"")
            if s2.status_code >= 400:
                raise FetchError(url, s2.status_code, {}, b"")
            return json.dumps(s2._json).encode()
        if "openalex.org" in url:
            if openalex is None:
                raise FetchError(url, 503, {}, b"")
            if openalex.status_code >= 400:
                raise FetchError(url, openalex.status_code, {}, b"")
            return json.dumps(openalex._json).encode()
        raise FetchError(url, 404, {}, b"")

    monkeypatch.setattr("sagent.tools.paper_search.fetch", _mock_fetch)
    monkeypatch.setattr("sagent.tools.paper_common.fetch", _mock_fetch)
    return tracker


@pytest.fixture(autouse=True)
def _clear_cache() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction] -- pytest fixture used via decorator
    ps_mod._cache.clear()
    yield
    ps_mod._cache.clear()


def _s2_hit(
    *,
    paper_id: str = "p1",
    title: str = "Example",
    doi: str | None = None,
    arxiv: str | None = None,
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
        "abstract": "s2 abstract",
        "authors": [{"name": "Alice"}],
        "year": 2020,
        "venue": "TestConf",
        "citationCount": 10,
        "referenceCount": 3,
        "openAccessPdf": {"url": "https://x/y"},
    }


def _oa_hit(
    *,
    doi: str | None = None,
    arxiv: str | None = None,
    title: str = "Example",
) -> dict[str, Any]:
    ids: dict[str, Any] = {}
    if doi:
        ids["doi"] = f"https://doi.org/{doi}"
    if arxiv:
        ids["arxiv"] = f"https://arxiv.org/abs/{arxiv}"
    return {
        "id": "https://openalex.org/W1",
        "doi": f"https://doi.org/{doi}" if doi else None,
        "ids": ids,
        "title": title,
        "display_name": title,
        "authorships": [{"author": {"display_name": "Bob"}}],
        "publication_year": 2021,
        "primary_location": {"source": {"display_name": "OAVenue"}},
        "cited_by_count": 7,
        "referenced_works_count": 4,
        "abstract_inverted_index": {
            "Hello": [0, 2],
            "world": [1, 3],
        },
        "open_access": {"is_oa": True, "oa_url": "https://oa/url"},
    }


class TestValidation:
    @pytest.mark.anyio
    async def test_missing_query(self) -> None:
        tool = ps_mod.PaperSearch()
        resp = await tool.run(_msg(json_freeze({"query": ""})))
        assert resp.descriptor == "text/x-error"
        assert "required" in str(resp.content)

    @pytest.mark.anyio
    async def test_bad_source(self) -> None:
        tool = ps_mod.PaperSearch()
        resp = await tool.run(_msg(json_freeze({"query": "x", "source": "google"})))
        assert resp.descriptor == "text/x-error"
        assert "Invalid source" in str(resp.content)

    @pytest.mark.anyio
    async def test_default_source_is_s2(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Without ``source=``, only S2 is queried - not OpenAlex."""
        mock = _patch_client(
            monkeypatch,
            s2=_Response(
                json_data={
                    "total": 1,
                    "data": [_s2_hit(title="Only", doi="10.1234/a")],
                },
            ),
            openalex=_Response(
                json_data={
                    "meta": {"count": 1},
                    "results": [_oa_hit(title="Other", doi="10.1234/b")],
                },
            ),
        )
        tool = ps_mod.PaperSearch()
        resp = await tool.run(_msg(json_freeze({"query": "x"})))
        assert "Only" in _txt(resp)
        assert "Other" not in _txt(resp)
        # Exactly one API call - to S2.
        assert len(mock.calls) == 1
        assert "semanticscholar.org" in mock.calls[0][1]


class TestS2Only:
    @pytest.mark.anyio
    async def test_returns_formatted_hits(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_client(
            monkeypatch,
            s2=_Response(
                json_data={
                    "total": 2,
                    "data": [
                        _s2_hit(title="Attn", doi="10.1234/a"),
                        _s2_hit(title="BERT", doi="10.1234/b", paper_id="p2"),
                    ],
                },
            ),
        )
        tool = ps_mod.PaperSearch()
        resp = await tool.run(
            _msg(json_freeze({"query": "transformer", "source": "s2"}))
        )
        assert "Attn" in _txt(resp)
        assert "BERT" in _txt(resp)
        assert "sources: s2" in _txt(resp)

    @pytest.mark.anyio
    async def test_year_filter_sent_to_s2(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock = _patch_client(
            monkeypatch,
            s2=_Response(json_data={"total": 0, "data": []}),
        )
        tool = ps_mod.PaperSearch()
        _ = await tool.run(
            _msg(
                json_freeze(
                    {"query": "x", "source": "s2", "year_from": 2020, "year_to": 2023}
                )
            )
        )
        _, _, params = mock.calls[0]
        assert params["year"] == "2020-2023"

    @pytest.mark.anyio
    async def test_oa_flag(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock = _patch_client(
            monkeypatch,
            s2=_Response(json_data={"total": 0, "data": []}),
        )
        tool = ps_mod.PaperSearch()
        _ = await tool.run(
            _msg(json_freeze({"query": "x", "source": "s2", "open_access_only": True}))
        )
        _, _, params = mock.calls[0]
        assert "openAccessPdf" in params

    @pytest.mark.anyio
    async def test_429_hint(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_client(monkeypatch, s2=_Response(status_code=429))
        tool = ps_mod.PaperSearch()
        resp = await tool.run(_msg(json_freeze({"query": "x", "source": "s2"})))
        assert resp.descriptor == "text/x-error"
        assert "rate limit" in str(resp.content).lower()


class TestOpenAlexOnly:
    @pytest.mark.anyio
    async def test_returns_formatted_hits(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_client(
            monkeypatch,
            openalex=_Response(
                json_data={
                    "meta": {"count": 1},
                    "results": [_oa_hit(doi="10.1234/oa", title="OA Paper")],
                },
            ),
        )
        tool = ps_mod.PaperSearch()
        resp = await tool.run(_msg(json_freeze({"query": "x", "source": "openalex"})))
        assert "OA Paper" in _txt(resp)
        assert "sources: openalex" in _txt(resp)
        assert "doi:10.1234/oa" in _txt(resp)
        # Abstract reconstructed from inverted index
        assert "Hello world" in _txt(resp)

    @pytest.mark.anyio
    async def test_filter_includes_open_access(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock = _patch_client(
            monkeypatch,
            openalex=_Response(json_data={"meta": {"count": 0}, "results": []}),
        )
        tool = ps_mod.PaperSearch()
        _ = await tool.run(
            _msg(
                json_freeze(
                    {"query": "x", "source": "openalex", "open_access_only": True}
                )
            )
        )
        _, _, params = mock.calls[0]
        assert "open_access.is_oa:true" in params["filter"]

    @pytest.mark.anyio
    async def test_year_range_becomes_date_range(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock = _patch_client(
            monkeypatch,
            openalex=_Response(json_data={"meta": {"count": 0}, "results": []}),
        )
        tool = ps_mod.PaperSearch()
        _ = await tool.run(
            _msg(
                json_freeze(
                    {
                        "query": "x",
                        "source": "openalex",
                        "year_from": 2020,
                        "year_to": 2023,
                    }
                )
            )
        )
        _, _, params = mock.calls[0]
        assert "from_publication_date:2020-01-01" in params["filter"]
        assert "to_publication_date:2023-12-31" in params["filter"]

    @pytest.mark.anyio
    async def test_polite_pool_ua(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OPENALEX_EMAIL", "test@example.com")
        # Just ensure this path doesn't raise; header inspection would
        # require a richer mock.
        _patch_client(
            monkeypatch,
            openalex=_Response(json_data={"meta": {"count": 0}, "results": []}),
        )
        tool = ps_mod.PaperSearch()
        resp = await tool.run(_msg(json_freeze({"query": "x", "source": "openalex"})))
        assert resp.content is not None


class TestFused:
    @pytest.mark.anyio
    async def test_merges_and_dedups_by_doi(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        shared_doi = "10.1234/shared"
        _patch_client(
            monkeypatch,
            s2=_Response(
                json_data={
                    "total": 2,
                    "data": [
                        _s2_hit(title="Shared", doi=shared_doi),
                        _s2_hit(title="S2Only", doi="10.1234/s2", paper_id="p2"),
                    ],
                },
            ),
            openalex=_Response(
                json_data={
                    "meta": {"count": 2},
                    "results": [
                        _oa_hit(title="Shared (casing off)", doi=shared_doi),
                        _oa_hit(title="OAOnly", doi="10.1234/oa"),
                    ],
                },
            ),
        )
        tool = ps_mod.PaperSearch()
        resp = await tool.run(_msg(json_freeze({"query": "x", "source": "fused"})))
        # S2 rank is first - "Shared" should precede "S2Only" should
        # precede "OAOnly" in the rendered output.
        lines = [ln for ln in _txt(resp).split("\n") if ln.startswith("[")]
        titles = list(lines)
        shared_idx = next(i for i, t in enumerate(titles) if "Shared" in t)
        s2_only_idx = next(i for i, t in enumerate(titles) if "S2Only" in t)
        oa_only_idx = next(i for i, t in enumerate(titles) if "OAOnly" in t)
        assert shared_idx < s2_only_idx
        assert s2_only_idx < oa_only_idx
        # Shared record should carry both sources
        shared_line = next(ln for ln in lines if "Shared" in ln)
        assert "sources: s2,openalex" in shared_line
        # S2-only record should have only s2 in sources
        s2_line = next(ln for ln in lines if "S2Only" in ln)
        assert "sources: s2" in s2_line
        assert "openalex" not in s2_line
        # OA-only record should have only openalex
        oa_line = next(ln for ln in lines if "OAOnly" in ln)
        assert "sources: openalex" in oa_line

    @pytest.mark.anyio
    async def test_dedup_by_title_when_doi_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_client(
            monkeypatch,
            s2=_Response(
                json_data={
                    "total": 1,
                    "data": [_s2_hit(title="A Paper About Things")],
                },
            ),
            openalex=_Response(
                json_data={
                    "meta": {"count": 1},
                    "results": [_oa_hit(title="A  paper  about  things!")],
                },
            ),
        )
        tool = ps_mod.PaperSearch()
        resp = await tool.run(_msg(json_freeze({"query": "x", "source": "fused"})))
        # Should be deduped (punctuation / case / spacing normalized)
        assert _txt(resp).count("A Paper About Things") == 1

    @pytest.mark.anyio
    async def test_s2_failure_degrades_to_openalex(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_client(
            monkeypatch,
            s2=_Response(status_code=500),
            openalex=_Response(
                json_data={
                    "meta": {"count": 1},
                    "results": [_oa_hit(title="OAHit", doi="10.1234/a")],
                },
            ),
        )
        tool = ps_mod.PaperSearch()
        resp = await tool.run(_msg(json_freeze({"query": "x", "source": "fused"})))
        assert "OAHit" in _txt(resp)

    @pytest.mark.anyio
    async def test_both_failures_is_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_client(
            monkeypatch,
            s2=_Response(status_code=500),
            openalex=_Response(status_code=500),
        )
        tool = ps_mod.PaperSearch()
        resp = await tool.run(_msg(json_freeze({"query": "x", "source": "fused"})))
        assert resp.descriptor == "text/x-error"


class TestCache:
    @pytest.mark.anyio
    async def test_cache_hit(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock = _patch_client(
            monkeypatch,
            s2=_Response(
                json_data={
                    "total": 1,
                    "data": [_s2_hit(title="Only", doi="10.1234/a")],
                },
            ),
        )
        tool = ps_mod.PaperSearch()
        _ = await tool.run(_msg(json_freeze({"query": "foo", "source": "s2"})))
        _ = await tool.run(_msg(json_freeze({"query": "foo", "source": "s2"})))
        assert len(mock.calls) == 1


class TestOpenAlexParsing:
    @pytest.mark.anyio
    async def test_strips_doi_prefix(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_client(
            monkeypatch,
            openalex=_Response(
                json_data={
                    "meta": {"count": 1},
                    "results": [_oa_hit(doi="10.1234/abc", title="T")],
                },
            ),
        )
        tool = ps_mod.PaperSearch()
        resp = await tool.run(_msg(json_freeze({"query": "x", "source": "openalex"})))
        assert "doi:10.1234/abc" in _txt(resp)
        assert "https://doi.org" not in _txt(resp)  # prefix stripped

    @pytest.mark.anyio
    async def test_extracts_arxiv_from_ids(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_client(
            monkeypatch,
            openalex=_Response(
                json_data={
                    "meta": {"count": 1},
                    "results": [_oa_hit(arxiv="2106.15928", title="T")],
                },
            ),
        )
        tool = ps_mod.PaperSearch()
        resp = await tool.run(_msg(json_freeze({"query": "x", "source": "openalex"})))
        assert "arXiv:2106.15928" in _txt(resp)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
