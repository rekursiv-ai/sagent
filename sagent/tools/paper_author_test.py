"""Tests for tools.paper_author. All HTTP calls are mocked."""

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
from sagent.tools import paper_author as pa_mod


def _msg(directive: JSON) -> Message:
    return MultipartMessage(
        (JsonMessage(directive, "application/x-tool-paperauthor"),),
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
    pa_mod._cache.clear()
    yield
    pa_mod._cache.clear()


# -- Fixture helpers -------------------------------------------------------


def _author(
    *,
    author_id: str = "1",
    name: str = "Alice Author",
    aliases: list[str] | None = None,
    affiliations: list[Any] | None = None,
    homepage: str | None = None,
    h_index: int | None = 10,
    citation_count: int | None = 100,
    paper_count: int | None = 50,
) -> dict[str, Any]:
    return {
        "authorId": author_id,
        "name": name,
        "aliases": aliases if aliases is not None else [],
        "affiliations": affiliations if affiliations is not None else [],
        "homepage": homepage,
        "hIndex": h_index,
        "citationCount": citation_count,
        "paperCount": paper_count,
    }


def _paper(
    *,
    paper_id: str = "p1",
    title: str = "A Paper",
    doi: str | None = None,
    year: int | None = 2020,
) -> dict[str, Any]:
    ext: dict[str, Any] = {}
    if doi:
        ext["DOI"] = doi
    return {
        "paperId": paper_id,
        "externalIds": ext,
        "title": title,
        "abstract": "abs",
        "authors": [{"name": "Alice"}],
        "year": year,
        "venue": "Venue",
        "citationCount": 5,
        "referenceCount": 3,
        "openAccessPdf": None,
    }


# -- Validation ------------------------------------------------------------


class TestValidation:
    @pytest.mark.anyio
    async def test_neither_query_nor_id(self) -> None:
        tool = pa_mod.PaperAuthor()
        resp = await tool.run(_msg(json_freeze({})))
        assert resp.descriptor == "text/x-error"
        assert "required" in str(resp.content)

    @pytest.mark.anyio
    async def test_both_query_and_id(self) -> None:
        tool = pa_mod.PaperAuthor()
        resp = await tool.run(_msg(json_freeze({"query": "x", "id": "1"})))
        assert resp.descriptor == "text/x-error"
        assert "exactly one" in str(resp.content)

    @pytest.mark.anyio
    async def test_bad_operation(self) -> None:
        tool = pa_mod.PaperAuthor()
        resp = await tool.run(_msg(json_freeze({"id": "1", "operation": "details"})))
        assert resp.descriptor == "text/x-error"
        assert "Unknown operation" in str(resp.content)

    @pytest.mark.anyio
    async def test_operation_without_id(self) -> None:
        tool = pa_mod.PaperAuthor()
        resp = await tool.run(_msg(json_freeze({"query": "x", "operation": "papers"})))
        assert resp.descriptor == "text/x-error"
        assert "operation" in str(resp.content)

    @pytest.mark.anyio
    async def test_operation_with_query(self) -> None:
        # query + operation="papers" is rejected: operation needs id.
        tool = pa_mod.PaperAuthor()
        resp = await tool.run(_msg(json_freeze({"query": "x", "operation": "papers"})))
        assert resp.descriptor == "text/x-error"
        assert "'operation' requires 'id'" in str(resp.content)

    @pytest.mark.anyio
    async def test_year_with_query(self) -> None:
        tool = pa_mod.PaperAuthor()
        resp = await tool.run(_msg(json_freeze({"query": "x", "year_from": 2020})))
        assert resp.descriptor == "text/x-error"
        assert "year_from" in str(resp.content)

    @pytest.mark.anyio
    async def test_year_with_bare_id(self) -> None:
        # year_from on id alone (no operation="papers") is rejected.
        tool = pa_mod.PaperAuthor()
        resp = await tool.run(_msg(json_freeze({"id": "1", "year_from": 2020})))
        assert resp.descriptor == "text/x-error"
        assert "year_from" in str(resp.content)


# -- Search ----------------------------------------------------------------


class TestSearch:
    @pytest.mark.anyio
    async def test_returns_ranked_candidates(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_client(
            monkeypatch,
            {
                "/author/search": _Response(
                    json_data={
                        "total": 2,
                        "data": [
                            _author(
                                author_id="1",
                                name="Junior",
                                h_index=5,
                                affiliations=["A"],
                            ),
                            _author(
                                author_id="2",
                                name="Senior",
                                h_index=50,
                                affiliations=["B"],
                            ),
                        ],
                    },
                ),
            },
        )
        tool = pa_mod.PaperAuthor()
        resp = await tool.run(_msg(json_freeze({"query": "common name"})))
        # Sorted by h-index desc, so Senior appears first.
        lines = _txt(resp).split("\n")
        assert "Senior" in lines[0]
        assert "Junior" in lines[1]
        assert "h-index:50" in lines[0]
        assert "h-index:5" in lines[1]

    @pytest.mark.anyio
    async def test_shows_affiliation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_client(
            monkeypatch,
            {
                "/author/search": _Response(
                    json_data={
                        "total": 1,
                        "data": [
                            _author(
                                name="Bob",
                                affiliations=["Univ X", "Lab Y"],
                            ),
                        ],
                    },
                ),
            },
        )
        tool = pa_mod.PaperAuthor()
        resp = await tool.run(_msg(json_freeze({"query": "bob"})))
        assert "Univ X" in _txt(resp)
        # Only primary affiliation on the one-liner
        assert "Lab Y" not in _txt(resp)

    @pytest.mark.anyio
    async def test_empty(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_client(
            monkeypatch,
            {
                "/author/search": _Response(
                    json_data={"total": 0, "data": []},
                ),
            },
        )
        tool = pa_mod.PaperAuthor()
        resp = await tool.run(_msg(json_freeze({"query": "nobody"})))
        assert "(no results)" in _txt(resp)

    @pytest.mark.anyio
    async def test_affiliations_dict_shape(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Some S2 responses ship affiliations as list-of-dicts; tool
        # should tolerate either shape.
        _patch_client(
            monkeypatch,
            {
                "/author/search": _Response(
                    json_data={
                        "total": 1,
                        "data": [
                            _author(
                                name="Carol",
                                affiliations=[{"name": "MIT"}],
                            ),
                        ],
                    },
                ),
            },
        )
        tool = pa_mod.PaperAuthor()
        resp = await tool.run(_msg(json_freeze({"query": "carol"})))
        assert "MIT" in _txt(resp)

    @pytest.mark.anyio
    async def test_429(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_client(
            monkeypatch,
            {"/author/search": _Response(status_code=429)},
        )
        tool = pa_mod.PaperAuthor()
        resp = await tool.run(_msg(json_freeze({"query": "x"})))
        assert resp.descriptor == "text/x-error"
        assert "rate limit" in str(resp.content).lower()


# -- Author details --------------------------------------------------------


class TestAuthorDetails:
    @pytest.mark.anyio
    async def test_returns_block(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_client(
            monkeypatch,
            {
                "/author/1741101": _Response(
                    json_data=_author(
                        author_id="1741101",
                        name="Yoshua Bengio",
                        aliases=["Y. Bengio"],
                        affiliations=["Mila", "Université de Montréal"],
                        homepage="https://example.org/bengio",
                        h_index=200,
                        citation_count=500_000,
                        paper_count=800,
                    ),
                ),
            },
        )
        tool = pa_mod.PaperAuthor()
        resp = await tool.run(_msg(json_freeze({"id": "1741101"})))
        assert "author_id: 1741101" in _txt(resp)
        assert "name: Yoshua Bengio" in _txt(resp)
        assert "aliases: Y. Bengio" in _txt(resp)
        assert "Mila, Université de Montréal" in _txt(resp)
        assert "homepage: https://example.org/bengio" in _txt(resp)
        assert "h_index: 200" in _txt(resp)
        assert "citation_count: 500000" in _txt(resp)
        assert "paper_count: 800" in _txt(resp)

    @pytest.mark.anyio
    async def test_404(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_client(
            monkeypatch,
            {"/author/": _Response(status_code=404)},
        )
        tool = pa_mod.PaperAuthor()
        resp = await tool.run(_msg(json_freeze({"id": "nonesuch"})))
        assert resp.descriptor == "text/x-error"
        assert "Not found" in str(resp.content)


# -- Author papers ---------------------------------------------------------


class TestAuthorPapers:
    @pytest.mark.anyio
    async def test_lists(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_client(
            monkeypatch,
            {
                "/author/1/papers": _Response(
                    json_data={
                        "data": [
                            _paper(
                                paper_id="p1",
                                title="Paper One",
                                doi="10.1234/a",
                                year=2022,
                            ),
                            _paper(
                                paper_id="p2",
                                title="Paper Two",
                                doi="10.1234/b",
                                year=2023,
                            ),
                        ],
                    },
                ),
            },
        )
        tool = pa_mod.PaperAuthor()
        resp = await tool.run(_msg(json_freeze({"id": "1", "operation": "papers"})))
        assert "Paper One" in _txt(resp)
        assert "Paper Two" in _txt(resp)
        assert "doi:10.1234/a" in _txt(resp)

    @pytest.mark.anyio
    async def test_year_from(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_client(
            monkeypatch,
            {
                "/author/1/papers": _Response(
                    json_data={
                        "data": [
                            _paper(paper_id="p1", title="Old", year=2015),
                            _paper(paper_id="p2", title="New", year=2023),
                        ],
                    },
                ),
            },
        )
        tool = pa_mod.PaperAuthor()
        resp = await tool.run(
            _msg(json_freeze({"id": "1", "operation": "papers", "year_from": 2020}))
        )
        assert "New" in _txt(resp)
        assert "Old" not in _txt(resp)

    @pytest.mark.anyio
    async def test_year_to(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_client(
            monkeypatch,
            {
                "/author/1/papers": _Response(
                    json_data={
                        "data": [
                            _paper(paper_id="p1", title="Old", year=2015),
                            _paper(paper_id="p2", title="New", year=2023),
                        ],
                    },
                ),
            },
        )
        tool = pa_mod.PaperAuthor()
        resp = await tool.run(
            _msg(json_freeze({"id": "1", "operation": "papers", "year_to": 2020}))
        )
        assert "Old" in _txt(resp)
        assert "New" not in _txt(resp)

    @pytest.mark.anyio
    async def test_abstract_chars(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        long_paper = _paper(paper_id="p1", title="Long")
        long_paper["abstract"] = "q" * 500
        _patch_client(
            monkeypatch,
            {
                "/author/1/papers": _Response(
                    json_data={"data": [long_paper]},
                ),
            },
        )
        tool = pa_mod.PaperAuthor()
        resp = await tool.run(
            _msg(json_freeze({"id": "1", "operation": "papers", "abstract_chars": 30}))
        )
        assert _txt(resp).count("q") == 30

    @pytest.mark.anyio
    async def test_empty_after_filter(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_client(
            monkeypatch,
            {
                "/author/1/papers": _Response(
                    json_data={
                        "data": [_paper(year=2010)],
                    },
                ),
            },
        )
        tool = pa_mod.PaperAuthor()
        resp = await tool.run(
            _msg(json_freeze({"id": "1", "operation": "papers", "year_from": 2020}))
        )
        assert "(no results)" in _txt(resp)


# -- Cache -----------------------------------------------------------------


class TestCache:
    @pytest.mark.anyio
    async def test_search_cache_hit(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock = _patch_client(
            monkeypatch,
            {
                "/author/search": _Response(
                    json_data={"total": 1, "data": [_author(name="Alice")]},
                ),
            },
        )
        tool = pa_mod.PaperAuthor()
        _ = await tool.run(_msg(json_freeze({"query": "alice"})))
        _ = await tool.run(_msg(json_freeze({"query": "alice"})))
        assert len(mock.calls) == 1

    @pytest.mark.anyio
    async def test_different_params_miss(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock = _patch_client(
            monkeypatch,
            {
                "/author/search": _Response(
                    json_data={"total": 1, "data": [_author(name="Alice")]},
                ),
            },
        )
        tool = pa_mod.PaperAuthor()
        _ = await tool.run(_msg(json_freeze({"query": "alice"})))
        _ = await tool.run(_msg(json_freeze({"query": "alice", "limit": 5})))
        assert len(mock.calls) == 2


# -- Describe labels -------------------------------------------------------


class TestDescribe:
    def test_search_label(self) -> None:
        tool = pa_mod.PaperAuthor()
        d = tool.summary(_msg(json_freeze({"query": "Alice"})))
        assert "search" in d
        assert "Alice" in d

    def test_details_label(self) -> None:
        tool = pa_mod.PaperAuthor()
        d = tool.summary(_msg(json_freeze({"id": "1741101"})))
        assert "1741101" in d

    def test_papers_label(self) -> None:
        tool = pa_mod.PaperAuthor()
        d = tool.summary(_msg(json_freeze({"id": "1741101", "operation": "papers"})))
        assert "papers" in d
        assert "1741101" in d


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
