"""Tests for tools.paper_fetch. All HTTP calls are mocked."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import json

import pytest

from sagent.custom_types import (
    JsonMessage,
    Message,
    MultipartMessage,
    TextMessage,
)
from sagent.lib.json import JSON, JSONValue, json_freeze
from sagent.lib.web.fetch import FetchError, HTTPConn
from sagent.tools import paper_fetch as pf_mod


def _msg(directive: JSON) -> Message:
    return MultipartMessage(
        (JsonMessage(directive, "application/x-tool-paperfetch"),),
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


_PDF_HEADER = b"%PDF-1.5\n" + b"x" * 200
_HTML_HEADER = b"<html><body>not a pdf</body></html>"


class _FetchRouter:
    """Route fetch() calls by URL pattern. Tracks calls for assertions."""

    def __init__(
        self,
        gets: dict[str, bytes | Exception] | None = None,
        posts: dict[str, bytes | Exception] | None = None,
    ) -> None:
        self._gets = gets or {}
        self._posts = posts or {}
        self.calls: list[tuple[str, str]] = []

    def __call__(
        self,
        url: str,
        *,
        method: str = "GET",
        data: dict[str, str] | None = None,
        json: JSONValue = None,
        headers: dict[str, str] | None = None,
        cookies: dict[str, str] | None = None,
        retries: int = 0,
        timeout_sec: float = 30,
        max_redirects: int = 10,
        on_redirect: Callable[[str], None] | None = None,
        return_connection: bool = False,
        http_conn: HTTPConn | None = None,
    ) -> bytes:
        del data, json, headers, cookies, retries, timeout_sec
        del max_redirects, on_redirect, return_connection, http_conn
        self.calls.append((method, url))
        table = self._posts if method == "POST" else self._gets
        for pattern, val in table.items():
            if pattern in url:
                if isinstance(val, Exception):
                    raise val
                return val
        raise FetchError(url, 404, {}, b"not found")


def _patch_fetch(
    monkeypatch: pytest.MonkeyPatch,
    *,
    gets: dict[str, bytes | Exception] | None = None,
    posts: dict[str, bytes | Exception] | None = None,
) -> _FetchRouter:
    router = _FetchRouter(gets, posts)
    monkeypatch.setattr("sagent.tools.paper_fetch.fetch", router)
    return router


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "papers"


class TestValidation:
    @pytest.mark.anyio
    async def test_missing_id(self, cache_dir: Path) -> None:
        tool = pf_mod.PaperFetch(cache_dir=cache_dir)
        resp = await tool.run(_msg(json_freeze({"id": ""})))
        assert resp.descriptor == "text/x-error"
        assert "required" in str(resp.content)

    @pytest.mark.anyio
    async def test_bad_id(self, cache_dir: Path) -> None:
        tool = pf_mod.PaperFetch(cache_dir=cache_dir)
        resp = await tool.run(_msg(json_freeze({"id": "not-an-id"})))
        assert resp.descriptor == "text/x-error"
        assert "Unrecognized" in str(resp.content)


class TestArxivPath:
    @pytest.mark.anyio
    async def test_direct_download(
        self,
        monkeypatch: pytest.MonkeyPatch,
        cache_dir: Path,
    ) -> None:
        router = _patch_fetch(
            monkeypatch,
            gets={"arxiv.org/pdf/2106.15928": _PDF_HEADER},
        )
        tool = pf_mod.PaperFetch(cache_dir=cache_dir)
        resp = await tool.run(_msg(json_freeze({"id": "arXiv:2106.15928"})))
        assert "arxiv" in _txt(resp).lower()
        path = cache_dir / "arxiv_2106.15928.pdf"
        assert path.exists()
        assert path.read_bytes()[:5] == b"%PDF-"
        assert any("arxiv.org/pdf" in call[1] for call in router.calls)
        assert not any("semanticscholar.org" in call[1] for call in router.calls)

    @pytest.mark.anyio
    async def test_falls_through_on_non_pdf(
        self,
        monkeypatch: pytest.MonkeyPatch,
        cache_dir: Path,
    ) -> None:
        s2_resp = json.dumps({"openAccessPdf": None}).encode()
        _patch_fetch(
            monkeypatch,
            gets={
                "arxiv.org/pdf": _HTML_HEADER,
                "semanticscholar.org": s2_resp,
            },
        )
        tool = pf_mod.PaperFetch(cache_dir=cache_dir)
        resp = await tool.run(_msg(json_freeze({"id": "arXiv:2106.15928"})))
        assert resp.descriptor == "text/x-error"
        assert "No source returned a PDF" in str(resp.content)


class TestOpenAccessPath:
    @pytest.mark.anyio
    async def test_via_s2_oa_url(
        self,
        monkeypatch: pytest.MonkeyPatch,
        cache_dir: Path,
    ) -> None:
        s2_resp = json.dumps(
            {"openAccessPdf": {"url": "https://oa.example.com/p.pdf"}},
        ).encode()
        router = _patch_fetch(
            monkeypatch,
            gets={
                "semanticscholar.org": s2_resp,
                "oa.example.com": _PDF_HEADER,
            },
        )
        tool = pf_mod.PaperFetch(cache_dir=cache_dir)
        resp = await tool.run(_msg(json_freeze({"id": "10.1234/oa"})))
        assert "open_access" in _txt(resp)

        del router

    @pytest.mark.parametrize(
        "s2_resp",
        [
            b"not json",
            b"[]",
            json.dumps({"openAccessPdf": []}).encode(),
        ],
    )
    @pytest.mark.anyio
    async def test_malformed_s2_oa_response_is_external_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        cache_dir: Path,
        s2_resp: bytes,
    ) -> None:
        _patch_fetch(monkeypatch, gets={"semanticscholar.org": s2_resp})
        tool = pf_mod.PaperFetch(cache_dir=cache_dir)

        resp = await tool.run(_msg(json_freeze({"id": "10.1234/malformed"})))

        assert resp.descriptor == "text/x-error"
        assert "No source returned a PDF" in str(resp.content)


class TestPdfValidation:
    def test_looks_like_pdf_true(self) -> None:
        assert pf_mod._looks_like_pdf(_PDF_HEADER)

    def test_looks_like_pdf_false_html(self) -> None:
        assert not pf_mod._looks_like_pdf(_HTML_HEADER)

    def test_looks_like_pdf_false_too_small(self) -> None:
        assert not pf_mod._looks_like_pdf(b"%PDF-")


class TestCache:
    @pytest.mark.anyio
    async def test_cached_path_returned_directly(
        self,
        monkeypatch: pytest.MonkeyPatch,
        cache_dir: Path,
    ) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240
        cached = cache_dir / "arxiv_2106.15928.pdf"
        cached.write_bytes(_PDF_HEADER)

        router = _patch_fetch(monkeypatch, gets={})
        tool = pf_mod.PaperFetch(cache_dir=cache_dir)
        resp = await tool.run(_msg(json_freeze({"id": "arXiv:2106.15928"})))
        assert "Cached" in _txt(resp)
        assert str(cached) in _txt(resp)
        assert not router.calls


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
