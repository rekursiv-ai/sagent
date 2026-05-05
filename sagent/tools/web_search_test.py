"""Tests for WebSearch tool."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from sagent.custom_types import (
    JsonMessage,
    Message,
    MultipartMessage,
    TextMessage,
)
from sagent.lib.json import JSON, json_freeze
from sagent.lib.web.search import SearchResult
from sagent.tools.web_search import WebSearch


websearch = WebSearch()

_DDG = "sagent.lib.web.search.duckduckgo"


def _msg(directive: JSON) -> Message:
    return MultipartMessage(
        (JsonMessage(directive, "application/x-tool-websearch"),),
        "multipart/x-tool-call",
    )


def _ddg_directive(extra: dict[str, object] | None = None) -> JSON:
    d: dict[str, object] = {"query": "test", "backend": "duckduckgo"}
    if extra:
        d.update(extra)
    return json_freeze(d)


class TestWebsearch:
    @pytest.mark.anyio
    async def test_success(self) -> None:
        result = SearchResult(
            title="Result",
            url="https://example.com",
            snippet="Desc.",
        )
        with patch(_DDG, return_value=[result]):
            response = await websearch.run(_msg(_ddg_directive()))
        assert isinstance(response, TextMessage)
        assert "Result" in response.content
        assert "example.com" in response.content

    @pytest.mark.anyio
    async def test_no_results(self) -> None:
        with patch(_DDG, return_value=[]):
            response = await websearch.run(_msg(_ddg_directive()))
        assert isinstance(response, TextMessage)
        assert "no results" in response.content.lower()

    @pytest.mark.anyio
    async def test_backend_runtime_error_returns_tool_error(self) -> None:
        with patch(_DDG, side_effect=RuntimeError("boom")):
            response = await websearch.run(_msg(_ddg_directive()))
        assert isinstance(response, TextMessage)
        assert response.descriptor == "text/x-error"
        assert "boom" in response.content

    @pytest.mark.anyio
    async def test_invalid_backend_returns_tool_error(self) -> None:
        with patch(_DDG) as search:
            response = await websearch.run(
                _msg(json_freeze({"query": "test", "backend": "bad"}))
            )
        assert isinstance(response, TextMessage)
        assert response.descriptor == "text/x-error"
        assert "Invalid backend 'bad'" in response.content
        search.assert_not_called()

    @pytest.mark.anyio
    async def test_blocked_domains(self) -> None:
        with patch(_DDG, return_value=[]) as mock:
            await websearch.run(_msg(_ddg_directive({"blocked_domains": ["bad.com"]})))
        mock.assert_called_once_with("test -site:bad.com", 10, None)

    @pytest.mark.anyio
    async def test_allowed_domains(self) -> None:
        with patch(_DDG, return_value=[]) as mock:
            await websearch.run(
                _msg(_ddg_directive({"allowed_domains": ["example.com"]}))
            )
        mock.assert_called_once_with("test site:example.com", 10, None)


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
