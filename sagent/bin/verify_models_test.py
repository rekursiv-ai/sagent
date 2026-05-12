"""Tests for ``bin.verify_models``: provider limits checking via stubbed HTTP."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Self

import sys

import httpx
import pytest

from sagent.bin import verify_models


class _Response:
    def __init__(
        self,
        json_body: dict[str, object] | None = None,
        text: str = "",
        status_code: int = 200,
    ) -> None:
        self._json_body = json_body or {}
        self.text = text
        self.status_code = status_code
        self.request = httpx.Request("GET", "https://example.test")

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self) -> dict[str, object]:
        return self._json_body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            response = httpx.Response(
                self.status_code,
                request=self.request,
            )
            raise httpx.HTTPStatusError("boom", request=self.request, response=response)


class _Client:
    def __init__(self, responses: list[_Response]) -> None:
        self._responses = responses

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def get(self, _url: str, **_kw: object) -> _Response:
        return self._responses.pop(0)


def _make_client(responses: list[_Response]) -> type:
    class _Factory:
        def __new__(cls, **_kw: object) -> _Client:
            return _Client(responses)

    return _Factory


class TestParseOpenaiPage:
    def test_parses_limits_with_markup_and_commas(self) -> None:
        html = """
        <!-- hidden -->
        <div>128,000 context window</div>
        <span>16,384 max output tokens</span>
        """
        limits = verify_models._parse_openai_page(html)
        assert limits == verify_models.ModelLimits(
            max_request_tokens=128_000,
            max_response_tokens=16_384,
        )

    def test_returns_none_without_both_limits(self) -> None:
        assert verify_models._parse_openai_page("128,000 context window") is None


class TestFetchGoogle:
    @pytest.mark.anyio
    async def test_fetches_model_limits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        response = _Response(
            {
                "models": [
                    {
                        "name": "models/gemini-test",
                        "inputTokenLimit": 10,
                        "outputTokenLimit": 20,
                    },
                    {"name": "models/incomplete", "inputTokenLimit": 1},
                ],
            }
        )
        monkeypatch.setattr(httpx, "AsyncClient", _make_client([response]))
        assert await verify_models.fetch_google("key") == {
            "gemini-test": verify_models.ModelLimits(
                max_request_tokens=10,
                max_response_tokens=20,
            )
        }


class TestFetchOpenai:
    @pytest.mark.anyio
    async def test_fetches_parseable_pages(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        responses = [
            _Response(text="1,000 context window 200 max output tokens"),
            _Response(text="not parseable"),
            _Response(status_code=404),
        ]
        monkeypatch.setattr(httpx, "AsyncClient", _make_client(responses))
        limits = await verify_models.fetch_openai(["ok", "bad", "missing"])
        assert limits == {
            "ok": verify_models.ModelLimits(
                max_request_tokens=1000,
                max_response_tokens=200,
            )
        }
        out = capsys.readouterr().out
        assert "could not parse" in out
        assert "HTTP 404" in out


class TestFetchAnthropic:
    @pytest.mark.anyio
    async def test_fetches_and_warns(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        responses = [
            _Response({"max_input_tokens": 100, "max_tokens": 10}),
            _Response({"max_input_tokens": 0, "max_tokens": 10}),
            _Response(status_code=500),
        ]
        monkeypatch.setattr(httpx, "AsyncClient", _make_client(responses))
        limits = await verify_models.fetch_anthropic("key", ["ok", "bad", "err"])
        assert limits == {
            "ok": verify_models.ModelLimits(
                max_request_tokens=100,
                max_response_tokens=10,
            )
        }
        out = capsys.readouterr().out
        assert "missing limits" in out
        assert "HTTP 500" in out

    @pytest.mark.anyio
    async def test_404_reports_not_found(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            httpx, "AsyncClient", _make_client([_Response(status_code=404)])
        )
        limits = await verify_models.fetch_anthropic("key", ["missing"])
        assert limits == {}
        assert "not found in API" in capsys.readouterr().out


class TestCompare:
    def test_reports_unknown_and_mismatches(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        known = {
            "same": SimpleNamespace(max_request_tokens=10, max_response_tokens=20),
            "diff": SimpleNamespace(max_request_tokens=1, max_response_tokens=2),
            "dead": SimpleNamespace(max_request_tokens=1, max_response_tokens=2),
        }
        live = {
            "same": verify_models.ModelLimits(
                max_request_tokens=10,
                max_response_tokens=20,
            ),
            "diff": verify_models.ModelLimits(
                max_request_tokens=3,
                max_response_tokens=4,
            ),
            "new": verify_models.ModelLimits(
                max_request_tokens=5,
                max_response_tokens=6,
            ),
        }
        assert verify_models.compare("Provider", known, live) == 3
        out = capsys.readouterr().out
        assert "in API but not in KNOWN_MODELS" in out
        assert "max_request_tokens" in out
        assert "max_response_tokens" in out

    def test_reports_all_ok(self, capsys: pytest.CaptureFixture[str]) -> None:
        known = {"m": SimpleNamespace(max_request_tokens=1, max_response_tokens=2)}
        live = {
            "m": verify_models.ModelLimits(max_request_tokens=1, max_response_tokens=2)
        }
        assert verify_models.compare("Provider", known, live) == 0
        assert "all 1 models OK" in capsys.readouterr().out


class TestMain:
    @pytest.mark.anyio
    async def test_skips_missing_api_keys(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["verify", "--provider", "google"])
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        assert await verify_models.main() == 0
        assert "GOOGLE_API_KEY not set" in capsys.readouterr().out

    @pytest.mark.anyio
    async def test_returns_nonzero_on_errors(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["verify", "--provider", "openai"])

        async def fake_fetch_openai(
            _: list[str],
        ) -> dict[str, verify_models.ModelLimits]:
            return {
                "extra": verify_models.ModelLimits(
                    max_request_tokens=1,
                    max_response_tokens=1,
                )
            }

        monkeypatch.setattr(verify_models, "fetch_openai", fake_fetch_openai)
        assert await verify_models.main() == 1

    @pytest.mark.anyio
    async def test_runs_google_when_key_set(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["verify", "--provider", "google"])
        monkeypatch.setenv("GOOGLE_API_KEY", "x")

        async def fake_fetch_google(
            _: str,
        ) -> dict[str, verify_models.ModelLimits]:
            return {}

        monkeypatch.setattr(verify_models, "fetch_google", fake_fetch_google)
        assert await verify_models.main() == 0
        assert "Google" in capsys.readouterr().out

    @pytest.mark.anyio
    async def test_skips_missing_anthropic_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["verify", "--provider", "anthropic"])
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert await verify_models.main() == 0
        assert "ANTHROPIC_API_KEY not set" in capsys.readouterr().out

    @pytest.mark.anyio
    async def test_runs_anthropic_when_key_set(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["verify", "--provider", "anthropic"])
        monkeypatch.setenv("ANTHROPIC_API_KEY", "x")

        async def fake_fetch_anthropic(
            _key: str,
            _ids: list[str],
        ) -> dict[str, verify_models.ModelLimits]:
            return {}

        monkeypatch.setattr(verify_models, "fetch_anthropic", fake_fetch_anthropic)
        assert await verify_models.main() == 0
        assert "Anthropic" in capsys.readouterr().out


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
