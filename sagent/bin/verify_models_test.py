"""Tests for ``bin.verify_models``: provider limits checking via stubbed HTTP."""

from __future__ import annotations

from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING

import sys

import httpx2
import pytest

from sagent.bin import verify_models
from sagent.providers.anthropic.api import Anthropic
from sagent.providers.google.api import Google
from sagent.providers.openai.api import OpenAI
from sagent.types.capability import ModelCapability, ModelLimits


if TYPE_CHECKING:
    from collections.abc import Callable


def _cap(request: int, response: int) -> ModelCapability:
    """Build a capability carrying only the two limits ``compare`` reads."""
    return ModelCapability(
        context=MappingProxyType(
            {"": ModelLimits(max_request_tokens=request, max_response_tokens=response)},
        ),
    )


def _limits(request: int, response: int) -> verify_models.LiveLimits:
    return verify_models.LiveLimits(
        max_request_tokens=request,
        max_response_tokens=response,
    )


def _serve(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx2.Request], httpx2.Response],
) -> list[httpx2.Request]:
    """Route every ``httpx2.AsyncClient`` through `handler`; return the requests."""
    seen: list[httpx2.Request] = []

    def record(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return handler(request)

    monkeypatch.setattr(
        httpx2,
        "AsyncClient",
        partial(httpx2.AsyncClient, transport=httpx2.MockTransport(record)),
    )
    return seen


class TestParseOpenaiPage:
    def test_parses_limits_with_markup_and_commas(self) -> None:
        html = """
        <!-- hidden
        <b>999,999 context window</b> -->
        <div>128,000 context window</div>
        <span>16,384 max output tokens</span>
        """
        assert verify_models._parse_openai_page(html) == _limits(128_000, 16_384)

    def test_prefers_maximum_input_tokens_over_context_window(self) -> None:
        page = (
            "- 1,050,000 context window\n"
            "- Maximum input tokens: 922,000\n"
            "- 128,000 max output tokens\n"
        )
        assert verify_models._parse_openai_page(page) == _limits(922_000, 128_000)

    def test_returns_none_without_both_limits(self) -> None:
        assert verify_models._parse_openai_page("128,000 context window") is None


class TestFetchGoogle:
    @pytest.mark.anyio
    async def test_fetches_model_limits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        body = {
            "models": [
                {
                    "name": "models/gemini-test",
                    "inputTokenLimit": 10,
                    "outputTokenLimit": 20,
                },
                {"name": "models/incomplete", "inputTokenLimit": 1},
            ],
        }
        _serve(monkeypatch, lambda _: httpx2.Response(200, json=body))
        assert await verify_models.fetch_google("key") == {
            "gemini-test": _limits(10, 20),
        }

    @pytest.mark.anyio
    async def test_follows_next_page_token(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def handler(request: httpx2.Request) -> httpx2.Response:
            if request.url.params.get("pageToken") == "p2":
                model = {
                    "name": "models/b",
                    "inputTokenLimit": 3,
                    "outputTokenLimit": 4,
                }
                return httpx2.Response(200, json={"models": [model]})
            model = {"name": "models/a", "inputTokenLimit": 1, "outputTokenLimit": 2}
            return httpx2.Response(
                200,
                json={"models": [model], "nextPageToken": "p2"},
            )

        _serve(monkeypatch, handler)
        assert await verify_models.fetch_google("key") == {
            "a": _limits(1, 2),
            "b": _limits(3, 4),
        }

    @pytest.mark.anyio
    async def test_key_rides_a_header_never_the_url(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        seen = _serve(monkeypatch, lambda _: httpx2.Response(403))
        assert await verify_models.fetch_google("s3cr3t-key") == {}
        assert seen
        assert all("s3cr3t-key" not in str(request.url) for request in seen)
        assert seen[0].headers["x-goog-api-key"] == "s3cr3t-key"
        out = capsys.readouterr().out
        assert "HTTP 403" in out
        assert "s3cr3t-key" not in out

    @pytest.mark.anyio
    async def test_network_error_warns_instead_of_raising(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        def refuse(request: httpx2.Request) -> httpx2.Response:
            raise httpx2.ConnectError(f"refused {request.url.host}")

        _serve(monkeypatch, refuse)
        assert await verify_models.fetch_google("key") == {}
        assert "ConnectError" in capsys.readouterr().out


class TestFetchOpenai:
    @pytest.mark.anyio
    async def test_fetches_parseable_pages(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        pages = {
            "ok": httpx2.Response(
                200,
                text="1,000 context window 200 max output tokens",
            ),
            "bad": httpx2.Response(200, text="not parseable"),
            "missing": httpx2.Response(404),
        }

        def handler(request: httpx2.Request) -> httpx2.Response:
            if request.url.path.endswith("/down.md"):
                raise httpx2.ConnectError("refused")
            return pages[request.url.path.rsplit("/", 1)[1].removesuffix(".md")]

        seen = _serve(monkeypatch, handler)
        limits = await verify_models.fetch_openai(["ok", "bad", "missing", "down"])
        assert limits == {"ok": _limits(1000, 200)}
        # The markdown rendering states "Maximum input tokens"; the HTML omits it.
        assert all(request.url.path.endswith(".md") for request in seen)
        out = capsys.readouterr().out
        assert "could not parse" in out
        assert "HTTP 404" in out
        assert "ConnectError" in out


class TestFetchAnthropic:
    @pytest.mark.anyio
    async def test_fetches_and_warns(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        bodies: dict[str, httpx2.Response] = {
            "ok": httpx2.Response(
                200,
                json={"max_input_tokens": 100, "max_tokens": 10},
            ),
            "bad": httpx2.Response(200, json={"max_input_tokens": 0, "max_tokens": 10}),
            "err": httpx2.Response(500),
            "missing": httpx2.Response(404),
        }

        def handler(request: httpx2.Request) -> httpx2.Response:
            if request.url.path.endswith("/down"):
                raise httpx2.ConnectError("refused")
            return bodies[request.url.path.rsplit("/", 1)[1]]

        _serve(monkeypatch, handler)
        limits = await verify_models.fetch_anthropic(
            "key",
            ["ok", "bad", "err", "missing", "down"],
        )
        assert limits == {"ok": _limits(100, 10)}
        out = capsys.readouterr().out
        assert "missing limits" in out
        assert "HTTP 500" in out
        assert "not found in API" in out
        assert "ConnectError" in out


class TestCanonicalModels:
    def test_keys_rows_by_vendor_wire_id(self) -> None:
        known = verify_models._canonical_models(Anthropic.catalog)
        assert "claude-opus-5-5" in known
        assert "opus-5.5" not in known
        assert known["claude-opus-5-5"].model_id == "opus-5.5"

    def test_falls_back_to_model_id_without_a_wire_id(self) -> None:
        known = verify_models._canonical_models(OpenAI.catalog)
        assert all(cap.wire_model_id in {mid, ""} for mid, cap in known.items())
        assert len(known) == len(OpenAI.catalog.models)

    @pytest.mark.anyio
    async def test_anthropic_is_queried_by_wire_id(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        seen = _serve(monkeypatch, lambda _: httpx2.Response(404))
        known = verify_models._canonical_models(Anthropic.catalog)
        await verify_models.fetch_anthropic("key", list(known))
        paths = {request.url.path for request in seen}
        assert "/v1/models/claude-opus-5-5" in paths
        assert "/v1/models/opus-5.5" not in paths


class TestCompare:
    def test_reports_unknown_and_mismatches(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        known = {
            "same": _cap(10, 20),
            "diff": _cap(1, 2),
            "dead": _cap(1, 2),
        }
        live = {
            "same": _limits(10, 20),
            "diff": _limits(3, 4),
            "new": _limits(5, 6),
        }
        assert verify_models.compare("Provider", known, live) == 3
        out = capsys.readouterr().out
        assert "Provider.new: in API but not in the catalog" in out
        assert "Provider.dead: not verified" in out
        assert "max_request_tokens" in out
        assert "max_response_tokens" in out
        assert "2/3 models verified" in out

    def test_reports_all_ok(self, capsys: pytest.CaptureFixture[str]) -> None:
        live = {"m": _limits(1, 2)}
        assert verify_models.compare("Provider", {"m": _cap(1, 2)}, live) == 0
        assert "all 1 models OK" in capsys.readouterr().out

    def test_zero_coverage_is_an_error(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        assert verify_models.compare("P", {"m": _cap(1, 2)}, {}) == 1
        out = capsys.readouterr().out
        assert "all 1 models OK" not in out
        assert "0/1 models verified" in out


def _exact(known: dict[str, ModelCapability]) -> dict[str, verify_models.LiveLimits]:
    """Live limits that agree with every catalog row."""
    return {
        mid: _limits(
            cap.context[""].max_request_tokens,
            cap.context[""].max_response_tokens,
        )
        for mid, cap in known.items()
    }


class TestRun:
    @pytest.mark.anyio
    async def test_default_runs_every_provider_and_sums_problems(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["verify"])
        monkeypatch.setenv("GOOGLE_API_KEY", "gkey")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "akey")
        google = verify_models._canonical_models(Google.catalog)
        openai = verify_models._canonical_models(OpenAI.catalog)
        anthropic = verify_models._canonical_models(Anthropic.catalog)
        wrong = next(iter(openai))

        async def fake_google(key: str) -> dict[str, verify_models.LiveLimits]:
            assert key == "gkey"
            return {**_exact(google), "zz-new": _limits(1, 1)}

        async def fake_openai(ids: list[str]) -> dict[str, verify_models.LiveLimits]:
            assert ids == list(openai)
            return {**_exact(openai), wrong: _limits(1, 1)}

        async def fake_anthropic(
            key: str,
            ids: list[str],
        ) -> dict[str, verify_models.LiveLimits]:
            assert key == "akey"
            assert ids == list(anthropic)
            return {**_exact(anthropic), "zz-new": _limits(1, 1)}

        monkeypatch.setattr(verify_models, "fetch_google", fake_google)
        monkeypatch.setattr(verify_models, "fetch_openai", fake_openai)
        monkeypatch.setattr(verify_models, "fetch_anthropic", fake_anthropic)
        monkeypatch.setattr(verify_models, "audit_catalogs", lambda: 1)
        assert await verify_models._run() == 1
        lines = capsys.readouterr().out.splitlines()
        assert lines[0] == "Catalog audit (offline):"
        for line in (
            "Google (API query):",
            "  Google.zz-new: in API but not in the catalog",
            "OpenAI (doc scrape):",
            f"  OpenAI.{wrong}: max_request_tokens",
            "Anthropic (API query):",
            "  Anthropic.zz-new: in API but not in the catalog",
        ):
            assert any(got.startswith(line) for got in lines), line
        assert lines[-1] == "5 problem(s) found."

    @pytest.mark.anyio
    async def test_missing_keys_each_count_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["verify", "--provider", "all"])
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        openai = verify_models._canonical_models(OpenAI.catalog)

        async def fake_openai(ids: list[str]) -> dict[str, verify_models.LiveLimits]:
            del ids
            return _exact(openai)

        monkeypatch.setattr(verify_models, "fetch_openai", fake_openai)
        monkeypatch.setattr(verify_models, "audit_catalogs", lambda: 1)
        assert await verify_models._run() == 1
        lines = capsys.readouterr().out.splitlines()
        assert "  [error] GOOGLE_API_KEY not set" in lines
        assert "  [error] ANTHROPIC_API_KEY not set" in lines
        assert lines[-1] == "3 problem(s) found."

    @pytest.mark.anyio
    async def test_offline_skips_the_network(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["verify", "--offline"])
        monkeypatch.delattr(verify_models, "fetch_openai")
        assert await verify_models._run() == 0
        assert capsys.readouterr().out.splitlines() == [
            "Catalog audit (offline):",
            "  catalogs: all rows OK",
        ]

    @pytest.mark.anyio
    async def test_offline_reports_audit_problems(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["verify", "--offline"])
        monkeypatch.setattr(verify_models, "audit_catalogs", lambda: 2)
        assert await verify_models._run() == 1
        assert capsys.readouterr().out.splitlines()[-1] == "2 problem(s) found."

    @pytest.mark.anyio
    async def test_help_shows_the_docstring_body(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["verify", "--help"])
        with pytest.raises(SystemExit) as exit_info:
            await verify_models._run()
        assert exit_info.value.code == 0
        out = capsys.readouterr().out
        assert "\nVerify provider model limits.\n" in out
        assert "\nSources:\n" in out
        assert "exec uv" not in out
        assert "{google,openai,anthropic,all}" in out

    @pytest.mark.anyio
    async def test_rejects_an_unknown_provider(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["verify", "--provider", "bogus"])
        with pytest.raises(SystemExit) as exit_info:
            await verify_models._run()
        assert exit_info.value.code == 2


class TestMain:
    @pytest.mark.anyio
    async def test_missing_api_key_fails_the_run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["verify", "--provider", "google"])
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        assert await verify_models._run() == 1
        out = capsys.readouterr().out
        assert "GOOGLE_API_KEY not set" in out
        assert "All limits verified" not in out

    @pytest.mark.anyio
    async def test_returns_nonzero_on_errors(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["verify", "--provider", "openai"])

        async def fake_fetch_openai(
            ids: list[str],
        ) -> dict[str, verify_models.LiveLimits]:
            return {mid: _limits(1, 1) for mid in ids}

        monkeypatch.setattr(verify_models, "fetch_openai", fake_fetch_openai)
        assert await verify_models._run() == 1

    @pytest.mark.anyio
    async def test_matching_limits_pass(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["verify", "--provider", "openai"])
        known = verify_models._canonical_models(OpenAI.catalog)

        async def fake_fetch_openai(
            ids: list[str],
        ) -> dict[str, verify_models.LiveLimits]:
            return {
                mid: _limits(
                    known[mid].context[""].max_request_tokens,
                    known[mid].context[""].max_response_tokens,
                )
                for mid in ids
            }

        monkeypatch.setattr(verify_models, "fetch_openai", fake_fetch_openai)
        assert await verify_models._run() == 0
        assert capsys.readouterr().out.splitlines()[-1] == "All limits verified."

    @pytest.mark.anyio
    async def test_google_with_no_live_rows_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["verify", "--provider", "google"])
        monkeypatch.setenv("GOOGLE_API_KEY", "x")

        async def fake_fetch_google(
            _: str,
        ) -> dict[str, verify_models.LiveLimits]:
            return {}

        monkeypatch.setattr(verify_models, "fetch_google", fake_fetch_google)
        assert await verify_models._run() == 1
        assert "All limits verified" not in capsys.readouterr().out

    @pytest.mark.anyio
    async def test_missing_anthropic_key_fails_the_run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["verify", "--provider", "anthropic"])
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert await verify_models._run() == 1
        assert "ANTHROPIC_API_KEY not set" in capsys.readouterr().out

    @pytest.mark.anyio
    async def test_runs_anthropic_by_wire_id_when_key_set(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["verify", "--provider", "anthropic"])
        monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
        queried: list[str] = []

        async def fake_fetch_anthropic(
            key: str,
            ids: list[str],
        ) -> dict[str, verify_models.LiveLimits]:
            del key
            queried.extend(ids)
            return {}

        monkeypatch.setattr(verify_models, "fetch_anthropic", fake_fetch_anthropic)
        assert await verify_models._run() == 1
        assert "claude-opus-5-5" in queried


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
