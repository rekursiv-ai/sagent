"""Unit tests for the User-Agent pools."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, cast
from unittest.mock import patch

import gzip
import json

import pytest

from sagent.lib.web.chrome import useragents
from sagent.lib.web.chrome.useragents import (
    UserAgentKind,
    draw_user_agent,
    impersonate_target,
    user_agent_pool,
)
from sagent.lib.web.fetch import FetchSession


@pytest.fixture(autouse=True)
def clear_pool_cache() -> Any:
    """Isolate the module-global ``@cache``: a ``refresh`` test clears it and a
    ``_pool_path`` patch can seed tmp content, so reset it around every test to
    keep that state from leaking into other modules under xdist.
    """
    user_agent_pool.cache_clear()
    yield
    user_agent_pool.cache_clear()


class TestPools:
    def test_desktop_pool_is_nonempty_desktop_chrome(self) -> None:
        pool = user_agent_pool("chrome_desktop")
        assert pool
        ua = pool[0]
        assert "Chrome" in ua
        assert "Mobile" not in ua  # desktop

    def test_android_pool_is_nonempty_mobile_chrome(self) -> None:
        pool = user_agent_pool("chrome_android")
        assert pool
        ua = pool[0]
        assert "Chrome" in ua
        assert "Android" in ua

    def test_pools_are_distinct(self) -> None:
        assert set(user_agent_pool("chrome_desktop")).isdisjoint(
            user_agent_pool("chrome_android")
        )


class TestDraw:
    def test_draws_from_the_requested_pool(self) -> None:
        assert draw_user_agent("chrome_desktop") in user_agent_pool("chrome_desktop")
        assert draw_user_agent("chrome_android") in user_agent_pool("chrome_android")

    def test_draws_vary(self) -> None:
        assert len({draw_user_agent("chrome_desktop") for _ in range(200)}) > 1


class TestImpersonateTarget:
    def test_desktop_maps_to_chrome(self) -> None:
        assert impersonate_target("chrome_desktop") == "chrome"

    def test_android_maps_to_chrome_android(self) -> None:
        assert impersonate_target("chrome_android") == "chrome_android"

    def test_kind_for_impersonate_is_the_inverse(self) -> None:
        # The impersonate<->kind bijection has ONE source of truth: the inverse
        # must round-trip both kinds, so fetch.py can call it instead of
        # inlining a parallel (drift-prone) mapping.
        kind_for_impersonate = useragents.kind_for_impersonate
        for kind in ("chrome_desktop", "chrome_android"):
            assert kind_for_impersonate(impersonate_target(kind)) == kind
        # An unknown impersonate target degrades to desktop.
        assert kind_for_impersonate("chrome") == "chrome_desktop"


class TestRefresh:
    """``refresh`` rewrites a kind's pool file from the intoli dataset, applying
    that kind's filter. The upstream download is mocked; the parsing/filtering
    and file write are exercised for real against a tmp pool path.
    """

    _DATASET: ClassVar[list[dict[str, str]]] = [
        {  # desktop Chrome -- kept by desktop, dropped by android
            "userAgent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36",
            "deviceCategory": "desktop",
        },
        {  # android Chrome -- kept by android, dropped by desktop
            "userAgent": "Mozilla/5.0 (Linux; Android 14; Pixel 8) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 "
            "Mobile Safari/537.36",
            "deviceCategory": "mobile",
        },
        {  # desktop Edge -- dropped by both (not plain Chrome)
            "userAgent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 "
            "Safari/537.36 Edg/149.0.0.0",
            "deviceCategory": "desktop",
        },
    ]

    def _refresh(self, kind: str, tmp_path: Path) -> list[str]:
        pool_file = tmp_path / f"{kind}.txt"
        payload = gzip.compress(json.dumps(self._DATASET).encode())
        with (
            patch("sagent.lib.web.fetch.fetch", return_value=(payload, FetchSession())),
            patch.object(useragents, "_pool_path", return_value=pool_file),
        ):
            useragents.refresh(cast("UserAgentKind", kind))
        return pool_file.read_text().splitlines()

    def test_desktop_filter_keeps_only_desktop_plain_chrome(
        self, tmp_path: Path
    ) -> None:
        lines = self._refresh("chrome_desktop", tmp_path)
        assert len(lines) == 1
        assert "Windows NT" in lines[0]
        assert "Mobile" not in lines[0]
        assert "Edg/" not in lines[0]

    def test_android_filter_keeps_only_android_chrome(self, tmp_path: Path) -> None:
        lines = self._refresh("chrome_android", tmp_path)
        assert len(lines) == 1
        assert "Android" in lines[0]

    def test_empty_result_raises(self, tmp_path: Path) -> None:
        pool_file = tmp_path / "chrome_desktop.txt"
        with (
            patch(
                "sagent.lib.web.fetch.fetch",
                return_value=(gzip.compress(json.dumps([]).encode()), FetchSession()),
            ),
            patch.object(useragents, "_pool_path", return_value=pool_file),
            pytest.raises(RuntimeError, match="0 user agents"),
        ):
            useragents.refresh("chrome_desktop")


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
