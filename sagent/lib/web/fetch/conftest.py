"""Hermetic fixtures for fetch package tests."""

from typing import Any

import importlib

import pytest

from sagent.lib.web.fetch.test_helpers import StubSession
from sagent.lib.web.profile import ProfileStore

import sagent.lib.web.fetch.curl as curl_mod


fetch_mod = importlib.import_module("sagent.lib.web.fetch.fetch")


@pytest.fixture(autouse=True)
def isolate_profiles(tmp_path: Any, monkeypatch: Any) -> Any:
    """Hermetic identity layer: a tmp store and a fixed egress, no network.

    ``fetch`` transparently loads/saves a per-``(egress_ip, domain)`` profile.
    Without isolation these tests would share the real on-disk store (state
    leaking between cases) and call the live egress echo. Point the store at a
    tmp dir and pin the egress so the transport assertions stay deterministic.
    """

    def fixed_egress(*, cache: bool = True, **_kw: Any) -> str:
        del cache, _kw
        return "203.0.113.1"

    store = ProfileStore(base_dir=tmp_path)
    monkeypatch.setattr(ProfileStore, "shared", classmethod(lambda _cls: store))
    monkeypatch.setattr(fetch_mod, "egress_ip", fixed_egress)
    monkeypatch.setattr(fetch_mod, "_last_egress_ip", None)

    def stub_session(egress: str, domain: str, impersonate: str) -> StubSession:
        del egress, domain, impersonate
        return StubSession()

    monkeypatch.setattr(fetch_mod, "curl_session", stub_session)
    monkeypatch.setattr(curl_mod, "_curl_sessions", {})
    return
