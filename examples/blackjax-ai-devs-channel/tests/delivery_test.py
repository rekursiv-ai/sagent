"""Tests for the audit-log helpers in :mod:`mcp_sagent.delivery`.

The HTTP-bridged ``route_send`` / ``schedule_defer`` are exercised by
the live smoke test (see README § "Verified end-to-end"); unit-testing
them would require a fake HTTP server fixture and would mostly be
testing ``urllib.request`` rather than logic. The audit-log primitives
below ARE process-local and DO have unit-test surface.
"""

from __future__ import annotations

import json

from mcp_sagent import delivery

import pytest


@pytest.fixture
def isolated_log(tmp_path, monkeypatch):
    log_path = tmp_path / "main.jsonl"
    monkeypatch.setattr(delivery, "MAIN_JSONL_PATH", log_path)
    return log_path


def test_append_record_writes_chat_compatible_line(isolated_log):
    delivery.append_record(from_role="tl", to=["swe"], body="hello")
    rec = json.loads(isolated_log.read_text().strip())
    assert rec["from"] == "tl"
    assert rec["to"] == ["swe"]
    assert rec["body"] == "hello"
    assert rec["ts"].endswith("Z")


def test_append_user_message_uses_user_as_from(isolated_log):
    delivery.append_user_message(to_role="tl", body="prompt from operator")
    rec = json.loads(isolated_log.read_text().strip())
    assert rec["from"] == "user"
    assert rec["to"] == ["tl"]


def test_multiple_records_append_in_order(isolated_log):
    delivery.append_record(from_role="tl", to=["swe"], body="one")
    delivery.append_record(from_role="swe", to=["tl"], body="two")
    delivery.append_record(from_role="tl", to=["user"], body="three")
    lines = isolated_log.read_text().splitlines()
    assert len(lines) == 3
    recs = [json.loads(l) for l in lines]
    assert [r["body"] for r in recs] == ["one", "two", "three"]


def test_iso8601_z_format():
    s = delivery.iso8601_z()
    # YYYY-MM-DDTHH:MM:SS.mmmZ — 24 chars.
    assert len(s) == 24
    assert s[4] == "-" and s[10] == "T" and s[19] == "."
    assert s.endswith("Z")


def test_cancel_all_deferred_returns_zero_in_http_bridge_mode():
    # The in-process scheduler has been replaced by the HTTP bridge;
    # cancel_all_deferred is now a stable-API no-op.
    assert delivery.cancel_all_deferred() == 0
