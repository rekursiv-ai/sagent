"""Provider-integration tests for the materializer opt-in.

Verifies the wiring between ``AnthropicCLI.model(materialize_session=...)``
and the on-disk JSONL writer. These don't spawn a real ``claude``
subprocess -- they exercise the materialize-before-spawn hook in
isolation by calling it directly with a hand-built model + request.

The end-to-end spawn behaviour (does ``claude --resume`` actually
consume what we wrote?) is what the Phase 3 tripwire verifies; this
file only proves the plumbing fires when the flag is set.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import os

import pytest

from sagent.providers.anthropic_cli import _AnthropicCLIModel
from sagent.providers.anthropic_cli_session import session_jsonl_path
from sagent.providers.lib.cost import ModelProfile, Pricing
from sagent.types.model import ModelRequest
from sagent.types.runtime import (
    AssistantMessage,
    ModelContextEvent,
    UserMessage,
)


def _stub_profile() -> ModelProfile:
    """Build the minimum ``ModelProfile`` the constructor needs."""
    return ModelProfile(
        max_request_tokens=200_000,
        max_response_tokens=8_000,
        supports_thinking=False,
        pricing=Pricing(),
    )


def _make_model(*, session_id: str, materialize: bool) -> _AnthropicCLIModel:
    """Build a model instance bypassing credential validation."""
    provider = MagicMock(name="AnthropicCLI")
    provider.account = None
    return _AnthropicCLIModel(
        provider=provider,
        model_id="claude-opus-4-8",
        profile=_stub_profile(),
        max_request_tokens=200_000,
        session_id=session_id,
        materialize_session=materialize,
    )


@pytest.fixture
def tmp_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def test_default_is_v2_behaviour() -> None:
    """Without ``materialize_session=True``, the flag stays False.

    Mirrors the proposal's default-off contract: existing v2 callers
    must see zero behaviour change.
    """
    m = _make_model(
        session_id="aaaaaaaa-1111-2222-3333-444444444444",
        materialize=False,
    )
    assert m._materialize_session is False


def test_flag_requires_session_id() -> None:
    """``materialize_session=True`` without ``session_id`` collapses to False.

    Materialization only makes sense in session-persistence mode --
    stateless mode re-feeds history via stdin every spawn, there's no
    on-disk file to overwrite.
    """
    provider = MagicMock(name="AnthropicCLI")
    provider.account = None
    m = _AnthropicCLIModel(
        provider=provider,
        model_id="claude-opus-4-8",
        profile=_stub_profile(),
        max_request_tokens=200_000,
        session_id=None,
        materialize_session=True,
    )
    assert m._materialize_session is False


def test_materialize_prior_state_no_op_on_first_turn(tmp_home: Path) -> None:
    """When ``_last_sent_index == 0`` (first turn), no file is written.

    The first spawn uses ``--session-id`` which creates the file; if
    we materialized an empty file first, ``_session_initialized``
    bookkeeping desyncs from reality.
    """
    sid = "aaaaaaaa-1111-2222-3333-444444444444"
    m = _make_model(session_id=sid, materialize=True)
    assert m._last_sent_index == 0

    req = ModelRequest(
        messages=cast(list[ModelContextEvent], [UserMessage(text="first turn")]),
    )
    m._materialize_prior_state(req)

    expected_path = session_jsonl_path(sid, cwd=Path.cwd(), home=tmp_home)
    assert not expected_path.exists()


def test_materialize_prior_state_writes_when_session_initialized(
    tmp_home: Path,
) -> None:
    """When ``_last_sent_index > 0``, the JSONL gets rewritten from sagent's view.

    The slice that's written is ``messages[:_last_sent_index]`` --
    everything that should already be on disk, NOT the new entries
    being fed via stdin.
    """
    sid = "bbbbbbbb-1111-2222-3333-444444444444"
    m = _make_model(session_id=sid, materialize=True)
    # Simulate two prior turns already on disk.
    m._last_sent_index = 2

    messages = cast(
        list[ModelContextEvent],
        [
            UserMessage(text="turn 1"),
            AssistantMessage(text="response 1"),
            UserMessage(text="turn 2 -- new, fed via stdin"),
        ],
    )
    req = ModelRequest(messages=messages)
    m._materialize_prior_state(req)

    expected_path = session_jsonl_path(sid, cwd=Path.cwd(), home=tmp_home)
    assert expected_path.exists()
    lines = expected_path.read_text().splitlines()
    # Only the first two entries -- the new entry (turn 2) is NOT in
    # the file because stdin will deliver it to claude.
    assert len(lines) == 2


def test_materialize_disabled_skips_write(tmp_home: Path) -> None:
    """With the flag off, the hook is a no-op."""
    sid = "cccccccc-1111-2222-3333-444444444444"
    m = _make_model(session_id=sid, materialize=False)
    m._last_sent_index = 2
    req = ModelRequest(
        messages=cast(
            list[ModelContextEvent],
            [UserMessage(text="a"), AssistantMessage(text="b")],
        ),
    )
    m._materialize_prior_state(req)

    expected_path = session_jsonl_path(sid, cwd=Path.cwd(), home=tmp_home)
    assert not expected_path.exists()


def test_materialize_overwrites_prior_contents(tmp_home: Path) -> None:
    """A second materialization fully replaces the first's bytes.

    The contract is "sagent's tape view becomes the canonical on-disk
    file"; if a torn / leftover write merged in, claude would see
    inconsistent history on ``--resume`` and the chain would break.
    """
    sid = "dddddddd-1111-2222-3333-444444444444"
    m = _make_model(session_id=sid, materialize=True)
    m._last_sent_index = 1

    long_text = "A" * 500
    short_text = "B"
    expected_path = session_jsonl_path(sid, cwd=Path.cwd(), home=tmp_home)

    m._materialize_prior_state(
        ModelRequest(
            messages=cast(list[ModelContextEvent], [UserMessage(text=long_text)]),
        )
    )
    long_bytes = expected_path.read_bytes()

    m._materialize_prior_state(
        ModelRequest(
            messages=cast(list[ModelContextEvent], [UserMessage(text=short_text)]),
        )
    )
    short_bytes = expected_path.read_bytes()
    assert short_bytes != long_bytes
    assert short_text.encode() in short_bytes
    assert b"A" * 500 not in short_bytes


def test_env_var_opt_out_semantics_via_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The plugin's ``common.py`` reads ``SAGENT_CLI_OWN_SESSION``
    with OPT-OUT semantics (default-on since v2.1-β graduated
    2026-06-09).

    A unit-style smoke check that mirrors the parsing in
    ``roles/common.py:build_agent``. Default-on; explicit
    ``0``/``false``/``no`` opts out.
    """

    def materialize_session() -> bool:
        opt_out = os.environ.get("SAGENT_CLI_OWN_SESSION", "").lower() in (
            "0",
            "false",
            "no",
        )
        return not opt_out

    monkeypatch.delenv("SAGENT_CLI_OWN_SESSION", raising=False)
    assert materialize_session() is True, "unset → default-on"
    monkeypatch.setenv("SAGENT_CLI_OWN_SESSION", "")
    assert materialize_session() is True, "empty → default-on"
    monkeypatch.setenv("SAGENT_CLI_OWN_SESSION", "0")
    assert materialize_session() is False, "explicit 0 → opt-out"
    monkeypatch.setenv("SAGENT_CLI_OWN_SESSION", "false")
    assert materialize_session() is False, "false → opt-out"
    monkeypatch.setenv("SAGENT_CLI_OWN_SESSION", "NO")
    assert materialize_session() is False, "NO (case-insensitive) → opt-out"
    monkeypatch.setenv("SAGENT_CLI_OWN_SESSION", "1")
    assert materialize_session() is True, "legacy truthy → default-on (forward-compat)"
    monkeypatch.setenv("SAGENT_CLI_OWN_SESSION", "garbage")
    assert materialize_session() is True, "unknown value → default-on"
