"""Tests for the serve.py startup tripwire gate (default-on since v2.1-β
graduated 2026-06-09).

The gate function is ``serve._run_materializer_tripwire`` — it reads
``SAGENT_CLI_OWN_SESSION`` with OPT-OUT semantics: the materializer
fires unless the env is explicitly set to ``0`` / ``false`` / ``no``.
Tripwire drift / canary unavailable / canary raises → sets env to ``0``
so the boot falls back to v2 CLI-owned mode.

These tests mock the canary so they don't spawn a real ``claude``
subprocess. The canary itself has its own coverage in
``sagent/providers/anthropic_cli_session/tripwire_test.py``.
"""

from __future__ import annotations

from pathlib import Path

import logging
import os
import sys

import pytest


# The serve module lives in ``plugin/.../bin/serve.py`` and is
# normally invoked as a script. Tests import it as a module — add the
# bin/ dir to sys.path so the relative import works without the
# script having to be in PYTHONPATH already.
_SERVE_DIR = Path(__file__).resolve().parent.parent / "bin"
if str(_SERVE_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVE_DIR))

import serve  # noqa: E402

from sagent.providers.anthropic_cli_session import (  # noqa: E402
    CanaryResult,
    DiffFinding,
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test with the env var unset so prior state can't leak."""
    monkeypatch.delenv(serve._MATERIALIZER_TRIPWIRE_ENV, raising=False)


@pytest.mark.asyncio
async def test_unset_env_runs_canary_and_keeps_default_on(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """No env var → fire the canary (default-on contract).

    This is the headline graduation behaviour: v2.1-β is default-on,
    so unset env MUST trigger the canary. If this test fails, the
    flip got reverted.
    """
    canary_called = False

    async def _fake_canary(**_kwargs: object) -> CanaryResult:
        nonlocal canary_called
        canary_called = True
        return CanaryResult(is_safe=True, findings=[], claude_jsonl_path=None)

    monkeypatch.setattr(serve, "arun_canary_against_live_cli", _fake_canary)
    with caplog.at_level(logging.INFO):
        await serve._run_materializer_tripwire()

    assert canary_called is True, "canary must fire when env unset (default-on)"
    # No fallback set — default-on stays in effect.
    assert os.environ.get(serve._MATERIALIZER_TRIPWIRE_ENV) is None
    assert "PASS" in caplog.text


@pytest.mark.asyncio
async def test_explicit_opt_out_skips_canary(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """``SAGENT_CLI_OWN_SESSION=0`` skips the canary entirely.

    Operator opt-out path. Costs nothing at boot, falls back to v2.
    Without this, an operator who genuinely wants v2 behaviour pays
    a 4-5s canary cost every boot.
    """
    canary_called = False

    async def _fake_canary(**_kwargs: object) -> CanaryResult:
        nonlocal canary_called
        canary_called = True
        return CanaryResult(is_safe=True, findings=[], claude_jsonl_path=None)

    monkeypatch.setattr(serve, "arun_canary_against_live_cli", _fake_canary)
    monkeypatch.setenv(serve._MATERIALIZER_TRIPWIRE_ENV, "0")
    with caplog.at_level(logging.INFO):
        await serve._run_materializer_tripwire()

    assert canary_called is False
    assert os.environ.get(serve._MATERIALIZER_TRIPWIRE_ENV) == "0"
    assert "opt-out" in caplog.text


@pytest.mark.asyncio
async def test_clean_verdict_leaves_env_unchanged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Canary returns ``is_safe=True`` → env stays as-is, PASS logged.

    Default-on path: env was unset, stays unset; the plugin's
    ``common.py`` reads it and sees no opt-out, so
    ``materialize_session=True``.
    """

    async def _fake_canary(**_kwargs: object) -> CanaryResult:
        return CanaryResult(is_safe=True, findings=[], claude_jsonl_path=None)

    monkeypatch.setattr(serve, "arun_canary_against_live_cli", _fake_canary)
    with caplog.at_level(logging.INFO):
        await serve._run_materializer_tripwire()

    assert os.environ.get(serve._MATERIALIZER_TRIPWIRE_ENV) is None
    assert "PASS" in caplog.text


@pytest.mark.asyncio
async def test_drift_verdict_sets_opt_out(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Drift findings → env set to ``0`` so plugin falls back to v2.

    Per-finding WARNING lines are emitted so operators can pin the
    CLI version or update the materializer.
    """

    async def _fake_canary(**_kwargs: object) -> CanaryResult:
        return CanaryResult(
            is_safe=False,
            findings=[
                DiffFinding(
                    location="entry[3].type",
                    detail="unknown entry type 'shiny-new-thing'",
                ),
                DiffFinding(
                    location="entry[5].message",
                    detail="required field 'message' missing from claude entry",
                ),
            ],
            claude_jsonl_path=None,
        )

    monkeypatch.setattr(serve, "arun_canary_against_live_cli", _fake_canary)
    with caplog.at_level(logging.WARNING):
        await serve._run_materializer_tripwire()

    assert os.environ.get(serve._MATERIALIZER_TRIPWIRE_ENV) == "0"
    assert "FAIL" in caplog.text
    assert "shiny-new-thing" in caplog.text
    assert "message" in caplog.text


@pytest.mark.asyncio
async def test_canary_exception_sets_opt_out(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An exception from the canary sets env to ``0`` (never crashes boot).

    The canary going wrong is operational news, but it must NOT block
    the server from starting. v2 (CLI-owned mode) is always a safe
    fallback.
    """

    async def _exploding_canary(**_kwargs: object) -> CanaryResult:
        raise RuntimeError("simulated canary failure")

    monkeypatch.setattr(serve, "arun_canary_against_live_cli", _exploding_canary)
    with caplog.at_level(logging.WARNING):
        await serve._run_materializer_tripwire()

    assert os.environ.get(serve._MATERIALIZER_TRIPWIRE_ENV) == "0"
    assert "simulated canary failure" in caplog.text


@pytest.mark.asyncio
async def test_opt_out_variants_all_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``0`` / ``false`` / ``no`` / mixed-case all opt out.

    Operators paste from docs, mix case, copy/paste artifacts.
    Accepting common falsy values matches the documented
    ``SAGENT_CLI_OWN_SESSION`` opt-out contract.
    """
    canary_calls = 0

    async def _fake_canary(**_kwargs: object) -> CanaryResult:
        nonlocal canary_calls
        canary_calls += 1
        return CanaryResult(is_safe=True, findings=[], claude_jsonl_path=None)

    monkeypatch.setattr(serve, "arun_canary_against_live_cli", _fake_canary)
    for value in ("0", "false", "FALSE", "No"):
        monkeypatch.setenv(serve._MATERIALIZER_TRIPWIRE_ENV, value)
        await serve._run_materializer_tripwire()
        assert os.environ.get(serve._MATERIALIZER_TRIPWIRE_ENV) == value
    assert canary_calls == 0


@pytest.mark.asyncio
async def test_unrecognized_env_value_triggers_canary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An env value that's not a recognized opt-out (e.g. ``1``,
    ``true``, garbage) is treated as default-on → canary fires.

    The opt-out is a strict allow-list; anything else means "honor
    the default". This is a forward-compat guard: if the env var
    semantics change later, an unset value AND a leftover legacy
    truthy value both keep working.
    """
    canary_called = False

    async def _fake_canary(**_kwargs: object) -> CanaryResult:
        nonlocal canary_called
        canary_called = True
        return CanaryResult(is_safe=True, findings=[], claude_jsonl_path=None)

    monkeypatch.setattr(serve, "arun_canary_against_live_cli", _fake_canary)
    for value in ("1", "true", "yes", "garbage"):
        canary_called = False
        monkeypatch.setenv(serve._MATERIALIZER_TRIPWIRE_ENV, value)
        await serve._run_materializer_tripwire()
        assert canary_called is True, (
            f"value {value!r} should be treated as default-on; canary must fire"
        )
