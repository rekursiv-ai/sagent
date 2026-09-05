"""Subscription authentication, request restrictions, and credential I/O tests."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import base64
import io
import json
import time

import httpx2
import openai
import pytest

from sagent.catalog import openai as openai_catalog
from sagent.providers import OpenAI
from sagent.providers.lib.errors import (
    PER_ITEM_STRING_CAP_BODY,
    StreamingResponseNotReadError,
    find_response_not_read,
)
from sagent.providers.openai import sub as openai_sub
from sagent.providers.openai.sub import (
    OpenAISubscription,
    _jwt_claim,
    _jwt_exp,
    _jwt_payload,
    _subscription_context,
)
from sagent.types.capability import (
    ModelCapability,
    ModelLimits,
    ModelSettings,
    ThinkingEffort,
)
from sagent.types.exceptions import AuthRefreshError
from sagent.types.model import (
    ModelRequest,
)
from sagent.types.runtime import (
    UserMessage,
)


def test_subscription_context_clamps_request_tokens() -> None:
    cap = ModelCapability(
        context=MappingProxyType(
            {
                "": ModelLimits(
                    max_request_tokens=1_000_000, max_response_tokens=1_000_000
                )
            }
        )
    )
    clamped = _subscription_context(cap)[""]
    # Wire contract is 272_000 / 32_000 -- see sub._SUBSCRIPTION_MAX_*.
    assert clamped.max_request_tokens == 272_000
    assert clamped.max_response_tokens == 32_000


def test_subscription_context_keeps_small_windows() -> None:
    cap = ModelCapability(
        context=MappingProxyType(
            {"": ModelLimits(max_request_tokens=100_000, max_response_tokens=10_000)}
        )
    )
    clamped = _subscription_context(cap)[""]
    assert clamped.max_request_tokens == 100_000
    assert clamped.max_response_tokens == 10_000


def test_subscription_context_drops_the_long_window_tag() -> None:
    """``+1m`` clamps to exactly the base id, so offering it would mislead."""
    cap = ModelCapability(
        context=MappingProxyType(
            {
                "": ModelLimits(max_request_tokens=272_000),
                "+1m": ModelLimits(max_request_tokens=1_050_000),
            }
        )
    )
    assert _subscription_context(cap).keys() == {""}


def test_subscription_context_inherits_size_caps_from_parent() -> None:
    # Only the token windows are subscription-specific; the image/wire byte
    # caps are a property of the underlying model and must flow through
    # unchanged, not be overwritten by a stale local constant. A divergent
    # parent capability proves inheritance rather than a hardcoded match.
    cap = ModelCapability(
        context=MappingProxyType(
            {
                "": ModelLimits(
                    max_response_tokens=1_000_000,
                    max_image_edge_px=4096,
                    max_image_bytes=7_000_000,
                    max_request_bytes=33_000_000,
                )
            }
        )
    )
    clamped = _subscription_context(cap)[""]
    assert clamped.max_image_edge_px == 4096
    assert clamped.max_image_bytes == 7_000_000
    assert clamped.max_request_bytes == 33_000_000


# 1×1 transparent PNG -- smallest valid PNG ``image_lib.resize`` will accept.


def _make_jwt(payload: dict[str, object]) -> str:
    header_b = base64.urlsafe_b64encode(b'{"alg":"none"}').rstrip(b"=")
    body_b = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=")
    return f"{header_b.decode()}.{body_b.decode()}.sig"


def test_login_manual_advertises_localhost_redirect_uri(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Manual login must send the ``localhost`` redirect_uri Hydra allow-lists.

    ``127.0.0.1`` is rejected with ``authorize_hydra_invalid_request``.
    """
    monkeypatch.setattr(openai_sub, "DEFAULT_CREDENTIALS_PATH", tmp_path / "auth.json")
    access = _make_jwt(
        {
            "exp": time.time() + 3600,
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acct-123",
                "chatgpt_plan_type": "pro",
            },
        }
    )
    token_resp = MagicMock()
    token_resp.status_code = 200
    token_resp.json.return_value = {
        "access_token": access,
        "refresh_token": "refresh-xyz",
        "expires_in": 3600,
    }
    captured: dict[str, object] = {}

    def _post(_url: str, *, data: dict[str, str], **_kw: object) -> MagicMock:
        captured["redirect_uri"] = data["redirect_uri"]
        return token_resp

    http_client = MagicMock()
    http_client.__enter__.return_value.post.side_effect = _post

    out = io.StringIO()
    with (
        patch(
            "sagent.providers.openai.sub.httpx2.Client",
            return_value=http_client,
        ),
        patch.object(
            openai_sub,
            "parse_manual_auth_code",
            return_value="the-code",
        ),
        patch("builtins.input", return_value="the-code#state"),
    ):
        OpenAISubscription.login(out, manual=True)

    printed = out.getvalue()
    assert "localhost" in printed
    assert "127.0.0.1" not in printed
    assert captured["redirect_uri"] == "http://localhost:1455/auth/callback"


def test_jwt_payload_round_trip() -> None:
    token = _make_jwt({"exp": 1234, "https://api.openai.com/auth": {"x": "y"}})
    assert _jwt_payload(token).get("exp") == 1234


def test_jwt_payload_short_token_returns_empty() -> None:
    assert _jwt_payload("only_one_part") == {}


def test_jwt_payload_invalid_base64_returns_empty() -> None:
    # Crafted body that decodes to non-JSON.
    bad = "h." + base64.urlsafe_b64encode(b"not json").rstrip(b"=").decode() + ".s"
    assert _jwt_payload(bad) == {}


def test_jwt_exp_extracts_value() -> None:
    token = _make_jwt({"exp": 9001})
    assert _jwt_exp(token) == 9001.0


def test_jwt_exp_missing_returns_zero() -> None:
    token = _make_jwt({})
    assert _jwt_exp(token) == 0.0


def test_jwt_claim_nested_value() -> None:
    token = _make_jwt(
        {"https://api.openai.com/auth": {"chatgpt_account_id": "acc-123"}}
    )
    assert _jwt_claim(token, "https://api.openai.com/auth", "chatgpt_account_id") == (
        "acc-123"
    )


def test_jwt_claim_namespace_missing_returns_empty() -> None:
    token = _make_jwt({})
    assert _jwt_claim(token, "ns", "key") == ""


def _write_creds_file(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_creds(
    access: str, *, expires_at: float = 0.0
) -> OpenAISubscription.Credentials:
    return OpenAISubscription.Credentials(
        access_token=access,
        refresh_token=_FAKE_REFRESH,
        account_id=_FAKE_ACCOUNT,
        expires_at=expires_at,
    )


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    access = _make_jwt({"exp": 1700.0})
    target = tmp_path / "auth.json"
    OpenAISubscription.save(_make_creds(access, expires_at=1700.0), path=target)
    loaded = OpenAISubscription.load(path=target)
    assert loaded["access_token"] == access
    assert loaded["refresh_token"] == _FAKE_REFRESH
    assert loaded["account_id"] == _FAKE_ACCOUNT
    # ``expires_at`` is recomputed from the JWT exp claim on load.
    assert loaded["expires_at"] == 1700.0


def test_default_credential_io_honors_codex_home(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Sagent follows the same relocated auth file as the Codex CLI."""
    codex_home = tmp_path / "custom-codex-home"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    access = _make_jwt({"exp": 1700.0})

    OpenAISubscription.save(_make_creds(access, expires_at=1700.0))
    loaded = OpenAISubscription.load()

    assert (codex_home / "auth.json").exists()
    assert loaded["access_token"] == access


def test_save_preserves_existing_fields(tmp_path: Path) -> None:
    target = tmp_path / "auth.json"
    _write_creds_file(target, {"keep_me": True, "tokens": {"old": "x"}})
    creds = _make_creds(_make_jwt({"exp": 0.0}))
    OpenAISubscription.save(creds, path=target)
    on_disk = json.loads(target.read_text(encoding="utf-8"))
    assert on_disk["keep_me"] is True
    assert on_disk["auth_mode"] == "chatgpt"
    assert on_disk["tokens"]["access_token"] == creds["access_token"]
    # Old fields under ``tokens`` survive.
    assert on_disk["tokens"]["old"] == "x"


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        OpenAISubscription.load(path=tmp_path / "nope.json")


def test_load_api_key_mode_file_raises_actionable_error(tmp_path: Path) -> None:
    target = tmp_path / "auth.json"
    _write_creds_file(
        target,
        {"auth_mode": "apikey", "OPENAI_API_KEY": "test-api-key"},
    )
    with pytest.raises(ValueError, match=r"API-key.*OpenAI.*env"):
        OpenAISubscription.load(path=target)


def test_load_incomplete_oauth_file_raises_value_error(tmp_path: Path) -> None:
    target = tmp_path / "auth.json"
    _write_creds_file(target, {"auth_mode": "chatgpt", "tokens": {}})
    with pytest.raises(ValueError, match="missing required OAuth fields"):
        OpenAISubscription.load(path=target)


@pytest.mark.parametrize("payload", [[], None, "not an object"])
def test_load_non_object_json_raises_value_error(
    tmp_path: Path,
    payload: object,
) -> None:
    target = tmp_path / "auth.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        OpenAISubscription.load(path=target)


def test_save_writes_id_token_when_present(tmp_path: Path) -> None:
    target = tmp_path / "auth.json"
    creds = _make_creds(_make_jwt({"exp": 0.0}))
    sentinel = "fake-jwt-marker"
    creds["id_token"] = sentinel
    OpenAISubscription.save(creds, path=target)
    on_disk = json.loads(target.read_text(encoding="utf-8"))
    assert on_disk["tokens"]["id_token"] == sentinel


# Test-only token values; not credentials. Held in module variables so the
# bandit-style S105/S106 rules don't fire on inline literals at constructor
# call sites.
_FAKE_ACCESS = ""
_FAKE_REFRESH = "rt-test"
_FAKE_ACCOUNT = "acc"
_FRESH_REFRESH = "rt-fresh"
_STALE_REFRESH = "rt-stale"


def _make_provider(*, expires_at: float = 9999.0) -> OpenAISubscription:
    return OpenAISubscription(
        access_token=_FAKE_ACCESS or _make_jwt({"exp": expires_at}),
        refresh_token=_FAKE_REFRESH,
        account_id=_FAKE_ACCOUNT,
        expires_at=expires_at,
    )


def test_subscription_from_key_returns_plain_openai() -> None:
    """``from_key`` delegates to the API-key ``OpenAI`` class."""
    p = OpenAISubscription.from_key("sk-test")
    assert isinstance(p, OpenAI)
    assert not isinstance(p, OpenAISubscription)


@pytest.mark.parametrize("subscription", [False, True])
def test_model_accepts_keyword_model_id(subscription: bool) -> None:
    provider = _make_provider() if subscription else OpenAI.from_key("test-key")
    assert provider.model(model_id="gpt-5.5").capability.model_id == "gpt-5.5"


def test_subscription_model_unknown_id_raises() -> None:
    p = _make_provider()
    with pytest.raises(ValueError, match="Unknown model"):
        _ = p.model("not-a-model")


def test_subscription_model_uses_default_when_unset() -> None:
    m = _make_provider().model()
    assert m.capability.model_id == OpenAISubscription.DEFAULT_MODEL


def test_subscription_default_model_is_openai_default_without_1m() -> None:
    """Sub default = the API default's base id (``+1m`` is not in the catalog)."""
    assert OpenAISubscription.DEFAULT_MODEL == "gpt-5.6-sol"
    assert OpenAI.DEFAULT_MODEL == "gpt-5.6-sol+1m"
    # The narrowed default must resolve against the narrowed catalog.
    assert OpenAISubscription.DEFAULT_MODEL in OpenAISubscription.CAPABILITIES


def test_subscription_default_utility_model_inherits_from_openai() -> None:
    """``OpenAISubscription`` defers to ``OpenAI.DEFAULT_UTILITY_MODEL``."""
    assert OpenAISubscription.DEFAULT_UTILITY_MODEL == OpenAI.DEFAULT_UTILITY_MODEL


def test_subscription_utility_model_uses_utility_default() -> None:
    m = _make_provider().utility_model()
    assert m.capability.model_id == OpenAISubscription.DEFAULT_UTILITY_MODEL


def test_subscription_model_clamps_against_wire_contract() -> None:
    m = _make_provider().model("gpt-5.5")
    # KNOWN_MODELS is clamped via ``_subscription_profile``.
    assert m.limits.max_request_tokens == 272_000
    assert m.limits.max_response_tokens == 32_000


def test_subscription_rejects_1m_ids() -> None:
    """``+1m`` buys nothing under the wire contract, so it is not a known id."""
    p = _make_provider()
    with pytest.raises(ValueError, match="Unknown model"):
        _ = p.model("gpt-5.6-sol+1m")
    assert not any(name.endswith("+1m") for name in OpenAISubscription.CAPABILITIES)


def test_subscription_model_supports_thinking_via_reasoning_effort() -> None:
    m = _make_provider().model("gpt-5.5")
    assert m.capability.thinking_budget == frozenset({"none", "auto"})
    assert m.capability.account_auth is True


def test_subscription_effort_matches_api_key_path() -> None:
    """Both OpenAI transports must agree on which ids are reasoning models.

    They share one capability catalog, so a divergence is a contract bug.
    Prefix guessing (which over-matched a hypothetical ``omni-*``) is gone;
    an id absent from the catalog is not resolvable on either path.
    """
    sub_p = _make_provider()
    api_p = OpenAI.from_key("k")
    for model_id in ("o1", "o3-mini", "gpt-5.5"):
        assert (
            sub_p.model(model_id).capability.thinking_effort
            == api_p.model(model_id).capability.thinking_effort
        )
    for provider in (sub_p, api_p):
        with pytest.raises(ValueError, match="Unknown model"):
            _ = provider.model("omni-foo")


def test_subscription_valid_service_tiers_priority_only() -> None:
    # ``/fast`` sets service_tier="priority"; an unqualified request still
    # sends the vendor default, so ``auto`` stays selectable and nothing else.
    m = _make_provider().model("gpt-5.5")
    assert m.capability.service_tier == frozenset({"auto", "priority"})


@pytest.mark.anyio
async def test_subscription_provider_close_releases_sdk() -> None:
    """Model teardown preserves the SDK until its owning provider closes."""
    provider = _make_provider()
    model = provider.model("gpt-5.5")
    sdk = MagicMock()
    sdk.close = AsyncMock()
    provider._sdk = sdk
    provider._sdk_token = "dummy-token"  # noqa: S105 -- test fixture, not a secret

    await model.close()
    sdk.close.assert_not_awaited()
    assert provider._sdk is sdk

    await provider.close_sdk()
    sdk.close.assert_awaited_once()
    assert provider._sdk is None
    assert provider._sdk_token is None


def test_subscription_priority_tier_reaches_the_wire() -> None:
    m = _make_provider().model("gpt-5.5")
    m._settings = ModelSettings(capability=m.capability, service_tier="priority")
    assert m._effective_service_tier() == "priority"


def test_subscription_omits_the_default_tier() -> None:
    """``auto`` is "let the vendor pick"; sending it pins nothing."""
    assert _make_provider().model("gpt-5.5")._effective_service_tier() is None


class _StatusError(Exception):
    """Error carrying a typed ``status_code`` (what ``error_status_code`` reads)."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def test_subscription_context_overflow_detection() -> None:
    m = _make_provider().model("gpt-5.5")
    assert m.is_context_overflow(RuntimeError("context_length_exceeded")) is True
    assert m.is_context_overflow(RuntimeError("exceeds the context window")) is True
    assert m.is_context_overflow(RuntimeError("other failure")) is False


def test_subscription_per_item_string_cap_is_context_overflow() -> None:
    """The per-item string cap must route to the compactor's shrink path.

    Session ``190b6baec7ed`` wedged here: an 11 MB tool result tripped
    OpenAI's 10 MiB per-ITEM cap, the phrase matched no classifier, and
    ``stream`` re-raised the raw ``BadRequestError`` (``sub.py:1054``).
    That is neither ``RequestTooLargeError`` nor ``PromptTooLongError``,
    so ``_shrink_groups_for_compaction`` never ran -- and because the
    compactor's own request carries the same item, every retry (and every
    later turn, and ``/compact``) failed identically until the session died.

    Classifying it as token overflow reaches the shrink-and-retry that
    caps oversized tool results, which is what actually clears it.
    """
    m = _make_provider().model("gpt-5.5")
    err = _StatusError(PER_ITEM_STRING_CAP_BODY, 400)
    assert m.is_context_overflow(err) is True


def test_subscription_byte_limit_not_context_overflow() -> None:
    """A 413 byte limit must not classify as token-context overflow.

    Uniform with the compat/google models: byte overflow routes to
    byte-overflow recovery (shed attachment bytes), never to the
    ``/model`` larger-window remediation.
    """
    m = _make_provider().model("gpt-5.5")
    err = _StatusError("Request entity too large", 413)
    assert m.is_context_overflow(err) is False


def test_subscription_byte_413_with_window_phrase_is_not_context_overflow() -> None:
    """A 413 whose body also names the context window stays byte overflow.

    Discriminating case: the subscription classifier must consult the shared
    ``is_request_too_large`` guard (status 413 -> byte) rather than its own
    substring list, so it matches the compat/google siblings. Without the
    guard, the bare ``"exceeds the context window"`` substring would
    mis-route a 413 byte limit to token-overflow recovery.
    """
    m = _make_provider().model("gpt-5.5")
    err = _StatusError("413: request entity too large; exceeds the context window", 413)
    assert m.is_context_overflow(err) is False


def test_subscription_retryable_provider_error() -> None:
    m = _make_provider().model("gpt-5.5")
    assert m.is_retryable_provider_error(RuntimeError("You can retry your request"))
    assert not m.is_retryable_provider_error(RuntimeError("permanent failure"))


def test_subscription_expired_property_true_when_past() -> None:
    p = _make_provider(expires_at=0.0)
    assert p.expired is True


def test_subscription_expired_property_false_when_far_future() -> None:
    p = _make_provider(expires_at=9_999_999_999.0)
    assert p.expired is False


def _create_kwargs_for(
    *, model_id: str = "gpt-5.6-sol", effort: ThinkingEffort = "none"
) -> dict[str, object]:
    model = _make_provider().model(model_id)
    model.settings.thinking_effort = effort
    return dict(model._build_kwargs(ModelRequest(messages=[UserMessage(text="hi")])))


def _wire_effort_for(*, model_id: str, effort: ThinkingEffort) -> object:
    """The ``reasoning.effort`` value the request carried."""
    reasoning = (_create_kwargs_for(model_id=model_id, effort=effort))["reasoning"]
    assert isinstance(reasoning, dict)
    return cast(dict[str, object], reasoning)["effort"]


@pytest.mark.parametrize(
    ("effort", "wire_effort"),
    [("low", "low"), ("medium", "medium"), ("xhigh", "xhigh")],
)
def test_subscription_stream_maps_pre_56_effort_to_wire_vocabulary(
    effort: ThinkingEffort,
    wire_effort: str,
) -> None:
    """Subscription requests preserve the model's native effort vocabulary."""
    assert _wire_effort_for(model_id="gpt-5.5", effort=effort) == wire_effort


@pytest.mark.parametrize(
    ("effort", "wire_effort"),
    [("min", "none"), ("medium", "medium"), ("xhigh", "xhigh"), ("max", "max")],
)
def test_subscription_stream_preserves_gpt_56_effort(
    effort: ThinkingEffort,
    wire_effort: str,
) -> None:
    assert _wire_effort_for(model_id="gpt-5.6-sol", effort=effort) == wire_effort


def test_subscription_catalog_efforts_are_all_buildable() -> None:
    """Every effort the catalog offers must reach the wire."""
    model_id = "gpt-5.6-sol"
    capability = _make_provider().model(model_id).capability
    for effort in capability.thinking_effort - {"none"}:
        assert _wire_effort_for(
            model_id=model_id, effort=effort
        ) == openai_catalog.reasoning_effort(effort, model_id=model_id)


def test_subscription_stream_requests_reasoning_summary_when_thinking() -> None:
    """Thinking on must request a reasoning summary, else no text streams.

    The OpenAI Responses API only emits reasoning-text deltas when
    ``reasoning.summary`` is set; without it the model reasons silently
    and ``on_thinking`` never fires. Requesting thinking must therefore
    set ``summary`` so the reasoning surface is actually populated.
    """
    reasoning = (_create_kwargs_for(model_id="gpt-5.5", effort="medium"))["reasoning"]
    assert isinstance(reasoning, dict)
    assert cast(dict[str, object], reasoning)["summary"] == "auto"


def test_subscription_stream_requests_encrypted_reasoning_for_stateless_history() -> (
    None
):
    kwargs = _create_kwargs_for(effort="medium")
    assert kwargs["store"] is False
    assert kwargs["include"] == ["reasoning.encrypted_content"]
    reasoning = kwargs["reasoning"]
    assert isinstance(reasoning, dict)
    assert cast(dict[str, object], reasoning)["context"] == "all_turns"


def test_subscription_stream_omits_reasoning_object_without_thinking() -> None:
    """``none`` means "send no knob", not "send the wire's off value"."""
    kwargs = _create_kwargs_for(model_id="gpt-5.5")
    assert "reasoning" not in kwargs
    assert "include" not in kwargs


class TestHandleAuthError:
    """After ``/login`` writes fresh creds, the live provider must reload.

    Bug: ``do_login`` (``repl/run_repl.py``) calls
    ``provider.handle_auth_error()`` to hot-reload in-memory tokens
    from disk after a successful re-auth. Without this hook on
    :class:`OpenAISubscription`, the live instance keeps its stale
    ``_access_token`` / ``_refresh_token`` after ``/login``; the next
    model call either sends the stale access token (-> API 401) or
    posts the stale refresh token (-> 400 ->
    :class:`AuthRefreshError`). Either way ``/login`` is a no-op until
    process restart.
    """

    @pytest.mark.anyio
    async def test_reloads_from_disk(self, tmp_path: Path) -> None:
        """Fresh creds on disk replace stale in-memory creds."""
        cred_file = tmp_path / "auth.json"
        fresh_access = _make_jwt({"exp": time.time() + 3600})
        OpenAISubscription.save(
            OpenAISubscription.Credentials(
                access_token=fresh_access,
                refresh_token=_FRESH_REFRESH,
                account_id=_FAKE_ACCOUNT,
                expires_at=time.time() + 3600,
            ),
            path=cred_file,
        )
        provider = OpenAISubscription(
            access_token=_make_jwt({"exp": 0.0}),
            refresh_token=_STALE_REFRESH,
            account_id=_FAKE_ACCOUNT,
            expires_at=0.0,
        )
        with patch(
            "sagent.providers.openai.sub.DEFAULT_CREDENTIALS_PATH",
            cred_file,
        ):
            await provider.handle_auth_error()
        assert provider._access_token == fresh_access
        assert provider._refresh_token == _FRESH_REFRESH

    @pytest.mark.anyio
    async def test_refreshes_when_adopted_disk_token_also_expired(
        self, tmp_path: Path
    ) -> None:
        """A DIFFERENT but already-expired disk token must still force a refresh.

        On a 401, a sibling may have written a token whose own lifetime already
        lapsed. Adopting it and returning (no re-expiry check) hands the caller a
        stale token -> the retry 401s again. ``handle_auth_error`` must mirror
        ``_ensure_valid``: after adopting a fresher-looking disk token, refresh
        anyway if it is still expired.
        """
        cred_file = tmp_path / "auth.json"
        expired_disk = _make_jwt({"exp": time.time() - 50})
        OpenAISubscription.save(
            OpenAISubscription.Credentials(
                access_token=expired_disk,
                refresh_token=_FRESH_REFRESH,
                account_id=_FAKE_ACCOUNT,
                expires_at=time.time() - 50,
            ),
            path=cred_file,
        )
        provider = OpenAISubscription(
            access_token=_make_jwt({"exp": 0.0}),
            refresh_token=_STALE_REFRESH,
            account_id=_FAKE_ACCOUNT,
            expires_at=time.time() - 100,
        )
        refreshed: list[bool] = []

        async def _fake_refresh() -> None:
            refreshed.append(True)
            provider._access_token = _make_jwt({"exp": time.time() + 3600})
            provider._expires_at = time.time() + 3600

        with (
            patch.object(provider, "_refresh", _fake_refresh),
            patch(
                "sagent.providers.openai.sub.DEFAULT_CREDENTIALS_PATH",
                cred_file,
            ),
        ):
            await provider.handle_auth_error()

        assert refreshed == [True], "expired adopted disk token must trigger _refresh"
        assert not provider.expired

    @pytest.mark.anyio
    async def test_force_refresh_when_disk_same_even_if_clock_valid(
        self, tmp_path: Path
    ) -> None:
        """A 401 on a clock-VALID token (same on disk) must STILL force a refresh.

        ``handle_auth_error`` runs because the server already rejected this exact
        token (revoked, or server/client clock skew). The local ``expired`` clock
        says "valid", but the server is authoritative -- trusting the clock and
        returning without refreshing leaves the caller retrying the same rejected
        token in an infinite 401 loop. Only a genuinely DIFFERENT disk token (a
        sibling's refresh) may be adopted without a network refresh.
        """
        cred_file = tmp_path / "auth.json"
        # exp far in the future -> ``self.expired`` is False, yet the server 401'd.
        clock_valid = _make_jwt({"exp": time.time() + 3600})
        OpenAISubscription.save(
            OpenAISubscription.Credentials(
                access_token=clock_valid,
                refresh_token=_STALE_REFRESH,
                account_id=_FAKE_ACCOUNT,
                expires_at=time.time() + 3600,
            ),
            path=cred_file,
        )
        provider = OpenAISubscription(
            access_token=clock_valid,  # SAME as disk
            refresh_token=_STALE_REFRESH,
            account_id=_FAKE_ACCOUNT,
            expires_at=time.time() + 3600,
        )
        refreshed: list[bool] = []

        async def _fake_refresh() -> None:
            refreshed.append(True)
            provider._access_token = _make_jwt({"exp": time.time() + 7200})

        with (
            patch.object(provider, "_refresh", _fake_refresh),
            patch(
                "sagent.providers.openai.sub.DEFAULT_CREDENTIALS_PATH",
                cred_file,
            ),
        ):
            await provider.handle_auth_error()

        assert refreshed == [True], (
            "a 401 on a clock-valid same-disk token must force _refresh, not return"
        )

    @pytest.mark.anyio
    async def test_force_refresh_when_disk_same(self, tmp_path: Path) -> None:
        """Disk matches in-memory -> fall through to network refresh."""
        in_memory_access = _make_jwt({"exp": 0.0})
        cred_file = tmp_path / "auth.json"
        OpenAISubscription.save(
            OpenAISubscription.Credentials(
                access_token=in_memory_access,
                refresh_token=_STALE_REFRESH,
                account_id=_FAKE_ACCOUNT,
                expires_at=0.0,
            ),
            path=cred_file,
        )
        provider = OpenAISubscription(
            access_token=in_memory_access,
            refresh_token=_STALE_REFRESH,
            account_id=_FAKE_ACCOUNT,
            expires_at=0.0,
        )
        refreshed_access = _make_jwt({"exp": time.time() + 3600})
        mock_resp = MagicMock()
        mock_resp.json = MagicMock(
            return_value={
                "access_token": refreshed_access,
                "refresh_token": _FRESH_REFRESH,
                "expires_in": 3600,
            },
        )
        mock_resp.status_code = 200
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_resp)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)
        with (
            patch(
                "sagent.providers.openai.sub.httpx2.AsyncClient",
                return_value=mock_http,
            ),
            patch(
                "sagent.providers.openai.sub.DEFAULT_CREDENTIALS_PATH",
                cred_file,
            ),
        ):
            await provider.handle_auth_error()
        assert provider._access_token == refreshed_access
        assert provider._refresh_token == _FRESH_REFRESH

    @pytest.mark.anyio
    async def test_force_refresh_when_disk_missing(self, tmp_path: Path) -> None:
        """Missing disk file -> caught -> fall through to network refresh."""
        provider = OpenAISubscription(
            access_token=_make_jwt({"exp": 0.0}),
            refresh_token=_STALE_REFRESH,
            account_id=_FAKE_ACCOUNT,
            expires_at=0.0,
        )
        refreshed_access = _make_jwt({"exp": time.time() + 3600})
        mock_resp = MagicMock()
        mock_resp.json = MagicMock(
            return_value={
                "access_token": refreshed_access,
                "refresh_token": _FRESH_REFRESH,
                "expires_in": 3600,
            },
        )
        mock_resp.status_code = 200
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_resp)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)
        with (
            patch(
                "sagent.providers.openai.sub.httpx2.AsyncClient",
                return_value=mock_http,
            ),
            patch(
                "sagent.providers.openai.sub.DEFAULT_CREDENTIALS_PATH",
                tmp_path / "does_not_exist.json",
            ),
        ):
            await provider.handle_auth_error()
        assert provider._access_token == refreshed_access
        assert provider._refresh_token == _FRESH_REFRESH


class TestEnsureValidRace:
    """``_ensure_valid`` must reload disk before refreshing.

    Multi-process race: process A's token expires; A refreshes,
    rotating the refresh_token and saving fresh creds to disk.
    Process B's in-memory refresh_token is now revoked. When B's
    token expires, ``_ensure_valid`` must check disk first; if disk
    has fresher creds, adopt them without posting B's stale
    refresh_token (which would 400). Mirrors ``handle_auth_error``'s
    disk-first pattern.
    """

    @pytest.mark.anyio
    async def test_reloads_from_disk_when_sibling_refreshed(
        self, tmp_path: Path
    ) -> None:
        """Fresh disk creds preempt the network refresh entirely."""
        cred_file = tmp_path / "auth.json"
        fresh_access = _make_jwt({"exp": time.time() + 3600})
        OpenAISubscription.save(
            OpenAISubscription.Credentials(
                access_token=fresh_access,
                refresh_token=_FRESH_REFRESH,
                account_id=_FAKE_ACCOUNT,
                expires_at=time.time() + 3600,
            ),
            path=cred_file,
        )
        provider = OpenAISubscription(
            access_token=_make_jwt({"exp": 0.0}),
            refresh_token=_STALE_REFRESH,
            account_id=_FAKE_ACCOUNT,
            expires_at=time.time() - 100,
        )

        # If ``_refresh`` is reached, the test fails fast -- the disk
        # reload should preempt it.
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(
            side_effect=AssertionError(
                "_ensure_valid called _refresh instead of reloading from disk"
            ),
        )
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "sagent.providers.openai.sub.httpx2.AsyncClient",
                return_value=mock_http,
            ),
            patch(
                "sagent.providers.openai.sub.DEFAULT_CREDENTIALS_PATH",
                cred_file,
            ),
        ):
            token = await provider._ensure_valid()

        assert token == fresh_access
        assert provider._access_token == fresh_access
        assert provider._refresh_token == _FRESH_REFRESH

    @pytest.mark.anyio
    async def test_refreshes_when_disk_token_also_expired(self, tmp_path: Path) -> None:
        """An adopted disk token that is ALSO expired must trigger a refresh.

        A sibling may have written a token whose own lifetime has already
        lapsed (its refresh aged out, or clock skew). Adopting it and returning
        it would 401 on the next call; ``_ensure_valid`` must fall through to
        ``_refresh`` instead of handing back the stale disk token.
        """
        cred_file = tmp_path / "auth.json"
        expired_disk = _make_jwt({"exp": time.time() - 50})
        OpenAISubscription.save(
            OpenAISubscription.Credentials(
                access_token=expired_disk,
                refresh_token=_FRESH_REFRESH,
                account_id=_FAKE_ACCOUNT,
                expires_at=time.time() - 50,
            ),
            path=cred_file,
        )
        provider = OpenAISubscription(
            access_token=_make_jwt({"exp": 0.0}),
            refresh_token=_STALE_REFRESH,
            account_id=_FAKE_ACCOUNT,
            expires_at=time.time() - 100,
        )
        refreshed: list[bool] = []

        async def _fake_refresh() -> None:
            refreshed.append(True)
            provider._access_token = _make_jwt({"exp": time.time() + 3600})
            provider._expires_at = time.time() + 3600

        with (
            patch.object(provider, "_refresh", _fake_refresh),
            patch(
                "sagent.providers.openai.sub.DEFAULT_CREDENTIALS_PATH",
                cred_file,
            ),
        ):
            token = await provider._ensure_valid()

        assert refreshed == [True], "expired disk token must trigger _refresh"
        assert token == provider._access_token
        assert not provider.expired

    @pytest.mark.anyio
    async def test_adopting_disk_token_closes_old_sdk(self, tmp_path: Path) -> None:
        """Adopting a sibling's fresher disk token must CLOSE the stale SDK.

        Dropping ``self._sdk`` without awaiting ``aclose`` orphans the old
        ``AsyncOpenAI`` HTTP client (its pooled connections leak until process
        exit). The credential-replacement path in ``get_sdk`` closes the old
        client; the disk-adopt path must do the same.
        """
        cred_file = tmp_path / "auth.json"
        fresh_access = _make_jwt({"exp": time.time() + 3600})
        OpenAISubscription.save(
            OpenAISubscription.Credentials(
                access_token=fresh_access,
                refresh_token=_FRESH_REFRESH,
                account_id=_FAKE_ACCOUNT,
                expires_at=time.time() + 3600,
            ),
            path=cred_file,
        )
        provider = OpenAISubscription(
            access_token=_make_jwt({"exp": 0.0}),  # expired -> triggers adopt
            refresh_token=_STALE_REFRESH,
            account_id=_FAKE_ACCOUNT,
            expires_at=0.0,
        )
        closed: list[bool] = []
        stale_sdk = MagicMock()
        stale_sdk.close = AsyncMock(side_effect=lambda: closed.append(True))
        provider._sdk = stale_sdk
        provider._sdk_token = _make_jwt({"exp": 0.0})

        with patch(
            "sagent.providers.openai.sub.DEFAULT_CREDENTIALS_PATH",
            cred_file,
        ):
            token = await provider._ensure_valid()

        assert token == fresh_access
        assert closed == [True], "adopting a disk token must close the old SDK client"


class TestStreamResponseNotRead:
    """Unread SDK streaming errors must not leak raw httpx2 exceptions."""

    @pytest.mark.anyio
    async def test_create_response_not_read_is_user_facing(self) -> None:
        provider = _make_provider(expires_at=time.time() + 3600)
        sdk = MagicMock()
        sdk.responses = MagicMock()
        sdk.responses.with_raw_response.create = AsyncMock(
            side_effect=httpx2.ResponseNotRead()
        )

        with patch.object(provider, "get_sdk", AsyncMock(return_value=sdk)):
            model = provider.model("gpt-5.5")
            with pytest.raises(StreamingResponseNotReadError) as raised:
                await model.stream(ModelRequest(messages=[UserMessage(text="hi")]))

        assert not isinstance(raised.value, httpx2.ResponseNotRead)
        assert "OpenAI streaming request failed" in str(raised.value)

    @pytest.mark.anyio
    async def test_wrapped_response_not_read_is_user_facing(self) -> None:
        provider = _make_provider(expires_at=time.time() + 3600)
        sdk = MagicMock()
        sdk.responses = MagicMock()
        err = RuntimeError("SDK failed")
        err.__cause__ = httpx2.ResponseNotRead()
        sdk.responses.with_raw_response.create = AsyncMock(side_effect=err)

        with patch.object(provider, "get_sdk", AsyncMock(return_value=sdk)):
            model = provider.model("gpt-5.5")
            with pytest.raises(StreamingResponseNotReadError) as raised:
                await model.stream(ModelRequest(messages=[UserMessage(text="hi")]))

        assert "OpenAI streaming request failed" in str(raised.value)

    def test_response_not_read_context_is_detected(self) -> None:
        err = RuntimeError("SDK failed")
        err.__context__ = httpx2.ResponseNotRead()

        assert find_response_not_read(err) is not None


class TestStreamAuthRetry:
    """Mid-call 401 must trigger ``handle_auth_error`` + one-shot retry.

    Bug: a stale in-memory bearer (rotated server-side between our
    local expiry check and the request arriving) surfaces as
    ``openai.AuthenticationError`` from ``sdk.responses.with_raw_response.create``.
    Without the catch + reload + retry wrapper, the error bubbles
    raw to the user; with it, the runtime reloads disk creds (or
    force-refreshes) and retries once before giving up.
    """

    @pytest.mark.anyio
    async def test_auth_error_triggers_reload_and_retries_once(self) -> None:
        provider = _make_provider(expires_at=time.time() + 3600)
        request = httpx2.Request("POST", "https://chatgpt.com/backend-api/codex")
        response = httpx2.Response(401, request=request)
        auth_err = openai.AuthenticationError(
            "Unauthorized", response=response, body=None
        )
        # Return DIFFERENT SDKs from get_sdk so the test can prove the
        # retry call landed on a freshly-built SDK (with a rotated
        # bearer baked into default_headers), not the cached one.
        sdk_stale = MagicMock()
        sdk_stale.responses = MagicMock()
        sdk_stale.responses.with_raw_response.create = AsyncMock(side_effect=auth_err)
        sdk_fresh = MagicMock()
        sdk_fresh.responses = MagicMock()
        sdk_fresh.responses.with_raw_response.create = AsyncMock(side_effect=auth_err)
        get_sdk = AsyncMock(side_effect=[sdk_stale, sdk_fresh])
        with (
            patch.object(provider, "get_sdk", get_sdk),
            patch.object(provider, "handle_auth_error", AsyncMock()) as ha,
        ):
            model = provider.model("gpt-5.5")
            req = ModelRequest(messages=[UserMessage(text="hi")])
            with pytest.raises(openai.AuthenticationError):
                await model.stream(req)
        ha.assert_awaited_once()
        # Each SDK saw exactly one create call: the original attempt on
        # the stale SDK, the retry on the freshly-fetched one.
        assert sdk_stale.responses.with_raw_response.create.await_count == 1
        assert sdk_fresh.responses.with_raw_response.create.await_count == 1
        assert get_sdk.await_count == 2


class TestRefreshErrors:
    """``_refresh`` must surface auth failures as :class:`AuthRefreshError`.

    Codex's ``auth.openai.com/oauth/token`` endpoint returns 400/401 when
    the refresh token has been rotated, revoked, or expired. The raw
    ``httpx2.HTTPStatusError`` is useless to the user -- it leaks the
    OAuth URL into the terminal. Convert it into a typed, user-facing
    error with actionable text so the renderer can present "Run /login"
    without dumping a traceback.
    """

    @pytest.mark.anyio
    @pytest.mark.parametrize("status", [400, 401])
    async def test_refresh_4xx_raises_auth_refresh_error(self, status: int) -> None:
        """400/401 on the token endpoint -> :class:`AuthRefreshError`."""
        provider = _make_provider(expires_at=0.0)
        request = httpx2.Request("POST", "https://auth.openai.com/oauth/token")
        response = httpx2.Response(status, request=request, text="invalid_grant")
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=response)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)
        with (
            patch(
                "sagent.providers.openai.sub.httpx2.AsyncClient",
                return_value=mock_http,
            ),
            pytest.raises(AuthRefreshError) as excinfo,
        ):
            await provider._refresh()

        msg = str(excinfo.value)
        assert "/login" in msg, (
            f"AuthRefreshError must guide the user to /login; got: {msg!r}"
        )
        assert "HTTPStatusError" not in msg

    @pytest.mark.anyio
    async def test_refresh_overwrites_missing_field_creds_file(
        self, tmp_path: Path
    ) -> None:
        """A creds file with valid JSON but missing fields must not crash refresh.

        ``load`` indexes ``raw["tokens"]["refresh_token"]`` etc., raising
        ``KeyError`` on ``{"tokens": {}}``. ``_refresh`` already holds fresh
        tokens by then and is meant to overwrite the corrupt file, so the
        load-recovery must catch ``KeyError`` like the other adopt sites do, not
        propagate it.
        """
        cred_file = tmp_path / "auth.json"
        cred_file.write_text('{"tokens": {}}', encoding="utf-8")
        provider = _make_provider(expires_at=0.0)
        refreshed_access = _make_jwt({"exp": time.time() + 3600})
        mock_resp = MagicMock()
        mock_resp.json = MagicMock(
            return_value={
                "access_token": refreshed_access,
                "refresh_token": _FRESH_REFRESH,
                "expires_in": 3600,
            },
        )
        mock_resp.status_code = 200
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_resp)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)
        with (
            patch(
                "sagent.providers.openai.sub.httpx2.AsyncClient",
                return_value=mock_http,
            ),
            patch(
                "sagent.providers.openai.sub.DEFAULT_CREDENTIALS_PATH",
                cred_file,
            ),
        ):
            await provider._refresh()  # must not raise KeyError
        assert provider._access_token == refreshed_access
        # The corrupt file was overwritten with a well-formed token set.
        reloaded = OpenAISubscription.load(path=cred_file)
        assert reloaded["access_token"] == refreshed_access

    @pytest.mark.anyio
    @pytest.mark.parametrize("status", [500, 502, 503])
    async def test_refresh_5xx_does_not_raise_auth_refresh_error(
        self, status: int
    ) -> None:
        """Server-side failures bubble as plain ``httpx2.HTTPStatusError``."""
        provider = _make_provider(expires_at=0.0)
        request = httpx2.Request("POST", "https://auth.openai.com/oauth/token")
        response = httpx2.Response(status, request=request, text="upstream down")
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=response)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)
        with (
            patch(
                "sagent.providers.openai.sub.httpx2.AsyncClient",
                return_value=mock_http,
            ),
            pytest.raises(httpx2.HTTPStatusError),
        ):
            await provider._refresh()


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
