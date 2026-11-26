"""Unit tests for MCP credentials, sign-in callbacks, and shared renewal."""

from __future__ import annotations

from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict
from unittest.mock import ANY, AsyncMock, MagicMock, Mock, call

import asyncio
import time

from mcp.client.auth import OAuthClientProvider
from mcp.client.session_group import StreamableHttpParameters
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
from pydantic import AnyUrl
from treekle import ReadError, json_loads

import httpx2
import pytest

from sagent.lib.userdirs import config_dir
from sagent.model.exceptions import AuthError
from sagent.providers.testing import refuse_browsers
from sagent.tools import mcp_auth
from sagent.tools.mcp import McpServer
from sagent.tools.mcp_auth import McpTokenStorage


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


@pytest.fixture(autouse=True)
def no_real_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse the user's browser; sign-in tests supply their own boundary fake."""
    refuse_browsers(monkeypatch, mcp_auth)


_WEB = McpServer(
    name="web",
    parameters=StreamableHttpParameters(url="https://h/mcp"),
    source="/p/mcp.json",
)
_OTHER = McpServer(name="other", parameters=StreamableHttpParameters(url="https://o"))


def _storage(server: McpServer = _WEB) -> McpTokenStorage:
    return McpTokenStorage(server.key, source=server.source)


def _kept(
    access: str = "old",
    *,
    expires_at: float | None = None,
) -> mcp_auth._MemoryStorage:
    return mcp_auth._MemoryStorage(
        tokens=OAuthToken(
            access_token=access,
            token_type="Bearer",  # noqa: S106 -- The scheme's name, no secret.
            refresh_token=f"r-{access}",
        ),
        client=OAuthClientInformationFull(
            client_id=f"c-{access}",
            redirect_uris=[AnyUrl("http://127.0.0.1:1/callback")],
        ),
        expires_at=expires_at,
    )


async def _session() -> mcp_auth._Shared:
    shared = await mcp_auth.session_auth(
        "web",
        "https://h/mcp",
        key=_WEB.key,
        source=_WEB.source,
    )
    assert isinstance(shared, mcp_auth._Shared)
    return shared


@pytest.mark.asyncio
async def test_tokens_and_client_are_kept_per_server_in_a_private_file() -> None:
    storage = _storage()
    assert await storage.get_tokens() is None
    assert await storage.get_client_info() is None
    kept = _kept()
    assert kept.tokens is not None
    assert kept.client is not None
    await storage.set_tokens(kept.tokens)
    await storage.set_client_info(kept.client)
    again = _storage()
    assert await again.get_tokens() == kept.tokens
    assert await again.get_client_info() == kept.client
    anonymous = McpTokenStorage(_OTHER.key)
    assert anonymous.source == ""
    assert await anonymous.get_tokens() is None
    plain = json_loads(storage.path.read_text(), _Saved)
    assert plain["source"] == "/p/mcp.json"
    assert plain["tokens"] == {
        "access_token": "old",
        "token_type": "Bearer",
        "refresh_token": "r-old",
    }
    assert all(value is not None for value in plain["client"].values())
    assert storage.path.stat().st_mode & 0o777 == 0o600
    assert storage.path.parent.stat().st_mode & 0o777 == 0o700


@pytest.mark.asyncio
async def test_malformed_saved_credentials_fail_at_the_json_boundary() -> None:
    storage = _storage()
    storage.path.parent.mkdir(parents=True)
    storage.path.write_text('{"tokens": 5}', encoding="utf-8")
    with pytest.raises(ReadError):
        await storage.read()


@pytest.mark.asyncio
async def test_saved_client_names_are_utf8_even_with_another_locale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage()
    storage.path.parent.mkdir(parents=True)
    storage.path.write_bytes(
        '{"client":{"client_id":"c","client_name":"Café",'
        '"redirect_uris":["http://127.0.0.1/callback"]}}'.encode(),
    )
    original = Path.read_text

    def read(
        path: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        return original(path, encoding=encoding or "latin-1", errors=errors)

    monkeypatch.setattr(Path, "read_text", read)
    client = await storage.get_client_info()
    assert client is not None
    assert client.client_name == "Café"


@pytest.mark.asyncio
async def test_sign_ins_are_the_files_that_record_the_servers_source() -> None:
    kept = _kept()
    assert kept.tokens is not None
    elsewhere = McpServer(
        name="web",
        parameters=StreamableHttpParameters(url="https://h/mcp"),
        source="/q/mcp.json",
    )
    for server in (_WEB, _OTHER, elsewhere):
        await _storage(server).set_tokens(kept.tokens)
    folder = config_dir() / "rekursiv-ai" / "secrets" / "mcp"
    assert mcp_auth.sign_ins("/p/mcp.json") == {_WEB.key: folder / f"{_WEB.key}.json"}
    assert mcp_auth.sign_ins("/q/mcp.json") == {
        elsewhere.key: folder / f"{elsewhere.key}.json",
    }
    assert mcp_auth.sign_ins("/r/mcp.json") == {}


@pytest.mark.asyncio
async def test_two_changes_at_once_keep_both() -> None:
    storage = _storage()
    kept = _kept()
    assert kept.tokens is not None
    assert kept.client is not None
    await asyncio.gather(
        storage.set_tokens(kept.tokens),
        storage.set_client_info(kept.client),
    )
    assert await storage.get_tokens() == kept.tokens
    assert await storage.get_client_info() == kept.client


@pytest.mark.asyncio
async def test_a_change_waits_for_another_task_holding_the_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage()
    kept = _kept()
    assert kept.tokens is not None
    gate, contending = asyncio.Lock(), asyncio.Event()
    calls: list[Path] = []

    @asynccontextmanager
    async def lock(path: Path) -> AsyncGenerator[None]:
        calls.append(path)
        if gate.locked():
            contending.set()
        async with gate:
            yield

    monkeypatch.setattr(mcp_auth, "file_lock", lock)
    async with asyncio.TaskGroup() as tasks, storage.locked():
        writing = tasks.create_task(storage.set_tokens(kept.tokens))
        async with asyncio.timeout(1):
            await contending.wait()
        assert not writing.done()
        async with storage.locked():
            assert await storage.get_tokens() is None
    assert await storage.get_tokens() == kept.tokens
    assert calls == [storage.path.with_suffix(".json.lock")] * 2


@pytest.mark.asyncio
async def test_a_token_expiry_is_saved_and_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(time, "time", lambda: 100.0)
    kept = _kept()
    assert kept.tokens is not None
    expiring = kept.tokens.model_copy(update={"expires_in": 60})
    pending = mcp_auth._MemoryStorage()
    await pending.set_tokens(expiring)
    assert pending.expires_at == 160.0
    assert kept.client is not None
    await pending.set_client_info(kept.client)
    assert await pending.get_tokens() == expiring
    assert await pending.get_client_info() == kept.client
    await _storage().save(pending)
    shared = await _session()
    assert shared.auth.context.token_expiry_time == 160.0
    await _storage().set_tokens(kept.tokens)
    assert (await _storage().read()).expires_at is None


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["tokens", "client", "both"])
async def test_a_session_without_complete_credentials_never_registers(
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    kept = _kept()
    if missing in {"tokens", "both"}:
        kept.tokens = None
    if missing in {"client", "both"}:
        kept.client = None
    await _storage().save(kept)
    build = Mock()
    monkeypatch.setattr(mcp_auth, "OAuthClientProvider", build)
    needed = await mcp_auth.session_auth(
        "web",
        "https://h/mcp",
        key=_WEB.key,
        source=_WEB.source,
    )
    assert isinstance(needed, mcp_auth._SignInNeeded)
    build.assert_not_called()
    request = httpx2.Request("GET", "https://h/mcp")
    flow = needed.auth_flow(request)
    assert next(flow) is request
    with pytest.raises(AuthError, match="run `sagent mcp login web`"):
        flow.send(httpx2.Response(401))


@pytest.mark.asyncio
async def test_a_saved_session_wires_its_url_credentials_and_refusal_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kept = _kept(expires_at=160.0)
    await _storage().save(kept)
    needed = mcp_auth._SignInNeeded(name="web")
    refusal = Mock(return_value=needed)
    provider = Mock(wraps=OAuthClientProvider)
    monkeypatch.setattr(mcp_auth, "_SignInNeeded", refusal)
    monkeypatch.setattr(mcp_auth, "OAuthClientProvider", provider)
    shared = await _session()
    refusal.assert_called_once_with(name="web")
    provider.assert_called_once_with(
        server_url="https://h/mcp",
        client_metadata=OAuthClientMetadata(
            client_name="sagent",
            redirect_uris=[AnyUrl("http://127.0.0.1/callback")],
            token_endpoint_auth_method="none",  # noqa: S106 -- OAuth public-client method.
        ),
        storage=shared.storage,
        redirect_handler=needed.redirect,
        callback_handler=needed.callback,
    )
    assert shared.storage.path == _storage().path
    assert shared.storage.source == "/p/mcp.json"
    assert shared.auth.context.token_expiry_time == 160.0


@pytest.mark.asyncio
async def test_a_session_never_asks_for_a_sign_in_or_its_code() -> None:
    needed = mcp_auth._SignInNeeded(name="web")
    request = httpx2.Request("GET", "https://h/mcp")
    flow = needed.auth_flow(request)
    assert next(flow) is request
    with pytest.raises(StopIteration):
        flow.send(httpx2.Response(200))
    with pytest.raises(AuthError, match="MCP server 'web' needs a sign-in"):
        await needed.redirect("https://authority/authorize")
    with pytest.raises(AuthError, match="MCP server 'web' needs a sign-in"):
        await needed.callback()


@pytest.mark.asyncio
@pytest.mark.parametrize("headless", [False, True])
async def test_a_sign_in_records_its_state_and_opens_only_a_local_browser(
    monkeypatch: pytest.MonkeyPatch,
    headless: bool,
) -> None:
    opened = Mock()
    monkeypatch.setattr(mcp_auth, "open_browser", opened)
    shown: list[str] = []
    sign_in = mcp_auth._SignIn(show=shown.append, headless=headless, wait_sec=0)
    url = "https://authority/authorize?state=this-sign-in"
    await sign_in.redirect(url)
    assert sign_in.state == "this-sign-in"
    assert shown == [f"Sign in to the MCP server in your browser:\n\n  {url}\n"]
    assert opened.call_args_list == ([] if headless else [call(url)])


@pytest.mark.asyncio
async def test_the_browser_callback_wins_without_asking_for_a_paste(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paste = AsyncMock()
    monkeypatch.setattr(mcp_auth, "read_pasted_code", paste)
    sign_in = mcp_auth._SignIn(
        show=list[str]().append,
        headless=False,
        wait_sec=1,
        state="s",
    )
    sign_in.code.set_result("browser-code")
    result = await sign_in.callback()
    assert (result.code, result.state) == ("browser-code", "s")
    paste.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("headless", [False, True])
async def test_headless_or_missing_browser_callbacks_use_the_paste_boundary(
    monkeypatch: pytest.MonkeyPatch,
    headless: bool,
) -> None:
    paste = AsyncMock(return_value="pasted-code")
    monkeypatch.setattr(mcp_auth, "read_pasted_code", paste)
    show = list[str]().append
    sign_in = mcp_auth._SignIn(show=show, headless=headless, wait_sec=0, state="s")
    result = await sign_in.callback()
    assert (result.code, result.state) == ("pasted-code", "s")
    paste.assert_awaited_once_with(state="s", show=show, browser_tried=not headless)
    assert not sign_in.code.cancelled()


@pytest.mark.asyncio
async def test_a_failed_paste_propagates_without_a_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paste = AsyncMock(side_effect=AuthError("invalid pasted address"))
    monkeypatch.setattr(mcp_auth, "read_pasted_code", paste)
    sign_in = mcp_auth._SignIn(
        show=list[str]().append,
        headless=True,
        wait_sec=0,
        state="s",
    )
    with pytest.raises(AuthError, match="invalid pasted address"):
        await sign_in.callback()
    assert not sign_in.code.done()


@pytest.mark.asyncio
async def test_the_loopback_answer_passes_the_sign_ins_state_and_future(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redirect = AsyncMock()
    monkeypatch.setattr(mcp_auth, "redirected", redirect)
    sign_in = mcp_auth._SignIn(
        show=list[str]().append,
        headless=False,
        wait_sec=1,
        state="s",
    )
    reader, writer = Mock(spec=asyncio.StreamReader), Mock(spec=asyncio.StreamWriter)
    await sign_in.answer(reader, writer)
    redirect.assert_awaited_once_with(
        reader,
        writer,
        path="/callback",
        state="s",
        code=sign_in.code,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("succeeds", [True, False])
async def test_a_login_publishes_its_client_and_tokens_only_after_successful_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    succeeds: bool,
) -> None:
    storage = _storage()
    old, new = _kept(), _kept("new", expires_at=160.0)
    await storage.save(old)
    pending: list[mcp_auth._MemoryStorage] = []
    metadata: list[OAuthClientMetadata] = []
    auth = Mock()

    def provider(**options: object) -> Mock:
        assert options["server_url"] == "https://h/mcp"
        memory, info = options["storage"], options["client_metadata"]
        assert isinstance(memory, mcp_auth._MemoryStorage)
        assert isinstance(info, OAuthClientMetadata)
        assert memory == mcp_auth._MemoryStorage()
        pending.append(memory)
        metadata.append(info)
        assert callable(options["redirect_handler"])
        assert callable(options["callback_handler"])
        return auth

    async def exited(*exc: object) -> None:
        del exc
        assert await storage.read() == old

    finish = AsyncMock(side_effect=exited)
    listener = MagicMock(
        sockets=[Mock(getsockname=Mock(return_value=("127.0.0.1", 4321)))],
        __aenter__=AsyncMock(),
        __aexit__=finish,
    )
    start = AsyncMock(return_value=listener)
    monkeypatch.setattr(asyncio, "start_server", start)
    monkeypatch.setattr(mcp_auth, "OAuthClientProvider", provider)
    with nullcontext() if succeeds else pytest.raises(AuthError, match="abandoned"):
        async with mcp_auth.login_auth(
            "https://h/mcp",
            key=_WEB.key,
            source=_WEB.source,
            show=list[str]().append,
            headless=True,
        ) as yielded:
            assert yielded is auth
            (memory,) = pending
            memory.tokens, memory.client, memory.expires_at = (
                new.tokens,
                new.client,
                new.expires_at,
            )
            if not succeeds:
                raise AuthError("abandoned")
    start.assert_awaited_once_with(ANY, "127.0.0.1")
    finish.assert_awaited_once()
    (client,) = metadata
    assert client.client_name == "sagent"
    assert client.token_endpoint_auth_method == "none"  # noqa: S105 -- A method's name.
    assert [str(uri) for uri in client.redirect_uris or []] == [
        "http://127.0.0.1:4321/callback",
    ]
    assert await storage.read() == (new if succeeds else old)


def _shared(
    kept: mcp_auth._MemoryStorage,
    *,
    valid: bool = True,
) -> mcp_auth._Shared:
    return mcp_auth._Shared(
        auth=Mock(
            spec=OAuthClientProvider,
            context=Mock(
                current_tokens=None,
                token_expiry_time=None,
                is_token_valid=Mock(return_value=valid),
            ),
        ),
        storage=Mock(
            spec=McpTokenStorage,
            read=AsyncMock(return_value=kept),
            locked=Mock(return_value=nullcontext()),
        ),
    )


@pytest.mark.asyncio
async def test_a_session_refused_after_another_renewed_takes_the_renewal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(time, "time", lambda: 100.0)
    old, new = _kept(), _kept("new", expires_at=200.0)
    shared = _shared(new)
    context = shared.auth.context
    context.current_tokens = old.tokens
    context.token_expiry_time = 90.0
    entering, leaving = AsyncMock(), AsyncMock(return_value=False)
    locked = Mock(
        return_value=MagicMock(__aenter__=entering, __aexit__=leaving),
    )

    def admitted() -> mcp_auth._MemoryStorage:
        assert entering.await_count == leaving.await_count + 1
        return new

    reading = AsyncMock(side_effect=admitted)
    monkeypatch.setattr(shared.storage, "locked", locked)
    monkeypatch.setattr(shared.storage, "read", reading)
    async with shared._renewing(old.tokens):
        assert context.current_tokens == new.tokens
        assert context.token_expiry_time == 200.0
    async with shared._renewing(new.tokens):
        assert context.current_tokens == new.tokens
        assert context.token_expiry_time == 99.0
    assert locked.call_args_list == [call(), call()]
    assert reading.await_args_list == [call(), call()]
    assert entering.await_count == leaving.await_count == 2


@pytest.mark.asyncio
async def test_empty_storage_does_not_replace_the_sdks_current_tokens() -> None:
    kept = _kept()
    shared = _shared(mcp_auth._MemoryStorage())
    shared.auth.context.current_tokens = kept.tokens
    shared.auth.context.token_expiry_time = 200.0
    async with shared._renewing(None):
        assert shared.auth.context.current_tokens == kept.tokens
        assert shared.auth.context.token_expiry_time == 200.0


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 401])
async def test_a_refused_request_closes_the_sdk_flow_before_one_expired_retry(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
) -> None:
    monkeypatch.setattr(time, "time", lambda: 100.0)
    kept = _kept(expires_at=200.0)
    shared = _shared(kept)
    shared.auth.context.current_tokens = kept.tokens
    shared.auth.context.token_expiry_time = 200.0
    seen: list[tuple[float | None, int]] = []
    closed: list[bool] = []
    answered: list[int] = []

    async def sdk(
        request: httpx2.Request,
    ) -> AsyncGenerator[httpx2.Request, httpx2.Response]:
        seen.append((shared.auth.context.token_expiry_time, len(closed)))
        try:
            response = yield request
            answered.append(response.status_code)
        finally:
            closed.append(True)

    monkeypatch.setattr(shared.auth, "async_auth_flow", sdk)
    request = httpx2.Request("GET", "https://h/mcp")
    flow = shared.async_auth_flow(request)
    try:
        assert await anext(flow) is request
        sent = await flow.asend(httpx2.Response(401))  # codespell:ignore asend
        assert sent is request
        with pytest.raises(StopAsyncIteration):
            await flow.asend(httpx2.Response(status))  # codespell:ignore asend
    finally:
        await flow.aclose()
    assert seen == [(200.0, 0), (99.0, 1)]
    assert closed == [True, True]
    assert answered == [status]


@pytest.mark.asyncio
async def test_expiry_reloads_storage_before_forwarding_sdk_exchange_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(time, "time", lambda: 100.0)
    old, new = _kept(expires_at=99.0), _kept("new", expires_at=200.0)
    shared = _shared(new, valid=False)
    shared.auth.context.current_tokens = old.tokens
    shared.auth.context.token_expiry_time = 99.0
    exchange = httpx2.Request("POST", "https://authority/token")
    answered: list[int] = []
    closed = Mock()

    async def sdk(
        request: httpx2.Request,
    ) -> AsyncGenerator[httpx2.Request, httpx2.Response]:
        assert shared.auth.context.current_tokens == new.tokens
        try:
            response = yield exchange
            answered.append(response.status_code)
            response = yield request
            answered.append(response.status_code)
        finally:
            closed()

    monkeypatch.setattr(shared.auth, "async_auth_flow", sdk)
    request = httpx2.Request("GET", "https://h/mcp")
    flow = shared.async_auth_flow(request)
    try:
        assert await anext(flow) is exchange
        sent = await flow.asend(httpx2.Response(401))  # codespell:ignore asend
        assert sent is request
        with pytest.raises(StopAsyncIteration):
            await flow.asend(httpx2.Response(200))  # codespell:ignore asend
    finally:
        await flow.aclose()
    assert answered == [401, 200]
    closed.assert_called_once_with()


class TestSignIns:
    """Exercise directory discovery without an OAuth server."""

    def test_only_json_files_recording_the_requested_source_are_returned(self) -> None:
        directory = config_dir() / "rekursiv-ai" / "secrets" / "mcp"
        assert mcp_auth.sign_ins("/p/mcp.json") == {}
        directory.mkdir(parents=True)
        matching = directory / "mine.json"
        matching.write_text('{"source": "/p/mcp.json"}', encoding="utf-8")
        (directory / "other.json").write_text('{"source": "/other"}', encoding="utf-8")
        (directory / "unsourced.json").write_text("{}", encoding="utf-8")
        (directory / "ignored.txt").write_text("not json", encoding="utf-8")
        assert mcp_auth.sign_ins("/p/mcp.json") == {"mine": matching}

    def test_a_listed_file_disappearing_does_not_hide_other_sign_ins(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        directory = config_dir() / "rekursiv-ai" / "secrets" / "mcp"
        directory.mkdir(parents=True)
        vanished, kept = directory / "a.json", directory / "b.json"
        vanished.touch()
        kept.touch()
        load = Mock(side_effect=[FileNotFoundError, {"source": "/p/mcp.json"}])
        monkeypatch.setattr(mcp_auth, "_load", load)
        assert mcp_auth.sign_ins("/p/mcp.json") == {"b": kept}
        assert load.call_args_list == [call(vanished), call(kept)]


class _Saved(TypedDict):
    """A completed credential write, with both client and tokens present."""

    source: str
    tokens: dict[str, object]
    client: dict[str, object]


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
