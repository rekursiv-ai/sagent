"""Tests for the managed llama.cpp provider, with a stand-in ``llama-server``.

Each stand-in is a shell script on ``PATH`` that logs a line, records its
command line, and then lives, dies, or ignores being asked to stop; the HTTP it
would serve is a fake transport, so nothing listens and nothing reaches the
network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, override

import asyncio
import gc
import signal
import socket

from treekle import json_dumps

import httpx2
import pytest

from sagent.model.types import Request, Response, TextChunk
from sagent.providers.llamacpp import server
from sagent.providers.llamacpp.server import LlamaCpp
from sagent.providers.resources import SharedResources
from sagent.session.types import AssistantMessage, UserMessage


if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine


_REQUEST = Request(records=(UserMessage(content="hi"),))

_LIVES = (
    'echo "loading model"\nprintf "%s\\n" "$@" > "$FAKE_LLAMA_ARGS"\nexec sleep 30\n'
)
_DIES = 'echo "error: failed to open model"\necho "done"\nexit 1\n'
_STUBBORN = "trap '' TERM\nprintf trapped > \"$FAKE_LLAMA_ARGS\"\nwhile :; do sleep 0.05; done\n"


@pytest.fixture(autouse=True)
def bare(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Clear every llama.cpp variable, and give PATH and HOME nothing to find."""
    for name in (
        "LLAMA_CPP_API_KEY",
        "LLAMA_CPP_BASE_URL",
        "LLAMA_CPP_MODEL",
        "LLAMA_CPP_SERVER",
        "LLAMA_CPP_CONTEXT",
        "LLAMA_CPP_KV",
        "LLAMA_CPP_SPEC_TYPE",
        "LLAMA_CPP_MTP_DRAFT",
        "LLAMA_CPP_EXTRA_ARGS",
    ):
        monkeypatch.delenv(name, raising=False)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("FAKE_LLAMA_ARGS", str(tmp_path / "args.txt"))
    return bin_dir


def _install(directory: Path, body: str, *, name: str = "llama-server") -> Path:
    """Write an executable stand-in for ``llama-server`` into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / name
    script.write_text(f"#!/bin/sh\n{body}")
    script.chmod(0o755)
    return script


def _argv(tmp_path: Path) -> list[str]:
    """Return the command line the stand-in recorded."""
    return (tmp_path / "args.txt").read_text().splitlines()


def _port(argv: list[str]) -> str:
    return argv[argv.index("--port") + 1]


# ---- settings ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_key_is_the_one_given_else_its_variable_else_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert await LlamaCpp().credentials() == {"authorization": "Bearer local"}
    monkeypatch.setenv("LLAMA_CPP_API_KEY", "sk-env")
    assert await LlamaCpp().credentials() == {"authorization": "Bearer sk-env"}
    assert await LlamaCpp(api_key="sk-given").credentials() == {
        "authorization": "Bearer sk-given",
    }


def test_the_provider_serves_its_sections_models() -> None:
    local = LlamaCpp()
    assert local.name == "llamacpp"
    assert list(local) == ["qwen-27b-12gb-3.6", "qwen-27b-mtp-64k-3.6", "local-0.0"]
    assert local["default"].capability.wire_model_id == "qwen3.6-27b-12gb"


def test_a_provider_keeps_its_name_and_reads_unset_settings_as_empty() -> None:
    assert LlamaCpp(name="gpu").name == "gpu"
    local = LlamaCpp()
    unset = (local._base_url, local._model_path, local._server, local._spec_type)
    assert unset == ("", "", "", "")
    assert (local._started_url, local._reached) == ("", False)
    assert local._startup_timeout_sec == 120.0


@pytest.mark.asyncio
async def test_count_is_estimated_without_starting_the_server(bare: Path) -> None:
    _install(bare, _DIES)
    local = LlamaCpp(model_path="m.gguf")
    assert await local["default"].count(_REQUEST) == 0
    assert local._process is None


@pytest.mark.asyncio
async def test_the_server_starts_on_the_first_request_with_the_settings_given(
    bare: Path,
    tmp_path: Path,
) -> None:
    _install(bare, _LIVES)
    fake = _Http(ready=tmp_path / "args.txt")
    async with _Served(fake) as shared:
        local = LlamaCpp(
            resources=shared,
            model_path="~/models/qwen.gguf",
            context_tokens=8_192,
            kv_cache="q8_0",
            spec_type="mtp",
            draft_tokens=2,
            extra_args=("--flash-attn", "on"),
        )
        model = local["default"]
        assert local._process is None
        events = [event async for event in model.stream(_REQUEST)]
        process = local._process
        assert process is not None
        # A second request reaches the same server, with no second start.
        [event async for event in model.stream(_REQUEST)]
        assert local._process is process
        stopping = local._stopping
        assert stopping is not None
        assert stopping.alive
        await local.aclose()
        # Closed, it no longer needs stopping when collected.
        assert not stopping.alive
    argv = _argv(tmp_path)
    port = _port(argv)
    assert argv == [
        "-m",
        str(tmp_path / "home" / "models" / "qwen.gguf"),
        "--host",
        "127.0.0.1",
        "--port",
        port,
        "--spec-type",
        "mtp",
        "--spec-draft-n-max",
        "2",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "--parallel",
        "1",
        "-c",
        "8192",
        "-ngl",
        "12",
        "--reasoning",
        "off",
        "--flash-attn",
        "on",
    ]
    urls = [str(request.url) for request in fake.requests]
    # Asked for its models until it answered, then sent both requests.
    assert set(urls[:-2]) == {f"http://127.0.0.1:{port}/v1/models"}
    assert urls[-2:] == [f"http://127.0.0.1:{port}/v1/chat/completions"] * 2
    assert fake.requests[-1].headers["authorization"] == "Bearer local"
    assert events[0] == TextChunk(text="Hi")
    response = events[-1]
    assert isinstance(response, Response)
    assert response.records == (AssistantMessage(content="Hi"),)
    # Closed, it stopped the server.
    assert process.returncode == -signal.SIGTERM
    assert local._process is None


@pytest.mark.asyncio
async def test_unnamed_settings_come_from_the_environment(
    bare: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script = _install(tmp_path / "elsewhere", _LIVES, name="my-llama")
    monkeypatch.setenv("LLAMA_CPP_MODEL", "/m/env.gguf")
    monkeypatch.setenv("LLAMA_CPP_SERVER", str(script))
    monkeypatch.setenv("LLAMA_CPP_CONTEXT", "4096")
    monkeypatch.setenv("LLAMA_CPP_KV", "f16")
    monkeypatch.setenv("LLAMA_CPP_SPEC_TYPE", "draft")
    monkeypatch.setenv("LLAMA_CPP_MTP_DRAFT", "5")
    monkeypatch.setenv("LLAMA_CPP_EXTRA_ARGS", "--threads '4'")
    del bare
    async with (
        _Served(_Http(ready=tmp_path / "args.txt")) as shared,
        LlamaCpp(resources=shared) as local,
    ):
        await local.endpoint()
    argv = _argv(tmp_path)
    assert argv[:2] == ["-m", "/m/env.gguf"]
    assert argv[6:] == [
        "--spec-type",
        "draft",
        "--spec-draft-n-max",
        "5",
        "--cache-type-k",
        "f16",
        "--cache-type-v",
        "f16",
        "--parallel",
        "1",
        "-c",
        "4096",
        "-ngl",
        "12",
        "--reasoning",
        "off",
        "--threads",
        "4",
    ]


@pytest.mark.asyncio
async def test_with_nothing_named_the_defaults_apply(
    bare: Path,
    tmp_path: Path,
) -> None:
    _install(bare, _LIVES)
    async with (
        _Served(_Http(ready=tmp_path / "args.txt")) as shared,
        LlamaCpp(resources=shared, model_path="/m.gguf") as local,
    ):
        await local.endpoint()
    argv = _argv(tmp_path)
    assert argv[6:] == [
        "--cache-type-k",
        "q4_0",
        "--cache-type-v",
        "q4_0",
        "--parallel",
        "1",
        "-c",
        "16384",
        "-ngl",
        "12",
        "--reasoning",
        "off",
    ]


@pytest.mark.asyncio
async def test_docker_desktops_server_is_found_when_none_is_on_path(
    tmp_path: Path,
) -> None:
    _install(tmp_path / "home" / ".docker" / "bin" / "inference", _LIVES)
    async with (
        _Served(_Http(ready=tmp_path / "args.txt")) as shared,
        LlamaCpp(resources=shared, model_path="/m.gguf") as local,
    ):
        await local.endpoint()
    assert _argv(tmp_path)[:2] == ["-m", "/m.gguf"]


# ---- what fails to start -----------------------------------------------------


@pytest.mark.asyncio
async def test_no_model_named_is_an_error_before_anything_starts() -> None:
    with pytest.raises(RuntimeError, match="LLAMA_CPP_MODEL"):
        await LlamaCpp().endpoint()


@pytest.mark.asyncio
async def test_no_server_found_says_how_to_name_one() -> None:
    with pytest.raises(RuntimeError, match="set LLAMA_CPP_SERVER"):
        await LlamaCpp(model_path="/m.gguf").endpoint()


@pytest.mark.asyncio
async def test_a_server_that_exits_reports_its_log(bare: Path) -> None:
    _install(bare, _DIES)
    fake = _Http(up=False)
    async with (
        _Served(fake) as shared,
        LlamaCpp(resources=shared, model_path="/m.gguf") as local,
    ):
        with pytest.raises(RuntimeError) as raised:
            await local.endpoint()
        assert local._process is None
    assert str(raised.value) == (
        "llama-server exited; its log:\nerror: failed to open model\ndone"
    )


@pytest.mark.asyncio
async def test_a_server_that_never_answers_is_stopped(
    bare: Path,
    tmp_path: Path,
) -> None:
    """Its log so far explains the failure; refused only once it logged, it has one."""
    _install(bare, _LIVES)
    async with _Served(_Http(up=False, ready=tmp_path / "args.txt")) as shared:
        local = LlamaCpp(
            resources=shared,
            model_path="/m.gguf",
            startup_timeout_sec=0.01,
        )
        with pytest.raises(
            RuntimeError,
            match=r"did not come up; its log:\nloading model",
        ):
            await local.endpoint()
        assert local._process is None
        await local.aclose()


@pytest.mark.asyncio
async def test_a_server_that_died_is_started_again(bare: Path) -> None:
    _install(bare, _LIVES)
    async with (
        _Served(_Http()) as shared,
        LlamaCpp(resources=shared, model_path="/m.gguf") as local,
    ):
        await local.endpoint()
        process = local._process
        assert process is not None
        process.kill()
        await process.wait()
        await local.endpoint()
        again = local._process
        assert again is not None
        assert again is not process
        assert again.returncode is None


# ---- stopping ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_server_that_ignores_the_request_to_stop_is_killed(
    bare: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install(bare, _STUBBORN)
    monkeypatch.setattr(LlamaCpp, "stop_timeout_sec", 0.01)
    async with _Served(_Http(ready=tmp_path / "args.txt")) as shared:
        local = LlamaCpp(resources=shared, model_path="/m.gguf")
        await local.endpoint()
        process = local._process
        assert process is not None
        await local.aclose()
    assert process.returncode == -signal.SIGKILL


@pytest.mark.asyncio
async def test_a_close_cancelled_midway_still_stops_the_server(
    bare: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Cancelled while it waits out the grace, the stop still kills the server."""
    _install(bare, _STUBBORN)
    monkeypatch.setattr(LlamaCpp, "stop_timeout_sec", 0.05)
    async with _Served(_Http(ready=tmp_path / "args.txt")) as shared:
        local = LlamaCpp(resources=shared, model_path="/m.gguf")
        await local.endpoint()
        process = local._process
        reading = local._reading
        assert process is not None
        assert reading is not None
        closing = asyncio.create_task(local.aclose())
        await asyncio.sleep(0.01)
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(closing, timeout=5)
    assert process.returncode == -signal.SIGKILL
    assert reading.done()


@pytest.mark.asyncio
async def test_a_provider_dropped_without_closing_stops_its_server(bare: Path) -> None:
    _install(bare, _LIVES)
    async with _Served(_Http()) as shared:
        local = LlamaCpp(resources=shared, model_path="/m.gguf")
        await local.endpoint()
        process = local._process
        assert process is not None
        del local
        gc.collect()
        assert await asyncio.wait_for(process.wait(), timeout=5.0) == -signal.SIGTERM


@pytest.mark.asyncio
async def test_a_server_already_gone_is_left_alone() -> None:
    gone = await asyncio.create_subprocess_exec("true")
    assert await gone.wait() == 0
    # Its pid may be another process's by now, so it is not signaled.
    server._terminate(gone)
    assert gone.returncode == 0


# ---- a server already running ------------------------------------------------


@pytest.mark.asyncio
async def test_a_running_server_is_checked_once_and_never_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLAMA_CPP_BASE_URL", "http://gpu-box:8081/v1")
    fake = _Http()
    async with _Served(fake) as shared, LlamaCpp(resources=shared) as local:
        assert await local.endpoint() == "http://gpu-box:8081/v1"
        assert await local.endpoint() == "http://gpu-box:8081/v1"
        assert local._process is None
    assert [str(request.url) for request in fake.requests] == [
        "http://gpu-box:8081/v1/models",
    ]
    async with _Served(_Http(up=False)) as shared:
        down = LlamaCpp(
            resources=shared,
            base_url="http://down:1/v1",
            startup_timeout_sec=0.01,
        )
        with pytest.raises(RuntimeError, match="did not come up"):
            await down.endpoint()


@pytest.mark.asyncio
async def test_a_running_server_that_answers_an_error_is_not_up() -> None:
    fake = _Http(statuses=[503, 200])
    async with _Served(fake) as shared:
        local = LlamaCpp(resources=shared, base_url="http://box/v1")
        assert await local.endpoint() == "http://box/v1"
    assert len(fake.requests) == 2


def test_a_port_is_read_only_from_an_ipv4_socket(tmp_path: Path) -> None:
    with socket.socket(socket.AF_UNIX) as unix:
        unix.bind(str(tmp_path / "s"))
        with pytest.raises(TypeError, match="IPv4"):
            server._port(unix)
    with socket.socket() as ipv4:
        ipv4.bind(("127.0.0.1", 0))
        assert 0 < server._port(ipv4) < 65_536


# ---- the log -----------------------------------------------------------------


def test_the_log_keeps_its_first_lines_and_its_last() -> None:
    log = server._Log()
    assert log.failure("failed") == "failed"
    for i in range(100):
        log.add(f"line {i}")
    kept = [f"line {i}" for i in (*range(20), *range(60, 100))]
    assert log.failure("failed") == "failed; its log:\n" + "\n".join(kept)


@dataclass(slots=True, kw_only=True)
class _Http:
    """Answers as a llama-server would: its models, then a short reply."""

    up: bool = True
    ready: Path | None = None
    """A file the server writes as it starts; until then a request waits."""

    statuses: list[int] = field(default_factory=list[int])
    requests: list[httpx2.Request] = field(default_factory=list[httpx2.Request])

    async def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if request.url.path.endswith("/models"):
            if self.statuses:
                return httpx2.Response(self.statuses.pop(0))
            if self.ready is not None:
                await _written(self.ready)
            if not self.up:
                raise httpx2.ConnectError("Connection refused")
            return httpx2.Response(200, content=b'{"data": []}')
        chunks: tuple[dict[str, object], ...] = (
            {"choices": [{"delta": {"content": "Hi"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        )
        body = "".join(f"data: {json_dumps(chunk)}\n\n" for chunk in chunks)
        return httpx2.Response(200, content=f"{body}data: [DONE]\n\n".encode())


async def _written(path: Path) -> None:
    """Wait until the stand-in has written ``path``."""
    async with asyncio.timeout(5):
        while True:
            if _has_text(path):
                return
            await asyncio.sleep(0.005)


def _has_text(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


class _Served(SharedResources):
    """Shared resources whose HTTP client answers from ``handler``, never the network."""

    def __init__(
        self,
        handler: Callable[[httpx2.Request], Coroutine[None, None, httpx2.Response]],
    ) -> None:
        super().__init__()
        self._client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))

    @property
    @override
    def http(self) -> httpx2.AsyncClient:
        return self._client

    @override
    async def aclose(self) -> None:
        await super().aclose()
        await self._client.aclose()


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
