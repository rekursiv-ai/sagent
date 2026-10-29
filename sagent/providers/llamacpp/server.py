"""Models in a ``llama-server`` this provider starts, or one already running.

The server starts on the first request, not on lookup, and stops when the
provider closes, or, if it never closes, when it is collected or Python exits. Settings come from keyword arguments, else the environment:
``LLAMA_CPP_MODEL`` (a GGUF file), ``LLAMA_CPP_SERVER``, ``LLAMA_CPP_CONTEXT``,
``LLAMA_CPP_KV``, ``LLAMA_CPP_SPEC_TYPE``, ``LLAMA_CPP_MTP_DRAFT``,
``LLAMA_CPP_EXTRA_ARGS``, and ``LLAMA_CPP_BASE_URL`` for a server already up.

Typical usage example:

  async with LlamaCpp(model_path="~/models/qwen.gguf") as local:
      async for event in local["default"].stream(request):   # Starts the server.
          ...

  running = LlamaCpp(base_url="http://127.0.0.1:8081/v1")
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Protocol, override

import asyncio
import contextlib
import os
import shlex
import shutil
import signal
import socket
import weakref

from sagent.lib.posix.child import Child, finish
from sagent.providers.openai.compat import OpenAICompatible


if TYPE_CHECKING:
    from asyncio.subprocess import Process
    from collections.abc import Sequence

    import httpx2

    from sagent.providers.types import ProviderSharedResources
else:
    from wrapt import lazy_import

    httpx2 = lazy_import("httpx2")


__all__ = [
    "LlamaCpp",
]


class LlamaCpp(OpenAICompatible):
    """Models in a ``llama-server`` process this provider manages.

    Args:
      name: The name the catalog knows this provider by; the section by default.
      resources: What this shares with other providers; ``None`` makes and
        closes its own.
      api_key: The key a running server wants; empty reads
        ``LLAMA_CPP_API_KEY``, else sends ``local``.
      base_url: A server already running; empty reads ``LLAMA_CPP_BASE_URL``,
        else starts one.
      model_path: The GGUF file to serve; empty reads ``LLAMA_CPP_MODEL``.
      server: The ``llama-server`` binary; empty reads ``LLAMA_CPP_SERVER``,
        else searches ``PATH`` and Docker Desktop's bundled one.
      context_tokens: The window to load the model with; ``0`` reads
        ``LLAMA_CPP_CONTEXT``, else 16384.
      kv_cache: The KV cache's quantization; empty reads ``LLAMA_CPP_KV``,
        else ``q4_0``.
      spec_type: The speculative decoder (``--spec-type``); empty reads
        ``LLAMA_CPP_SPEC_TYPE``, else none.
      draft_tokens: The tokens it drafts a step; ``0`` reads
        ``LLAMA_CPP_MTP_DRAFT``, else 3.
      extra_args: More flags; empty reads ``LLAMA_CPP_EXTRA_ARGS``.
      startup_timeout_sec: How long the server may take to come up.

    """

    section: ClassVar[str] = "llamacpp"
    package: ClassVar[str] = "sagent.providers.llamacpp"
    document: ClassVar[str] = "llamacpp.yaml"
    key_variable: ClassVar[str] = "LLAMA_CPP_API_KEY"

    stop_timeout_sec: ClassVar[float] = 10.0
    """How long a stopped server may take to exit before it is killed."""

    def __init__(
        self,
        *,
        name: str = "",
        resources: ProviderSharedResources | None = None,
        api_key: str = "",
        base_url: str = "",
        model_path: str = "",
        server: str = "",
        context_tokens: int = 0,
        kv_cache: str = "",
        spec_type: str = "",
        draft_tokens: int = 0,
        extra_args: Sequence[str] = (),
        startup_timeout_sec: float = 120.0,
    ) -> None:
        env = os.environ
        super().__init__(
            name=name,
            resources=resources,
            api_key=api_key,
            base_url=base_url or env.get("LLAMA_CPP_BASE_URL", ""),
        )
        self._model_path = model_path or env.get("LLAMA_CPP_MODEL", "")
        self._server = server or env.get("LLAMA_CPP_SERVER", "")
        self._context_tokens = context_tokens or int(
            env.get("LLAMA_CPP_CONTEXT", "16384"),
        )
        self._kv_cache = kv_cache or env.get("LLAMA_CPP_KV", "q4_0")
        self._spec_type = spec_type or env.get("LLAMA_CPP_SPEC_TYPE", "")
        self._draft_tokens = draft_tokens or int(env.get("LLAMA_CPP_MTP_DRAFT", "3"))
        self._extra_args = tuple(
            extra_args or shlex.split(env.get("LLAMA_CPP_EXTRA_ARGS", "")),
        )
        self._startup_timeout_sec = startup_timeout_sec
        self._process: Process | None = None
        self._started_url = ""
        self._reached = False
        """Whether the running server named by ``base_url`` has answered once."""
        self._log: _Log = _Log()
        self._reading: asyncio.Task[None] | None = None
        self._stopping: weakref.finalize[[Process], LlamaCpp] | None = None
        self._lock = asyncio.Lock()

    @override
    async def endpoint(self) -> str:
        """Return the server's endpoint, starting the server if it is not up.

        Raises:
          RuntimeError: No model or binary is named, or the server exited or
            did not come up in time; the message carries its log.

        """
        async with self._lock:
            if self._process is not None and self._process.returncode is None:
                return self._started_url
            if self._base_url:
                if not self._reached:
                    await self._ready(self._base_url)
                    self._reached = True
                return self._base_url
            await self._stop()
            return await self._start()

    @override
    async def credentials(self) -> dict[str, str]:
        key = self._api_key or os.environ.get(self.key_variable, "") or "local"
        return {"authorization": f"Bearer {key}"}

    @override
    async def aclose(self) -> None:
        await self._stop()
        await super().aclose()

    async def _start(self) -> str:
        if not self._model_path:
            raise RuntimeError("Name a GGUF file with model_path= or LLAMA_CPP_MODEL.")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = _port(probe)
        self._process = (
            await Child.start(
                *self._argv(port),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        ).process
        # A provider dropped without ``aclose``, or a Python exiting first,
        # would leave the server holding its GPU memory.
        self._stopping = weakref.finalize(self, _terminate, self._process)
        stdout = self._process.stdout
        assert isinstance(stdout, asyncio.StreamReader)
        self._log = _Log()
        # Not a method: a task holding ``self`` would keep a dropped provider,
        # and its server, alive.
        self._reading = asyncio.create_task(_read_log(stdout, into=self._log))
        self._started_url = f"http://127.0.0.1:{port}/v1"
        try:
            await self._ready(self._started_url)
        except BaseException:
            await self._stop()
            raise
        return self._started_url

    def _argv(self, port: int) -> list[str]:
        """Return the command line that serves the model on ``port``."""
        docker = Path("~/.docker/bin/inference/llama-server").expanduser()
        server = (
            self._server
            or shutil.which("llama-server")
            or (str(docker) if docker.exists() else "")
        )
        if not server:
            raise RuntimeError("llama-server was not found; set LLAMA_CPP_SERVER.")
        argv = [
            server,
            "-m",
            str(Path(self._model_path).expanduser()),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        if self._spec_type:
            argv += [
                "--spec-type",
                self._spec_type,
                "--spec-draft-n-max",
                str(self._draft_tokens),
            ]
        return [
            *argv,
            "--cache-type-k",
            self._kv_cache,
            "--cache-type-v",
            self._kv_cache,
            "--parallel",
            "1",
            "-c",
            str(self._context_tokens),
            "-ngl",
            "12",
            "--reasoning",
            "off",
            *self._extra_args,
        ]

    async def _ready(self, url: str) -> None:
        """Wait until the server at ``url`` lists its models."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._startup_timeout_sec
        while loop.time() < deadline:
            process = self._process
            if process is not None and process.returncode is not None:
                # The log explains the exit, and its reader may still be behind.
                assert isinstance(self._reading, asyncio.Task)
                await asyncio.wait({self._reading}, timeout=1.0)
                raise RuntimeError(self._log.failure("llama-server exited"))
            try:
                response = await self.resources.http.get(f"{url}/models", timeout=1.0)
                if response.is_success:
                    return
            except httpx2.TransportError:
                pass
            await asyncio.sleep(0.05)
        raise RuntimeError(self._log.failure("llama-server did not come up"))

    async def _stop(self) -> None:
        """Stop the server: politely, then by force after ``stop_timeout_sec``."""
        process, self._process = self._process, None
        reading, self._reading = self._reading, None
        stopping, self._stopping = self._stopping, None
        if stopping is not None:
            stopping.detach()
        try:
            if process is not None:
                await Child(
                    process=process,
                    group=False,
                    grace_sec=self.stop_timeout_sec,
                ).stop()
        finally:
            if reading is not None:
                reading.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await finish(reading)


def _port(bound: _Bound) -> int:
    """Return the port an IPv4 socket is bound to."""
    match bound.getsockname():
        case (str(), int() as port):
            return port
        case _:
            raise TypeError("The socket is not bound to an IPv4 address.")


class _Bound(Protocol):
    """A bound socket, whose address typeshed leaves untyped: its family decides it."""

    def getsockname(self) -> object:
        """Return the address the socket is bound to."""
        ...


async def _read_log(stdout: asyncio.StreamReader, *, into: _Log) -> None:
    async for line in stdout:
        into.add(line.decode(errors="replace").rstrip())


def _terminate(process: Process) -> None:
    """Ask a server whose provider never closed to exit."""
    if process.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            os.kill(process.pid, signal.SIGTERM)


class _Log:
    """The server's output: its first lines and its last."""

    def __init__(self) -> None:
        self._head: list[str] = []
        self._tail: deque[str] = deque(maxlen=40)

    def add(self, line: str) -> None:
        # llama.cpp names the cause of a failed start (a missing file, a GPU
        # layer count that cannot fit) in its first lines, before the noise.
        if len(self._head) < 20:
            self._head.append(line)
        else:
            self._tail.append(line)

    def failure(self, prefix: str) -> str:
        """Return ``prefix`` and the log, for an error that the log explains."""
        lines = [*self._head, *self._tail]
        return f"{prefix}; its log:\n" + "\n".join(lines) if lines else prefix
