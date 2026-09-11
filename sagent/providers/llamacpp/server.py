"""Managed llama.cpp provider."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import ClassVar, Final, Self, override

import atexit
import os
import shlex
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request

from sagent.catalog import llamacpp as llamacpp_catalog
from sagent.providers.openai.compat import (
    OpenAICompat,
    OpenAICompatModel,
)
from sagent.types.capability import ModelCapability


class LlamaCpp(OpenAICompat):
    """OpenAI-compatible provider backed by a managed llama-server process."""

    DEFAULT_MODEL: ClassVar[str] = "qwen3.6-27b-12gb"
    DEFAULT_UTILITY_MODEL: ClassVar[str] = "qwen3.6-27b-12gb"
    ENV_VAR: ClassVar[str] = "LLAMA_CPP_API_KEY"
    BASE_URL: ClassVar[str] = "http://127.0.0.1:8081/v1"
    CAPABILITIES: ClassVar[Mapping[str, ModelCapability]] = llamacpp_catalog.models()
    """Per-model capability; transport limits live on ``TRANSPORT``."""

    def __init__(
        self,
        *,
        api_key: str = "local",
        base_url: str | None = None,
        model_path: str = "",
        server_bin: str = "",
        context_tokens: int = 16_384,
        kv_cache: str = "q4_0",
        spec_type: str = "",
        mtp_draft: int = 3,
        extra_args: Sequence[str] = (),
        startup_timeout_sec: float = 120.0,
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url)
        self._model_path = model_path
        self._server_bin = server_bin
        self._context_tokens = context_tokens
        self._kv_cache = kv_cache
        self._spec_type = spec_type
        self._mtp_draft = mtp_draft
        self._extra_args = tuple(extra_args)
        self._startup_timeout_sec = startup_timeout_sec
        self._process: subprocess.Popen[str] | None = None
        self._log: list[str] = []
        self._log_queue: Queue[str] = Queue()
        self._started = False
        atexit.register(self.close)

    @classmethod
    @override
    def from_env(cls, *, base_url: str | None = None) -> Self:
        """Build provider from llama.cpp environment variables.

        Args:
          base_url: OpenAI-compatible endpoint override.

        Returns:
          provider: Configured llama.cpp provider.

        """
        return cls._from_values(
            base_url=base_url or os.environ.get("LLAMA_CPP_BASE_URL")
        )

    @classmethod
    @override
    def from_key(cls, api_key: str, *, base_url: str | None = None) -> Self:
        """Build provider from a GGUF path or dummy local API key.

        Args:
          api_key: GGUF path when launching a managed server, otherwise API key.
          base_url: Existing OpenAI-compatible endpoint override.

        Returns:
          provider: Configured llama.cpp provider.

        """
        model_path = api_key if _looks_like_path(api_key) else ""
        key = "local" if model_path else api_key or "local"
        return cls._from_values(api_key=key, model_path=model_path, base_url=base_url)

    @classmethod
    def _from_values(
        cls,
        *,
        api_key: str = "",
        model_path: str = "",
        base_url: str | None = None,
    ) -> Self:
        """Construct provider, defaulting unset values from environment variables."""
        return cls(
            api_key=api_key or os.environ.get(cls.ENV_VAR, "local") or "local",
            base_url=base_url,
            model_path=model_path or os.environ.get("LLAMA_CPP_MODEL", ""),
            server_bin=os.environ.get("LLAMA_CPP_SERVER", ""),
            context_tokens=int(os.environ.get("LLAMA_CPP_CONTEXT", "16384")),
            kv_cache=os.environ.get("LLAMA_CPP_KV", "q4_0"),
            spec_type=os.environ.get("LLAMA_CPP_SPEC_TYPE", ""),
            mtp_draft=int(os.environ.get("LLAMA_CPP_MTP_DRAFT", "3")),
            extra_args=shlex.split(os.environ.get("LLAMA_CPP_EXTRA_ARGS", "")),
        )

    @override
    def model(
        self,
        model_id: str | None = None,
    ) -> OpenAICompatModel:
        """Start llama-server if needed and return the OpenAI-compatible model.

        Args:
          model_id: Model ID exposed to the OpenAI-compatible endpoint.

        Returns:
          model: Chat-completions model backend.

        """
        self._ensure_started()
        return super().model(model_id)

    def close(self) -> None:
        """Terminate the managed llama-server process if it is running."""
        proc = self._process
        self._process = None
        self._started = False
        if proc is not None:
            self.base_url = self.BASE_URL
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)

    def _ensure_started(self) -> None:
        """Launch llama-server if not already running and wait for readiness."""
        if self._started:
            proc = self._process
            if proc is None or proc.poll() is None:
                return
            self.close()
        if self.base_url != self.BASE_URL or os.environ.get("LLAMA_CPP_BASE_URL"):
            self._wait_ready()
            self._started = True
            return
        if not self._model_path:
            raise RuntimeError(
                "LLAMA_CPP_MODEL or --auth /path/to/model.gguf is required."
            )
        port = _free_port()
        self.base_url = f"http://127.0.0.1:{port}/v1"
        argv = self._argv(port)
        self._process = subprocess.Popen(  # noqa: S603 -- argv is fixed plus explicit user-selected llama.cpp flags.
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        _start_log_reader(self._process, self._log_queue)
        try:
            self._wait_ready()
        except Exception:
            self.close()
            raise
        self._started = True

    def _argv(self, port: int) -> list[str]:
        """Build the llama-server command-line argv for the configured model."""
        server = self._server_bin or shutil.which("llama-server") or _docker_server()
        if server is None:
            raise RuntimeError("llama-server not found; set LLAMA_CPP_SERVER.")
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
            argv.extend(
                [
                    "--spec-type",
                    self._spec_type,
                    "--spec-draft-n-max",
                    str(self._mtp_draft),
                ]
            )
        argv.extend(
            [
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
        )
        return argv

    def _wait_ready(self) -> None:
        """Poll the server's /models endpoint until it responds or timeout fires."""
        deadline = time.monotonic() + self._startup_timeout_sec
        while time.monotonic() < deadline:
            self._drain_log()
            proc = self._process
            if proc is not None and proc.poll() is not None:
                self._drain_log()
                raise RuntimeError(_startup_error("llama-server exited", self._log))
            if _http_ok(f"{self.base_url}/models"):
                return
            time.sleep(0.05)
        raise RuntimeError(
            _startup_error("llama-server did not become ready", self._log)
        )

    def _drain_log(self) -> None:
        """Drain queued server log lines into the bounded buffer.

        Keeps the HEAD as well as the tail: llama.cpp names the real
        startup failure in its first lines (a missing model file, an
        impossible GPU layer count), so a tail-only window reported the
        generic noise that followed and dropped the diagnosis.
        """
        while True:
            try:
                line = self._log_queue.get_nowait()
            except Empty:
                return
            self._log.append(line)
            if len(self._log) > _LOG_HEAD_LINES + _LOG_TAIL_LINES:
                del self._log[_LOG_HEAD_LINES:-_LOG_TAIL_LINES]


def _start_log_reader(proc: subprocess.Popen[str], out: Queue[str]) -> None:
    """Spawn a daemon thread that copies the process stdout into ``out``."""
    stream = proc.stdout
    if stream is None:
        return

    def read_lines() -> None:
        """Forward each stripped line from the subprocess stream to ``out``."""
        for line in stream:
            out.put(line.rstrip())

    Thread(target=read_lines, daemon=True).start()


def _looks_like_path(value: str) -> bool:
    """True if ``value`` looks like a filesystem path or a GGUF model file."""
    return value.startswith(("/", "./", "../", "~/")) or value.endswith(".gguf")


def _docker_server() -> str | None:
    """Return Docker Desktop's bundled llama-server path if present."""
    path = Path.home() / ".docker/bin/inference/llama-server"
    return str(path) if path.exists() else None


# Startup-log retention. Head keeps the failure cause (llama.cpp reports
# it first); tail keeps the symptom that stopped the process.
_LOG_HEAD_LINES: Final = 20
_LOG_TAIL_LINES: Final = 40


def _free_port() -> int:
    """Return an OS-assigned free localhost TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _http_ok(url: str) -> bool:
    """True if a GET on ``url`` returns a 2xx response within 200ms."""
    try:
        with urllib.request.urlopen(url, timeout=0.2) as response:  # noqa: S310 -- local/provider-supplied readiness URL only.
            return bool(200 <= response.status < 300)
    except (OSError, urllib.error.URLError):
        return False


def _startup_error(prefix: str, lines: Sequence[str]) -> str:
    """Format a startup-failure message with the captured log appended.

    ``lines`` is already bounded head-and-tail by :meth:`_drain_log`, so
    the whole buffer is emitted: the cause is usually in the head and
    the symptom in the tail, and dropping either loses the diagnosis.
    """
    if not lines:
        return prefix
    return f"{prefix}; recent log:\n" + "\n".join(lines)
