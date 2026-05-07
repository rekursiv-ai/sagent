from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sagent.providers import LlamaCpp, build_provider
from sagent.providers.llamacpp import _startup_error


class _FakePopen:
    def __init__(self, argv: list[str], **kwargs: object) -> None:
        self.argv = argv
        self.kwargs = kwargs
        self.stdout = None
        self.terminated = False
        self.killed = False
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return self.returncode or 0


def test_from_key_treats_gguf_as_model_path() -> None:
    provider = LlamaCpp.from_key("/models/qwen.gguf")

    assert provider.api_key == "local"
    assert provider._model_path == "/models/qwen.gguf"


def test_from_env_reads_managed_server_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLAMA_CPP_MODEL", "/models/qwen.gguf")
    monkeypatch.setenv("LLAMA_CPP_SERVER", "/bin/llama-server")
    monkeypatch.setenv("LLAMA_CPP_CONTEXT", "16384")
    monkeypatch.setenv("LLAMA_CPP_KV", "q4_0")
    monkeypatch.setenv("LLAMA_CPP_SPEC_TYPE", "ngram-cache")
    monkeypatch.setenv("LLAMA_CPP_MTP_DRAFT", "2")
    monkeypatch.setenv("LLAMA_CPP_EXTRA_ARGS", "-fa off -tb 18")

    provider = LlamaCpp.from_env()
    argv = provider._argv(1234)

    assert argv == [
        "/bin/llama-server",
        "-m",
        "/models/qwen.gguf",
        "--host",
        "127.0.0.1",
        "--port",
        "1234",
        "--spec-type",
        "ngram-cache",
        "--spec-draft-n-max",
        "2",
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
        "-fa",
        "off",
        "-tb",
        "18",
    ]


def test_empty_spec_type_disables_speculation_flags() -> None:
    provider = LlamaCpp(
        model_path="/models/tiny.gguf",
        server_bin="/bin/llama-server",
        spec_type="",
    )
    argv = provider._argv(1234)

    assert "--spec-type" not in argv
    assert "--spec-draft-n-max" not in argv


def test_model_reuses_external_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    def http_ok(url: str) -> bool:
        return url == "http://127.0.0.1:9999/v1/models"

    monkeypatch.setattr(
        "sagent.providers.llamacpp._http_ok",
        http_ok,
    )
    provider = LlamaCpp.from_env(base_url="http://127.0.0.1:9999/v1")

    model = provider.model()

    assert model.model_id == "qwen3.6-27b-12gb"
    assert provider._process is None


def test_model_starts_managed_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    popens: list[_FakePopen] = []

    def fake_popen(argv: list[str], **kwargs: Any) -> _FakePopen:
        proc = _FakePopen(argv, **kwargs)
        popens.append(proc)
        return proc

    def which(name: str) -> str | None:
        return "/usr/bin/llama-server" if name == "llama-server" else None

    def free_port() -> int:
        return 54_321

    def http_ok(url: str) -> bool:
        return url == "http://127.0.0.1:54321/v1/models"

    monkeypatch.setattr(
        "sagent.providers.llamacpp.shutil.which",
        which,
    )
    monkeypatch.setattr(
        "sagent.providers.llamacpp._free_port",
        free_port,
    )
    monkeypatch.setattr(
        "sagent.providers.llamacpp.subprocess.Popen",
        fake_popen,
    )
    monkeypatch.setattr(
        "sagent.providers.llamacpp._http_ok",
        http_ok,
    )
    provider = LlamaCpp.from_key("/models/qwen.gguf")

    model = provider.model()

    assert model.model_id == "qwen3.6-27b-12gb"
    assert popens[0].argv[:7] == [
        "/usr/bin/llama-server",
        "-m",
        "/models/qwen.gguf",
        "--host",
        "127.0.0.1",
        "--port",
        "54321",
    ]
    provider.close()
    assert popens[0].terminated


def test_startup_failure_closes_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    popens: list[_FakePopen] = []

    def fake_popen(argv: list[str], **kwargs: Any) -> _FakePopen:
        proc = _FakePopen(argv, **kwargs)
        popens.append(proc)
        return proc

    def http_ok(url: str) -> bool:
        del url
        return False

    monkeypatch.setattr(
        "sagent.providers.llamacpp.subprocess.Popen",
        fake_popen,
    )
    monkeypatch.setattr(
        "sagent.providers.llamacpp._http_ok",
        http_ok,
    )
    provider = LlamaCpp(
        model_path="/models/qwen.gguf",
        server_bin="/bin/llama-server",
        startup_timeout_sec=0.01,
    )

    with pytest.raises(RuntimeError, match="did not become ready"):
        provider.model()

    assert popens[0].terminated
    assert provider._process is None
    assert not provider._started


def test_dead_process_restarts(monkeypatch: pytest.MonkeyPatch) -> None:
    popens: list[_FakePopen] = []

    def fake_popen(argv: list[str], **kwargs: Any) -> _FakePopen:
        proc = _FakePopen(argv, **kwargs)
        popens.append(proc)
        return proc

    def http_ok(url: str) -> bool:
        del url
        return True

    monkeypatch.setattr(
        "sagent.providers.llamacpp.subprocess.Popen",
        fake_popen,
    )
    monkeypatch.setattr(
        "sagent.providers.llamacpp._http_ok",
        http_ok,
    )
    provider = LlamaCpp(
        model_path="/models/qwen.gguf",
        server_bin="/bin/llama-server",
    )

    provider.model()
    popens[0].returncode = -9
    provider.model()

    assert len(popens) == 2
    provider.close()


def test_close_allows_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    popens: list[_FakePopen] = []

    def fake_popen(argv: list[str], **kwargs: Any) -> _FakePopen:
        proc = _FakePopen(argv, **kwargs)
        popens.append(proc)
        return proc

    def http_ok(url: str) -> bool:
        del url
        return True

    monkeypatch.setattr(
        "sagent.providers.llamacpp.subprocess.Popen",
        fake_popen,
    )
    monkeypatch.setattr(
        "sagent.providers.llamacpp._http_ok",
        http_ok,
    )
    provider = LlamaCpp(
        model_path="/models/qwen.gguf",
        server_bin="/bin/llama-server",
    )

    provider.model()
    provider.close()
    provider.model()

    assert len(popens) == 2
    assert popens[0].terminated
    provider.close()


def test_argv_uses_docker_server_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_server(name: str) -> None:
        del name

    monkeypatch.setattr(
        "sagent.providers.llamacpp.shutil.which",
        missing_server,
    )
    monkeypatch.setattr(
        "sagent.providers.llamacpp._docker_server",
        lambda: str(Path.home() / ".docker/bin/inference/llama-server"),
    )
    provider = LlamaCpp(model_path="/models/qwen.gguf")

    assert provider._argv(1234)[0] == str(
        Path.home() / ".docker/bin/inference/llama-server"
    )


def test_missing_model_path_fails_before_launch() -> None:
    provider = LlamaCpp(startup_timeout_sec=0.01)

    with pytest.raises(RuntimeError, match="LLAMA_CPP_MODEL"):
        provider.model()


def test_startup_error_includes_log_tail() -> None:
    message = _startup_error("llama-server exited", ["a", "b"])

    assert message == "llama-server exited; recent log:\na\nb"


def test_build_provider_supports_llamacpp_literal_model_path() -> None:
    provider = build_provider("LlamaCpp", "/models/qwen.gguf")

    assert isinstance(provider, LlamaCpp)
    assert provider._model_path == "/models/qwen.gguf"
