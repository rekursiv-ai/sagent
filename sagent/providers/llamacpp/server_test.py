"""Tests for ``providers.llamacpp``: managed llama-server provider plumbing."""

from __future__ import annotations

from pathlib import Path

import shutil

import pytest

from sagent.providers.llamacpp import server as llamacpp_mod
from sagent.providers.llamacpp.server import (
    LlamaCpp,
    _free_port,
    _looks_like_path,
    _startup_error,
)


@pytest.mark.parametrize(
    "value",
    [
        "/abs/path/model.gguf",
        "./relative/model.gguf",
        "../parent/model.gguf",
        "~/home/model.gguf",
        "anything.gguf",
    ],
)
def test_looks_like_path_positive(value: str) -> None:
    assert _looks_like_path(value) is True


@pytest.mark.parametrize("value", ["sk-key", "no-auth", "local"])
def test_looks_like_path_negative(value: str) -> None:
    assert _looks_like_path(value) is False


def test_startup_error_no_log() -> None:
    assert _startup_error("server died", []) == "server died"


def test_startup_error_with_tail() -> None:
    log = [f"line {i}" for i in range(30)]
    out = _startup_error("server died", log)
    assert out.startswith("server died; recent log:")
    # Tail is the last 20 lines.
    assert "line 29" in out
    assert "line 9" not in out


def test_free_port_returns_int_within_range() -> None:
    p = _free_port()
    assert isinstance(p, int)
    assert 1 <= p <= 65_535


def test_llamacpp_from_key_with_path_treated_as_model_path() -> None:
    p = LlamaCpp.from_key("/opt/models/model.gguf")
    # ``api_key`` falls back to "local"; model_path stored separately.
    assert p.api_key == "local"


def test_llamacpp_from_key_non_path_stays_as_api_key() -> None:
    p = LlamaCpp.from_key("api-key-string")
    # Non-path treats the string as a literal api key.
    assert p.api_key == "api-key-string"


def test_llamacpp_from_key_empty_falls_back_to_local() -> None:
    p = LlamaCpp.from_key("")
    assert p.api_key == "local"


def test_llamacpp_from_env_uses_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLAMA_CPP_API_KEY", "env-key")
    monkeypatch.setenv("LLAMA_CPP_BASE_URL", "http://stub.test/v1")
    monkeypatch.delenv("LLAMA_CPP_MODEL", raising=False)
    p = LlamaCpp.from_env()
    assert p.api_key == "env-key"


def test_llamacpp_known_models_include_local() -> None:
    assert "local" in LlamaCpp.CAPABILITIES
    assert "qwen3.6-27b-12gb" in LlamaCpp.CAPABILITIES


def test_llamacpp_close_idempotent() -> None:
    p = LlamaCpp.from_key("local")
    p.close()
    p.close()  # Second call must not raise.


def test_llamacpp_model_without_model_path_or_base_url_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLAMA_CPP_BASE_URL", raising=False)
    p = LlamaCpp.from_key("not-a-path")
    with pytest.raises(RuntimeError, match="LLAMA_CPP_MODEL"):
        _ = p.model()


def test_llamacpp_argv_missing_server_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """No ``llama-server`` on PATH and no env override → RuntimeError."""
    fake_model = tmp_path / "missing.gguf"
    p = LlamaCpp.from_key(str(fake_model))

    def _which(name: str) -> str | None:
        del name
        return None

    def _no_docker() -> str | None:
        return None

    monkeypatch.setattr(shutil, "which", _which)
    monkeypatch.setattr(llamacpp_mod, "_docker_server", _no_docker)
    with pytest.raises(RuntimeError, match="llama-server not found"):
        _ = p._argv(8080)


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
