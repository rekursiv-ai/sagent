from __future__ import annotations

import pytest

from sagent.providers import build_provider, infer_provider
from sagent.providers.anthropic import Anthropic


@pytest.mark.parametrize(
    ("model_id", "current", "expected"),
    [
        # Cross-family: API-key context
        ("claude-opus-4-6", "OpenAI", ("Anthropic", "env")),
        ("gpt-4o", "Anthropic", ("OpenAI", "env")),
        ("gemini-2.5-pro", "Anthropic", ("Google", "env")),
        # Same family → None
        ("claude-sonnet-4-6", "Anthropic", None),
        ("gpt-5.4", "OpenAI", None),
        # Unknown prefix → None
        ("llama-3", "Anthropic", None),
        # Other providers
        ("kimi-k2", "Anthropic", ("Moonshot", "env")),
        ("qwen3-235b", "Anthropic", ("DashScope", "env")),
    ],
)
def test_infer_provider(
    model_id: str,
    current: str,
    expected: tuple[str, str] | None,
) -> None:
    assert infer_provider(model_id, current) == expected


def test_build_provider_literal_key() -> None:
    prov = build_provider("Anthropic", "sk-ant-test-literal")
    assert isinstance(prov, Anthropic)
    assert prov._api_key == "sk-ant-test-literal"
