"""Live GPT-5.6 integration tests for ``OpenAISubscription``.

These calls exercise the ChatGPT subscription Responses backend, which mocked
unit tests cannot prove accepts a model or effort value. They are deselected by
default; run explicitly with::

    uv --quiet run --frozen pytest -m integration \
      sagent/providers/openai_sub_integration_test.py

Requires OAuth credentials from ``sagent --provider OpenAISubscription login``.
An API-key-mode ``~/.codex/auth.json`` is intentionally not sufficient.
"""

from __future__ import annotations

import pytest

from sagent.providers.openai.sub import OpenAISubscription
from sagent.types.model import ModelRequest
from sagent.types.runtime import UserMessage


pytestmark = pytest.mark.cli_codex


def _subscription_credentials_available() -> bool:
    try:
        OpenAISubscription.load()
    except (FileNotFoundError, ValueError):
        return False
    return True


_requires_subscription = pytest.mark.skipif(
    not _subscription_credentials_available(),
    reason="requires OpenAI ChatGPT subscription OAuth credentials",
)


@_requires_subscription
@pytest.mark.parametrize(
    ("model_id", "effort"),
    [
        ("gpt-5.6-sol", "low"),
        ("gpt-5.6-terra", "low"),
        ("gpt-5.6-luna", "low"),
        ("gpt-5.6-sol", "max"),
    ],
)
@pytest.mark.asyncio
async def test_gpt_56_subscription_turn(model_id: str, effort: str) -> None:
    provider = OpenAISubscription.from_credentials()
    model = provider.model(model_id)
    try:
        response = await model.buffer(
            ModelRequest(
                messages=[UserMessage(text="Reply with exactly: OK")],
                effort=effort,
            )
        )
    finally:
        await model.close()
    assert response.message.text.strip() == "OK"
