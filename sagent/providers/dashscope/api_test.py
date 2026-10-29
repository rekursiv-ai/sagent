"""Tests for the DashScope request body: Qwen's thinking knobs and token cap."""

from __future__ import annotations

from dataclasses import replace

import pytest

from sagent.model.capability import ModelLimits
from sagent.model.exceptions import AuthError, UnknownModelError
from sagent.model.types import Request
from sagent.providers.catalog import choose
from sagent.providers.dashscope.api import DashScope
from sagent.providers.openai.compat import OpenAIModel
from sagent.session.types import UserMessage


_REQUEST = Request(records=(UserMessage(content="hi"),))


def _body(model_id: str, **choices: object) -> dict[str, object]:
    dashscope = DashScope(api_key="k")
    model = dashscope[model_id]
    for field, value in choices.items():
        model.settings = choose(model.settings, field, value)
    return dashscope.encode(_REQUEST, model=model)


def _knobs(body: dict[str, object]) -> dict[str, object]:
    names = ("enable_thinking", "reasoning_effort", "thinking_budget")
    return {name: body[name] for name in names if name in body}


@pytest.mark.asyncio
async def test_the_provider_is_dashscopes_international_endpoint() -> None:
    dashscope = DashScope()
    assert dashscope.name == "dashscope"
    assert (
        await dashscope.endpoint()
        == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    )
    assert dashscope.key_variable == "DASHSCOPE_API_KEY"
    assert dashscope.reasoning_field == "reasoning_content"
    assert dashscope["default"].capability.model_id == "qwen-plus-3.7"


@pytest.mark.parametrize(
    ("model_id", "cap"),
    [
        ("qwen-plus-3.7", "max_completion_tokens"),
        ("qwen-max-3.8", "max_completion_tokens"),
        # Reasoning rows the API page does not list for the new name.
        ("qwen-27b-3.8", "max_tokens"),
        ("qwen-max-3.6", "max_tokens"),
        ("qwen-coder-480b-a35b-3.0", "max_tokens"),
    ],
)
def test_the_token_cap_takes_the_name_each_row_reads(model_id: str, cap: str) -> None:
    body = _body(model_id)
    other = {"max_tokens", "max_completion_tokens"} - {cap}
    assert (
        body[cap]
        == DashScope(api_key="k")[model_id].capability.limits.max_response_tokens
    )
    assert not other & body.keys()


def test_a_row_that_rejects_the_switch_is_sent_none() -> None:
    assert _knobs(_body("qwen-coder-480b-a35b-3.0")) == {}


def test_no_budget_turns_thinking_off_whatever_the_effort() -> None:
    assert _knobs(_body("qwen-plus-3.7", thinking_budget="none")) == {
        "enable_thinking": False,
    }
    assert _knobs(
        _body("qwen-plus-3.7", thinking_budget="none", thinking_effort="high"),
    ) == {"enable_thinking": False}


def test_a_level_row_takes_its_effort_as_named() -> None:
    # OpenAI's spelling would fold ``xhigh`` into ``high``.
    body = _body("qwen-max-3.8", thinking_budget="auto", thinking_effort="xhigh")
    assert _knobs(body) == {"enable_thinking": True, "reasoning_effort": "xhigh"}
    # Unasked, the row starts where Qwen 3.8 does when sent no effort.
    assert _knobs(_body("qwen-max-3.8", thinking_budget="auto")) == {
        "enable_thinking": True,
        "reasoning_effort": "xhigh",
    }


@pytest.mark.parametrize(
    ("effort", "budget"),
    [
        ("min", 1024),
        ("low", 4096),
        ("medium", 8192),
        ("high", 16_384),
        ("xhigh", 20_480),
        ("max", 24_576),
    ],
)
def test_a_fixed_budget_caps_the_reasoning_by_effort(effort: str, budget: int) -> None:
    body = _body("qwen-plus-3.7", thinking_budget="fixed", thinking_effort=effort)
    assert _knobs(body) == {"enable_thinking": True, "thinking_budget": budget}


def test_an_unfixed_budget_thinks_at_the_models_own_depth() -> None:
    body = _body("qwen-plus-3.7", thinking_budget="auto", thinking_effort="high")
    assert _knobs(body) == {"enable_thinking": True}
    stopped = _body("qwen-plus-3.7", thinking_budget="fixed", thinking_effort="none")
    assert _knobs(stopped) == {"enable_thinking": False}
    level = _body("qwen-max-3.8", thinking_budget="auto", thinking_effort="none")
    assert _knobs(level) == {"enable_thinking": False}
    # A row that always thinks still names the switch it cannot turn off.
    assert _knobs(_body("qwen-235b-a22b-thinking-3.0")) == {"enable_thinking": True}


def test_every_row_named_for_the_new_cap_is_in_the_catalog() -> None:
    assert DashScope.completion_cap <= set(DashScope())


# Model Studio's deep-thinking page (read 2026-10-27): Qwen 3.5 and later think
# unless told not to; qwen-plus and qwen-turbo do not. Qwen3-32B's model card:
# its thinking is on by default.
@pytest.mark.parametrize(
    ("model_id", "thinks"),
    [
        ("qwen-max-3.8", True),
        ("qwen-flash-3.8", True),
        ("qwen-27b-3.8", True),
        ("qwen-2400b-a95b-3.8", True),
        ("qwen-max-3.7", True),
        ("qwen-plus-3.7", True),
        ("qwen-flash-3.7", True),
        ("qwen-max-3.6", True),
        ("qwen-plus-3.6", True),
        ("qwen-flash-3.6", True),
        ("qwen-32b-3.0", True),
        ("qwen-235b-a22b-thinking-3.0", True),
        ("qwen-235b-a22b-instruct-3.0", False),
        ("qwen-plus-3.0", False),
        ("qwen-turbo-3.0", False),
    ],
)
def test_a_model_starts_thinking_as_model_studio_has_it(
    model_id: str,
    thinks: bool,
) -> None:
    assert _body(model_id).get("enable_thinking", False) is thinks


def test_an_unknown_model_names_the_known_ones() -> None:
    with pytest.raises(UnknownModelError, match=r"qwen-plus-3\.7"):
        DashScope()["qwen-mega-9.9"]


@pytest.mark.asyncio
async def test_the_key_is_read_from_dashscope_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-ds")
    assert await DashScope().credentials() == {"authorization": "Bearer sk-ds"}
    monkeypatch.delenv("DASHSCOPE_API_KEY")
    with pytest.raises(AuthError, match="DASHSCOPE_API_KEY"):
        await DashScope().credentials()


def test_a_model_of_unknown_reply_length_is_sent_no_cap() -> None:
    dashscope = DashScope(api_key="k")
    capability = replace(
        dashscope["qwen-plus-3.7"].capability,
        limits=ModelLimits(),
    )
    model = OpenAIModel(
        provider=dashscope,
        capability=capability,
        settings=capability.default,
    )
    body = dashscope.encode(_REQUEST, model=model)
    assert not {"max_tokens", "max_completion_tokens"} & body.keys()


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
