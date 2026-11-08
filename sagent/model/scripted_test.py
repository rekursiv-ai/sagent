"""Tests for scripted replies, streams, and provider resource ownership."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from sagent.model import scripted
from sagent.model.capability import (
    ModelCapability,
    ModelLimits,
    ModelSettings,
)
from sagent.model.cost import TokenCost, TokenCount
from sagent.model.exceptions import UnknownModelError
from sagent.model.scripted import ScriptedProvider, call, reply
from sagent.model.types import Request, Response, TextChunk
from sagent.providers.resources import SharedResources
from sagent.providers.table import ModelTable
from sagent.session.types import (
    AssistantMessage,
    Attachment,
    UserMessage,
)


def test_a_reply_may_be_computed_from_the_request() -> None:
    seen: list[Request] = []

    def answer(request: Request) -> Response:
        seen.append(request)
        said = [r.content for r in request.records if isinstance(r, UserMessage)]
        return reply(f"you said {said[-1]}")

    provider = ScriptedProvider(replies=[answer])
    request = Request(records=(UserMessage(content="hi"),))
    assert provider.next_reply(request) == reply("you said hi")
    assert seen == provider.requests == [request]
    assert seen[0] is request


def test_reply_stops_for_tools_only_when_it_calls_them() -> None:
    assert reply("x").stop_reason == "end_turn"
    assert reply(calls=[call("Read")]).stop_reason == "tool_use"
    assert reply("x", stop_reason="max_tokens").stop_reason == "max_tokens"


def test_call_names_its_id_after_the_tool() -> None:
    made = call("Read", file_path="a")
    assert made.name == "Read"
    assert made.call_id == "call_Read"
    assert dict(made.arguments) == {"file_path": "a"}
    assert call("Read", "id1").call_id == "id1"


def test_a_provider_is_named_for_its_section_unless_told() -> None:
    assert ScriptedProvider.section == "scripted"
    assert ScriptedProvider().name == "scripted"
    assert ScriptedProvider(name="other").name == "other"


def test_a_provider_has_no_rate_limits_and_one_model() -> None:
    provider = ScriptedProvider()
    assert provider.usage == ()
    assert list(provider) == ["scripted"]
    assert list(ScriptedProvider(capability=ModelCapability(model_id="m"))) == ["m"]


def test_every_model_shares_the_provider_and_starts_from_the_default() -> None:
    capability = ModelCapability(
        model_id="m",
        thinking_effort=frozenset({"low", "high"}),
        default=ModelSettings(thinking_effort="low"),
    )
    provider = ScriptedProvider(capability=capability)
    model = provider["anything"]
    assert model.provider is provider
    assert model.capability is capability
    assert model.settings == capability.default


def test_a_model_takes_only_settings_its_capability_offers() -> None:
    capability = ModelCapability(
        model_id="m",
        thinking_effort=frozenset({"low", "high"}),
        default=ModelSettings(thinking_effort="low"),
    )
    model = ScriptedProvider(capability=capability)["m"]
    high = replace(capability.default, thinking_effort="high")

    model.settings = high

    assert model.settings == high
    with pytest.raises(ValueError, match="thinking_effort"):
        model.settings = replace(capability.default, thinking_effort="max")
    assert model.settings == high


def test_a_provider_given_models_serves_them_as_a_vendor_table_does() -> None:
    wide = ModelCapability(
        model_id="wide-1.0",
        limits=ModelLimits(max_request_tokens=1_000_000),
    )
    provider = ScriptedProvider(models=ModelTable(rows=(wide,)))
    model = provider["wide+200k"]
    assert model.provider is provider
    assert (model.capability.model_id, model.settings.context_size.tokens) == (
        "wide-1.0",
        200_000,
    )
    assert list(provider) == ["wide-1.0"]
    with pytest.raises(UnknownModelError):
        provider["scripted"]


def test_the_default_capability_is_a_small_window() -> None:
    capability = ScriptedProvider().capability
    limits = capability.limits
    assert (limits.max_request_tokens, limits.max_response_tokens) == (200_000, 8_000)
    assert capability.prices.window == 200_000
    assert capability.default.context_size == capability.prices["fullcontext"]
    assert capability.default.context_size.tokens == 200_000


@pytest.mark.asyncio
async def test_a_provider_closes_only_the_resources_it_made() -> None:
    close_owned, close_shared = AsyncMock(), AsyncMock()
    owned_resources = Mock(spec=SharedResources, aclose=close_owned)
    shared = Mock(spec=SharedResources, aclose=close_shared)
    with patch.object(
        scripted,
        "SharedResources",
        return_value=owned_resources,
    ) as make:
        owned = ScriptedProvider()
        assert owned.resources is owned_resources
        await owned.aclose()
        close_owned.assert_awaited_once_with()
        borrowed = ScriptedProvider(resources=shared)
        assert borrowed.resources is shared
        await borrowed.aclose()
        close_shared.assert_not_awaited()
    make.assert_called_once_with()


@pytest.mark.asyncio
async def test_count_scales_the_characters_of_system_and_records() -> None:
    model = ScriptedProvider(capability=ModelCapability(approx_chars_per_token=2.0))[
        "m"
    ]
    record = UserMessage(content="hi")
    request = Request(system=("abcd", "ef"), records=(record,))
    assert await model.count(request) == int((6 + len(repr(record))) / 2.0)
    assert await model.count(Request()) == 0


@pytest.mark.asyncio
async def test_stream_yields_the_text_then_the_reply() -> None:
    turn = reply("hello", calls=[call("Read")])
    empty = reply()
    provider = ScriptedProvider(replies=[turn, empty])
    first = Request(system=("first",))
    second = Request(system=("second",))
    events = [event async for event in provider["m"].stream(first)]
    assert events == [TextChunk(text="hello"), turn]
    silent = [event async for event in provider["m"].stream(second)]
    assert silent == [empty]
    assert provider.requests == [first, second]
    assert provider.requests[0] is first
    assert provider.requests[1] is second


def test_the_script_is_taken_in_order_and_every_request_kept() -> None:
    first, second = reply("one"), reply("two")
    provider = ScriptedProvider(replies=[first, second])
    a, b, c = Request(system=("a",)), Request(system=("b",)), Request(system=("c",))
    assert provider.next_reply(a) is first
    assert provider.next_reply(b) is second
    with pytest.raises(RuntimeError, match=r"^The script ran out at request 3\.$"):
        provider.next_reply(c)
    assert provider.requests == [a, b, c]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [ValueError("boom"), KeyboardInterrupt()])
async def test_a_scripted_exception_is_raised_after_its_request_is_kept(
    error: BaseException,
) -> None:
    provider = ScriptedProvider(replies=[error])
    request = Request(records=(UserMessage(content="hi"),))
    with pytest.raises(type(error)) as raised:
        _ = [event async for event in provider["scripted"].stream(request)]
    assert raised.value is error
    assert provider.requests == [request]
    assert provider.requests[0] is request


def test_reply_carries_its_text_calls_tokens_and_spend() -> None:
    read = call("Read", path="a")
    turn = reply(
        "said",
        calls=[read],
        tokens=TokenCount(request=3),
        spend=TokenCost(request=0.5),
    )
    assert turn.records == (AssistantMessage(content="said"), read)
    assert (turn.tokens, turn.spend) == (TokenCount(request=3), TokenCost(request=0.5))
    bare = reply()
    assert bare.records == ()
    assert (bare.tokens, bare.spend) == (TokenCount(), TokenCost())


@pytest.mark.asyncio
async def test_stream_joins_every_message_of_the_reply() -> None:
    turn = Response(
        records=(
            AssistantMessage(content="one "),
            AssistantMessage(),
            AssistantMessage(content=""),
            AssistantMessage(content="two"),
        ),
        tokens=TokenCount(),
        spend=TokenCost(),
        stop_reason="end_turn",
    )
    provider = ScriptedProvider(replies=[turn])
    events = [event async for event in provider["m"].stream(Request())]
    assert events == [TextChunk(text="one "), TextChunk(text="two", record=3), turn]


def test_an_attachment_costs_its_bytes_scaled_as_characters_are() -> None:
    model = ScriptedProvider()["scripted"]
    attached = Attachment(mime_descriptor="image/png", data=b"x" * 40)
    assert model.attachment_tokens(attached) == int(
        40 / model.capability.approx_chars_per_token,
    )


def test_settings_accept_offered_choices_and_preserve_them_on_refusal() -> None:
    model = ScriptedProvider()["scripted"]
    chosen = replace(model.settings, service_tier="default")
    model.settings = chosen
    assert model.settings is chosen
    with pytest.raises(ValueError, match="thinking_effort='high' is not offered"):
        model.settings = replace(chosen, thinking_effort="high")
    assert model.settings is chosen


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
