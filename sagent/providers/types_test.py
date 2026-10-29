"""Tests that each provider, model, and shared resource meets its protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING

import io

from PIL import Image

import pytest

from sagent.model.scripted import ScriptedModel, ScriptedProvider
from sagent.providers.anthropic import wire as anthropic_wire
from sagent.providers.anthropic.api import Anthropic, AnthropicModel
from sagent.providers.anthropic.cli import AnthropicCLI, AnthropicCLIModel
from sagent.providers.catalog import (
    ModelCatalog as Catalog,
    listings,
)
from sagent.providers.dashscope.api import DashScope
from sagent.providers.google import wire as google_wire
from sagent.providers.google.api import Google, GoogleModel
from sagent.providers.google.cli import GoogleCLI, GoogleCLIModel
from sagent.providers.llamacpp.server import LlamaCpp
from sagent.providers.minimax.api import MiniMax
from sagent.providers.moonshot.api import Moonshot
from sagent.providers.openai import token_count
from sagent.providers.openai.api import OpenAI
from sagent.providers.openai.compat import OpenAIModel
from sagent.providers.openai.sub import OpenAISub
from sagent.providers.resources import SharedResources, SubprocessPool
from sagent.providers.selfhosted.server import SelfHosted, SelfHostedModel
from sagent.providers.tool_server import McpMount, McpToolServer
from sagent.providers.types import (
    Login,
    Model,
    ModelCatalog,
    ProcessPool,
    Provider,
    ProviderSharedResources,
    ToolMount,
    ToolServer,
    UsageWindow,
)
from sagent.session.types import Attachment


if TYPE_CHECKING:
    from sagent.providers.listing import Listing


def _members(protocol: type) -> set[str]:
    """Return every public name ``protocol`` declares, and ``__getitem__``/``__iter__``."""
    dunders = {"__getitem__", "__iter__"}
    return {
        name
        for name in (*vars(protocol), *getattr(protocol, "__annotations__", {}))
        if not name.startswith("_") or name in dunders
    }


@pytest.mark.asyncio
async def test_every_catalog_provider_is_a_provider_and_serves_models() -> None:
    # What each provider's models are; one kept out of the export tests its own.
    expected: dict[str, type] = {
        Anthropic.section: AnthropicModel,
        AnthropicCLI.section: AnthropicCLIModel,
        OpenAI.section: OpenAIModel,
        OpenAISub.section: OpenAIModel,
        Google.section: GoogleModel,
        GoogleCLI.section: GoogleCLIModel,
        DashScope.section: OpenAIModel,
        MiniMax.section: OpenAIModel,
        Moonshot.section: OpenAIModel,
        LlamaCpp.section: OpenAIModel,
        SelfHosted.section: SelfHostedModel,
    }
    async with SharedResources() as shared:
        offered = listings()
        providers: list[Provider] = [
            listing.factory(name="", resources=shared) for listing in offered
        ]
        models: list[Model] = [p["default"] for p in providers]
        served = {
            listing.section: type(model)
            for listing, model in zip(offered, models, strict=True)
            if listing.section in expected
        }

        assert served == expected
        assert all(m.provider is p for m, p in zip(models, providers, strict=True))
        assert all(p.resources is shared for p in providers)
        members = _members(Provider) | _members(Model)
        assert {"name", "usage", "provider", "count", "stream"} <= members
        for provider, model in zip(providers, models, strict=True):
            assert _members(Provider) <= set(dir(provider)), type(provider)
            assert _members(Model) <= set(dir(model)), type(model)


def _png(width: int, height: int) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (width, height), "red").save(out, format="PNG")
    return out.getvalue()


@pytest.mark.asyncio
async def test_each_model_bills_an_attachment_by_its_vendors_formula() -> None:
    picture = Attachment(mime_descriptor="image/png", data=_png(1600, 800))
    async with SharedResources() as shared:
        for factory, bill in (
            (Anthropic, anthropic_wire.attachment_tokens),
            (AnthropicCLI, anthropic_wire.attachment_tokens),
            (Google, google_wire.attachment_tokens),
        ):
            model = factory(name="", resources=shared)["default"]
            assert model.attachment_tokens(picture) == bill(
                picture,
                limits=model.capability.limits,
            ), factory
        for factory in (OpenAI, DashScope, LlamaCpp):
            model = factory(name="", resources=shared)["default"]
            assert model.attachment_tokens(picture) == token_count.attachment_tokens(
                picture,
                capability=model.capability,
            ), factory
        # The local model reads text alone; an attachment never reaches it.
        local = SelfHosted(name="", resources=shared)["default"]
        assert local.attachment_tokens(picture) == 0
        # `agy` sends text alone: a line names the attachment in its place.
        agy = GoogleCLI(name="", resources=shared)["default"]
        image = google_wire.attachment_tokens(picture, limits=agy.capability.limits)
        assert 0 < agy.attachment_tokens(picture) < image


def test_a_scripted_provider_is_a_provider_too() -> None:
    provider: Provider = ScriptedProvider()
    model: Model = provider["any"]

    assert isinstance(model, ScriptedModel)
    assert list(provider) == ["scripted"]
    assert provider.usage == ()


@pytest.mark.parametrize(
    "listing",
    listings(),
    ids=[listing.section for listing in listings()],
)
def test_a_login_is_any_provider_that_signs_in(listing: Listing) -> None:
    built = listing.factory(name="", resources=SharedResources())
    assert isinstance(built, Login) is (listing.default_login is not None)


@pytest.mark.asyncio
async def test_the_shared_resources_meet_their_protocols() -> None:
    async with Catalog() as catalog:
        shared: ProviderSharedResources = catalog
        typed: ModelCatalog = catalog
        pool: ProcessPool = shared.process_pool("claude")
        server: ToolServer = shared.tool_server
        mount: ToolMount = McpMount("http://127.0.0.1:1/token")

        assert typed["anthropic"] is catalog["anthropic"]
        assert isinstance(pool, SubprocessPool)
        assert isinstance(server, McpToolServer)
        assert mount.url == "http://127.0.0.1:1/token"
        for protocol, value in (
            (ProviderSharedResources, shared),
            (ModelCatalog, typed),
            (ProcessPool, pool),
            (ToolServer, server),
            (ToolMount, mount),
        ):
            assert _members(protocol) <= set(dir(value)), protocol


def test_a_usage_window_says_nothing_it_was_not_told() -> None:
    window = UsageWindow(label="5h")

    assert (window.utilization, window.resets_at, window.blocked) == (
        None,
        None,
        False,
    )


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
