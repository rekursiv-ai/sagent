"""Tests for the catalog: providers by name, built once, closed with it."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, override

import re

import pytest

from sagent.model.capability import ModelCapability, ModelLimits, ModelSettings
from sagent.model.cost import PriceCatalog
from sagent.model.exceptions import UnknownModelError, UnknownProviderError
from sagent.model.scripted import ScriptedModel, ScriptedProvider
from sagent.providers.anthropic.api import Anthropic
from sagent.providers.anthropic.cli import AnthropicCLI
from sagent.providers.anthropic.login import ClaudeLogin
from sagent.providers.catalog import (
    CachedModel,
    ModelCatalog,
    ModelMemory,
    account_credentials,
    accounts,
    choose,
    default_provider,
    listings,
    offered,
    pick,
    recall,
    successor,
)
from sagent.providers.selfhosted import server
from sagent.providers.table import ModelTable, ModelTags, chosen, spelled
from sagent.types import AsyncResource


if TYPE_CHECKING:
    from collections.abc import Iterator

    import httpx2

    from sagent.providers.types import Model, ProviderSharedResources, UsageWindow


def test_every_registered_provider_is_built_under_its_name() -> None:
    catalog = ModelCatalog()
    for name in catalog.factories:
        provider = catalog[name]
        assert provider.name == name
        assert provider.resources is catalog


@pytest.fixture
def rows(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, ModelCapability]]:
    """Return every catalog row, as ``provider/model``, with its capability.

    SelfHosted reads its window from the model's config on the Hub, which took
    2.5s; the window it states is not what these tests check.
    """

    def stated(path: str, *, trust_remote_code: bool) -> int:
        del path, trust_remote_code
        return 262_144

    monkeypatch.setattr(server, "_window", stated)
    catalog = ModelCatalog()
    return [
        (f"{name}/{model}", provider[model].capability)
        for name in catalog.factories
        for provider in (catalog[name],)
        for model in provider
    ]


def test_every_row_offers_the_effort_it_starts_at(
    rows: list[tuple[str, ModelCapability]],
) -> None:
    """A row's default is one it offers, so nothing falls to a neighbor unstated."""
    unoffered: dict[str, str] = {}
    for row, capability in rows:
        try:
            capability.check(capability.default)
        except ValueError as e:
            unoffered[row] = str(e)
    assert unoffered == {}


def test_every_row_starts_at_the_size_its_prices_name(
    rows: list[tuple[str, ModelCapability]],
) -> None:
    """The default's size is resolved once, from the row's own prices."""
    unresolved = {
        row: default.context_size
        for row, capability in rows
        if (default := capability.default).context_size
        != capability.prices[default.service_tier, default.context_size.context_name]
    }
    assert unresolved == {}


def test_every_row_says_what_a_reply_past_its_window_does(
    rows: list[tuple[str, ModelCapability]],
) -> None:
    """Truncated, or the request rejected: unknown caps every reply needlessly."""
    unknown = sorted(
        row
        for row, capability in rows
        if capability.limits.truncates_at_context is None
    )
    assert unknown == []


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("anthropic", Anthropic),
        ("anthropic-cli", AnthropicCLI),
    ],
)
def test_a_section_name_builds_its_provider_class(name: str, kind: type) -> None:
    assert isinstance(ModelCatalog()[name], kind)


def test_the_default_provider_is_the_one_preferred_else_the_first() -> None:
    offered = listings()
    preferred = [listing.section for listing in offered if listing.preferred]
    assert len(preferred) <= 1
    assert (preferred or [offered[0].section])[0] == default_provider()
    assert default_provider() in ModelCatalog().factories


_ACCOUNT_FILES = (
    ("anthropic-cli", "<claude>/.credentials-work.json"),
    ("openai-sub", "~/.codex/auth-work.json"),
)


def _file(path: str) -> Path:
    """Return a ``_ACCOUNT_FILES`` path, ``<claude>`` where Claude's logins live."""
    claude = str(ClaudeLogin.default_path().parent)
    return Path(path.replace("<claude>", claude)).expanduser()


@pytest.mark.parametrize(("name", "path"), _ACCOUNT_FILES)
def test_an_account_names_its_own_login_file(
    name: str,
    path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CODEX_HOME", raising=False)
    assert account_credentials(name, "work") == _file(path)


def test_a_codex_account_lives_where_codex_keeps_its_login(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    assert account_credentials("openai-sub", "work") == tmp_path / "auth-work.json"


@pytest.mark.parametrize(("name", "path"), _ACCOUNT_FILES)
def test_a_named_account_signs_every_login_provider_in_as_it(
    name: str,
    path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CODEX_HOME", raising=False)
    factory = ModelCatalog(account="work").factories[name]

    assert isinstance(factory, partial)
    assert factory.keywords == {"credentials": _file(path)}


@pytest.mark.parametrize("account", [None, "default", ""])
def test_the_default_account_keeps_each_providers_own_login(
    account: str | None,
) -> None:
    factories = ModelCatalog(account=account).factories
    assert not any(isinstance(factory, partial) for factory in factories.values())


@pytest.mark.parametrize("account", ["../x", "a/b", "-x", "a b", "work.x", "a..b"])
def test_an_account_that_is_no_name_is_refused(account: str) -> None:
    with pytest.raises(ValueError, match="is not a name"):
        ModelCatalog(account=account)


@pytest.mark.parametrize("account", ["Work", "w2", "a_b-C"])
def test_an_account_of_letters_digits_and_separators_is_a_name(account: str) -> None:
    assert ModelCatalog(account=account).account == account


def test_using_an_account_signs_in_as_it_from_then_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CODEX_HOME", raising=False)
    catalog = ModelCatalog()
    catalog.allow(["anthropic-cli", "openai"])
    before = catalog["anthropic-cli"]

    catalog.use_account("work")

    assert catalog.account == "work"
    assert list(catalog.factories) == ["anthropic-cli", "openai"]
    factory = catalog.factories["anthropic-cli"]
    assert isinstance(factory, partial)
    assert factory.keywords == {
        "credentials": ClaudeLogin.default_path().with_name(".credentials-work.json"),
    }
    assert catalog["anthropic-cli"] is not before


@pytest.mark.parametrize("account", [None, "default", ""])
def test_using_the_default_account_again_keeps_each_providers_own_login(
    account: str | None,
) -> None:
    catalog = ModelCatalog(account="work")
    catalog.use_account(account)
    assert catalog.account is None
    assert not isinstance(catalog.factories["anthropic-cli"], partial)


@pytest.mark.asyncio
async def test_a_provider_left_by_an_account_switch_closes_with_the_catalog() -> None:
    built: list[_Recorded] = []

    def per_account(account: str | None) -> dict[str, _Factory]:
        del account
        return {"a": _Factory(built=built)}

    catalog = ModelCatalog(factories=per_account)
    catalog["a"]
    catalog.use_account("work")
    catalog["a"]
    assert [provider.closes for provider in built] == [0, 0]

    await catalog.aclose()

    assert [provider.closes for provider in built] == [1, 1]


def test_switching_back_to_an_account_finds_its_providers_as_left() -> None:
    built: list[_Recorded] = []

    def per_account(account: str | None) -> dict[str, _Factory]:
        del account
        return {"a": _Factory(built=built)}

    catalog = ModelCatalog(factories=per_account)
    first = catalog["a"]
    catalog.use_account(None)
    assert catalog["a"] is first, "the account in use is no switch"
    catalog.use_account("work")
    work = catalog["a"]
    catalog.use_account("default")
    assert catalog["a"] is first
    catalog.use_account("work")
    assert catalog["a"] is work
    assert built == [first, work]


@pytest.mark.asyncio
async def test_a_provider_signed_in_as_another_account_leaves_the_catalog_s_own() -> (
    None
):
    built: list[_Recorded] = []
    asked: list[str | None] = []

    def per_account(account: str | None) -> dict[str, _Factory]:
        asked.append(account)
        return {"a": _Factory(built=built)}

    catalog = ModelCatalog(factories=per_account)
    mine = catalog["a"]
    work = catalog.signed_in("a", account="work")

    assert catalog.account is None
    assert catalog["a"] is mine
    assert work is not mine
    assert catalog.signed_in("a", account="work") is work
    assert catalog.signed_in("a", account="default") is mine
    assert asked == [None, "work"]
    await catalog.aclose()
    assert [provider.closes for provider in built] == [1, 1]


def test_a_catalog_of_fixed_factories_has_no_accounts() -> None:
    catalog = ModelCatalog(factories={"work": Anthropic})
    with pytest.raises(ValueError, match="builds its providers for no account"):
        catalog.use_account("work")
    with pytest.raises(ValueError, match="builds its providers for no account"):
        ModelCatalog(factories={"work": Anthropic}, account="work")
    assert (
        ModelCatalog(factories={"work": Anthropic}, account="default").account is None
    )


def test_an_api_provider_keeps_no_named_accounts() -> None:
    with pytest.raises(ValueError, match="has no named accounts"):
        account_credentials("anthropic", "work")
    assert accounts("anthropic") == ()


def test_the_accounts_are_the_login_files_beside_the_default_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    assert accounts("openai-sub") == ()
    for name in ("auth.json", "auth-work.json", "auth-home.json", "auth-.json"):
        (tmp_path / name).write_text("{}")
    (tmp_path / "auth-x.txt").write_text("{}")
    assert accounts("openai-sub") == ("home", "work")


def test_a_provider_is_built_once_on_first_lookup() -> None:
    built: list[_Recorded] = []
    catalog = ModelCatalog(factories={"work": _Factory(built=built)})
    assert built == []  # Building waits for the first lookup.
    first = catalog["work"]
    assert catalog["work"] is first
    assert built == [first]


def test_given_factories_replace_the_registry() -> None:
    catalog = ModelCatalog(factories={"work": Anthropic})
    work = catalog["work"]
    assert isinstance(work, Anthropic)
    assert work.name == "work"
    assert work.resources is catalog
    with pytest.raises(
        UnknownProviderError,
        match=r"^unknown provider 'anthropic'; known: work$",
    ):
        catalog["anthropic"]


def test_an_unknown_provider_names_the_known_ones() -> None:
    with pytest.raises(
        UnknownProviderError,
        match=r"^unknown provider 'nope'; known: anthropic, ",
    ):
        ModelCatalog()["nope"]


@pytest.mark.parametrize(
    ("spelling", "name"),
    [
        ("openai-sub", "openai-sub"),
        ("OpenAISub", "openai-sub"),
        ("openai_sub", "openai-sub"),
        ("OPENAI SUB", "openai-sub"),
        ("OpenAISubscription", "openai-sub"),
        ("oaisub", "openai-sub"),
        ("osub", "openai-sub"),
        ("antcli", "anthropic-cli"),
        ("OpenAI", "openai"),
        ("LlamaCpp", "llamacpp"),
        ("ds", "dashscope"),
    ],
)
def test_a_provider_answers_to_every_spelling_of_its_name(
    spelling: str,
    name: str,
) -> None:
    catalog = ModelCatalog()
    assert catalog.name(spelling) == name
    assert catalog[spelling] is catalog[name]


_VENDORS = {
    "anthropic": ("a", "ant", "anth", "anthropic"),
    "openai": ("o", "oai", "openai"),
    "google": ("g", "goo", "google"),
}


def _kinds(vendor: str) -> list[str]:
    """Return how each of ``vendor``'s providers is reached: ``api``, ``cli``, ..."""
    return [
        "api" if listing.section == vendor else kind
        for listing in listings()
        for head, _, kind in [listing.section.partition("-")]
        if head == vendor
    ]


def test_each_vendor_offers_its_api_and_another_way_in() -> None:
    assert {vendor: _kinds(vendor)[0] for vendor in _VENDORS} == dict.fromkeys(
        _VENDORS,
        "api",
    )
    assert all(len(_kinds(vendor)) > 1 for vendor in _VENDORS)


@pytest.mark.parametrize(
    ("spelling", "name"),
    [
        (f"{short}{kind}", vendor if kind == "api" else f"{vendor}-{kind}")
        for vendor, shorts in _VENDORS.items()
        for short in shorts
        for kind in _kinds(vendor)
    ],
)
def test_a_vendor_short_or_whole_then_its_transport_names_the_provider(
    spelling: str,
    name: str,
) -> None:
    """``aapi``, ``antcli``, ``oaisub``, ``googleapi``: the API is the bare vendor."""
    assert ModelCatalog().name(spelling) == name


@pytest.mark.parametrize(
    ("spelling", "message"),
    [
        ("oai", r"^provider 'oai' could be any of: openai, openai-sub$"),
        ("sub", r"^unknown provider 'sub'; known: "),
        ("", r"^unknown provider ''; known: "),
        ("nasub", r"^unknown provider 'nasub'; known: "),
    ],
)
def test_a_spelling_no_single_provider_answers_to_is_refused(
    spelling: str,
    message: str,
) -> None:
    with pytest.raises(UnknownProviderError, match=message):
        ModelCatalog().name(spelling)


def test_allowing_narrows_the_catalog_to_the_providers_named() -> None:
    catalog = ModelCatalog()
    assert catalog.allow(["OpenAISubscription", "antcli", "openai-sub"]) == [
        "openai-sub",
        "anthropic-cli",
    ]
    assert list(catalog.factories) == ["openai-sub", "anthropic-cli"]
    with pytest.raises(UnknownProviderError, match="known: openai-sub, anthropic-cli"):
        catalog["anthropic"]


def test_allowing_names_every_spelling_that_answers_to_nothing() -> None:
    catalog = ModelCatalog()
    with pytest.raises(
        UnknownProviderError,
        match=r"^unknown providers ghost, zed; known: anthropic, ",
    ):
        catalog.allow(["zed", "openai", "ghost"])
    assert list(catalog.factories) == list(ModelCatalog().factories)


class _Serving(ScriptedProvider):
    """Serves only the models in ``serves``, refusing the rest as a table does."""

    def __init__(
        self,
        *,
        serves: frozenset[str],
        name: str = "",
        resources: ProviderSharedResources | None = None,
    ) -> None:
        super().__init__(name=name, resources=resources)
        self.serves = serves

    @override
    def __getitem__(self, model: str) -> ScriptedModel:
        if model not in self.serves:
            raise UnknownModelError(f"{self.name} has no {model!r}.")
        return super().__getitem__(model)


def _serving() -> ModelCatalog:
    return ModelCatalog(
        factories={
            "anthropic": partial(_Serving, serves=frozenset({"opus"})),
            "anthropic-cli": partial(_Serving, serves=frozenset({"opus"})),
            "openai-sub": partial(_Serving, serves=frozenset({"gpt"})),
            "google": partial(_Serving, serves=frozenset({"gemini"})),
            "selfhosted": partial(
                _Serving,
                serves=frozenset({"local/m", "google/gemma"}),
            ),
        },
    )


@pytest.mark.parametrize(
    ("name", "current", "served_by"),
    [
        ("opus", "anthropic-cli", "anthropic-cli"),
        ("opus", "openai-sub", "anthropic"),
        ("gpt", "anthropic-cli", "openai-sub"),
        ("anthropic-cli/opus", "openai-sub", "anthropic-cli"),
        ("antcli/opus", "anthropic", "anthropic-cli"),
        ("local/m", "anthropic-cli", "selfhosted"),
        ("selfhosted/google/gemma", "anthropic-cli", "selfhosted"),
    ],
)
def test_a_model_comes_from_the_provider_named_else_current_else_the_first_serving_it(
    name: str,
    current: str,
    served_by: str,
) -> None:
    assert _serving().model(name, current=current).provider.name == served_by


def test_the_allow_list_orders_which_provider_serves_a_bare_model() -> None:
    catalog = _serving()
    catalog.allow(["openai-sub", "anthropic-cli", "anthropic"])
    assert catalog.model("opus", current="openai-sub").provider.name == "anthropic-cli"


@pytest.mark.parametrize(
    ("name", "error"),
    [
        ("nope", "anthropic-cli has no 'nope'."),
        ("openai-sub/opus", "openai-sub has no 'opus'."),
    ],
)
def test_a_model_no_provider_serves_raises_the_first_refusal(
    name: str,
    error: str,
) -> None:
    with pytest.raises(UnknownModelError, match=f"^{re.escape(error)}"):
        _serving().model(name, current="anthropic-cli")


class _AnyRepo(ScriptedProvider):
    """Serves any ``ORG/NAME``, as SelfHosted takes one for a Hugging Face repo."""

    @override
    def __getitem__(self, model: str) -> ScriptedModel:
        if "/" not in model:
            raise UnknownModelError(f"{model!r} is no repo id.")
        return super().__getitem__(model)


@pytest.mark.parametrize("name", ["anthropic/opus-9", "openai-sub/opus"])
def test_a_provider_named_that_refuses_is_the_answer(name: str) -> None:
    """A typo in a model a provider was named for never downloads a repo instead."""
    catalog = ModelCatalog(
        factories={
            "anthropic": partial(_Serving, serves=frozenset({"opus"})),
            "openai-sub": partial(_Serving, serves=frozenset({"gpt"})),
            "selfhosted": _AnyRepo,
        },
    )
    with pytest.raises(UnknownModelError, match="selfhosted/"):
        catalog.model(name, current="anthropic")


def _windowed() -> ScriptedProvider:
    """Serve ``wide-1.0``, with a 1M window, and ``narrow-1.0``, with 200k."""

    def row(model_id: str, window: int, tiers: frozenset[str]) -> ModelCapability:
        prices = PriceCatalog(window=window)
        return ModelCapability(
            model_id=model_id,
            limits=ModelLimits(max_request_tokens=window),
            prices=prices,
            thinking_effort=frozenset({"none", "high"}),
            thinking_budget=frozenset({"none", "auto"}),
            service_tier=frozenset(t for t in ("auto", "priority") if t in tiers),
            default=ModelSettings(context_size=prices["auto", "fullcontext"]),
        )

    rows = (
        row("wide-1.0", 1_000_000, frozenset({"auto", "priority"})),
        row("narrow-1.0", 200_000, frozenset({"auto"})),
    )
    return ScriptedProvider(models=ModelTable(rows=rows))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("current", "name", "context"),
    [
        ("wide+300k", "wide", 300_000),
        ("wide+300k", "wide+200k", 200_000),
        ("wide", "wide", 1_000_000),
        ("wide+300k", "narrow", 200_000),
    ],
)
async def test_a_successor_keeps_a_chosen_size_unless_named(
    current: str,
    name: str,
    context: int,
) -> None:
    provider = _windowed()
    after = await successor(provider.__getitem__, name, current=provider[current])
    assert after.settings.context_size.tokens == context


@pytest.mark.asyncio
async def test_a_successor_adopts_the_settings_and_takes_a_named_effort() -> None:
    provider = _windowed()
    current = provider["wide"]
    current.settings = replace(current.settings, thinking_effort="high")
    kept = await successor(provider.__getitem__, "narrow", current=current)
    named = await successor(provider.__getitem__, "narrow+none", current=current)
    assert (kept.settings.thinking_effort, named.settings.thinking_effort) == (
        "high",
        "none",
    )
    with pytest.raises(ValueError, match="thinking_effort='low' is not offered"):
        await successor(provider.__getitem__, "narrow+low", current=current)


@pytest.mark.asyncio
async def test_a_successor_takes_a_named_tier_in_any_position() -> None:
    provider = _windowed()
    current = provider["wide"]
    after = await successor(provider.__getitem__, "wide+fast+200k", current=current)
    assert (after.settings.service_tier, after.settings.context_size.tokens) == (
        "priority",
        200_000,
    )
    with pytest.raises(ValueError, match="service_tier='flex' is not offered"):
        await successor(provider.__getitem__, "wide+flex", current=current)


@pytest.mark.asyncio
async def test_a_successor_takes_the_choices_its_name_spells() -> None:
    provider = _windowed()
    after = await successor(
        provider.__getitem__,
        "wide budget=auto",
        current=provider["wide"],
    )
    assert after.settings.thinking_budget == "auto"
    message = "budget: 'fixed' is not one of: none, auto"
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        await successor(provider.__getitem__, "wide budget=fixed", current=after)
    with pytest.raises(ValueError, match=r"^speed is no choice"):
        await successor(provider.__getitem__, "wide speed=1", current=after)


@pytest.mark.asyncio
async def test_a_successor_restores_what_a_model_was_left_with() -> None:
    provider = _windowed()
    left = provider["narrow+100k"]
    left.settings = replace(left.settings, thinking_effort="high")
    cache = {
        "scripted/narrow-1.0": CachedModel(
            capability=left.capability,
            settings=left.settings,
        ),
    }
    current = provider["wide"]

    back = await successor(
        provider.__getitem__,
        "narrow",
        current=current,
        model_cache=cache,
    )
    tagged = await successor(
        provider.__getitem__,
        "narrow+none",
        current=current,
        model_cache=cache,
    )

    fresh = provider["narrow"]
    again = await successor(
        provider.__getitem__,
        "narrow",
        current=back,
        model_cache={
            "scripted/narrow-1.0": CachedModel(
                capability=fresh.capability,
                settings=fresh.settings,
            ),
        },
    )

    assert (back.settings.thinking_effort, back.settings.context_size.tokens) == (
        "high",
        100_000,
    )
    assert tagged.settings.thinking_effort == "none"
    assert back.settings is not left.settings
    assert again.settings.context_size.tokens == 100_000


def test_describe_spells_the_model_as_its_name_takes_it() -> None:
    model = _windowed()["wide+300k"]
    model.settings = replace(
        model.settings,
        service_tier="priority",
        thinking_budget="auto",
    )
    assert (
        str(
            ModelTags.from_settings(
                model.capability,
                model.settings,
                provider="scripted",
                account="work",
            ),
        )
        == (
            "work@scripted/wide-1.0+300k+nothink+priority budget=auto"  # codespell:ignore nothink
        )
    )


@pytest.mark.asyncio
async def test_recall_builds_the_model_and_its_cache_from_their_lines() -> None:
    catalog = ModelCatalog(
        factories={"scripted": partial(ScriptedProvider, models=_windowed().models)},
    )
    memory = ModelMemory(
        model="scripted/wide-1.0+300k+highthink budget=auto",
        cache=("scripted/narrow-1.0+highthink",),
    )

    model, cache = await recall(memory, catalog=catalog)

    settings = model.settings
    assert (
        settings.context_size.tokens,
        settings.thinking_effort,
        settings.thinking_budget,
    ) == (
        300_000,
        "high",
        "auto",
    )
    assert list(cache) == ["scripted/narrow-1.0"]
    assert cache["scripted/narrow-1.0"].settings.thinking_effort == "high"
    assert (
        str(ModelTags.from_settings(model.capability, settings, provider="scripted"))
        == memory.model
    )


@pytest.mark.asyncio
async def test_recall_reads_a_line_journaled_before_the_tags_were_renamed() -> None:
    """A resumed session's journal says ``+large+high``; it means ``+fullcontext+highthink``."""
    catalog = ModelCatalog(
        factories={"scripted": partial(ScriptedProvider, models=_windowed().models)},
    )
    memory = ModelMemory(
        model="scripted/wide-1.0+large+high budget=auto",
        cache=("scripted/narrow-1.0+high",),
    )

    model, cache = await recall(memory, catalog=catalog)

    assert str(
        ModelTags.from_settings(model.capability, model.settings, provider="scripted"),
    ) == ("scripted/wide-1.0+fullcontext+highthink budget=auto")
    assert cache["scripted/narrow-1.0"].settings.thinking_effort == "high"


@pytest.mark.asyncio
async def test_recall_closes_a_model_its_line_sets_as_no_longer_offered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[str] = []

    async def aclose(model: ScriptedModel) -> None:
        closed.append(model.capability.model_id)

    monkeypatch.setattr(ScriptedModel, "aclose", aclose)
    catalog = ModelCatalog(
        factories={"scripted": partial(ScriptedProvider, models=_windowed().models)},
    )

    with pytest.raises(ValueError, match=r"^budget: 'never' is not one of"):
        await recall(
            ModelMemory(model="scripted/wide-1.0 budget=never"),
            catalog=catalog,
        )

    assert closed == ["wide-1.0"]


@pytest.mark.parametrize(
    ("provider", "model", "stops"),
    [
        # Effort is ``low`` to ``max``; ``none`` is no level Anthropic has.
        ("anthropic", "opus-5.5", False),
        ("anthropic", "opus-4.6", False),
        ("anthropic", "opus-4.5", False),
        # Gemini 3 and 2.5 Pro think whatever is asked; 2.5 Flash takes a 0 budget.
        ("google", "gemini-pro-3.1", False),
        ("google", "gemini-flash-3.6", False),
        ("google", "gemini-pro-2.5", False),
        ("google", "gemini-flash-2.5", True),
        ("moonshot", "kimi-3.0", False),
        ("dashscope", "qwen-2400b-a95b-3.8", False),
        ("dashscope", "qwen-max-3.8", True),
        ("openai", "astra-6.0", False),
        ("openai", "sol-6.1", False),
        ("openai", "luna-6.0", True),
    ],
)
def test_an_effort_of_none_is_offered_only_where_the_vendor_stops_reasoning(
    provider: str,
    model: str,
    stops: bool,
) -> None:
    effort = ModelCatalog()[provider][model].capability.thinking_effort
    assert ("none" in effort) is stops


@pytest.mark.asyncio
async def test_closing_closes_each_provider_before_what_they_share() -> None:
    built: list[_Recorded] = []
    async with ModelCatalog(
        factories={"a": _Factory(built=built), "b": _Factory(built=built)},
    ) as catalog:
        client = catalog.http
        catalog["a"]
        catalog["b"]
    assert [provider.closed_with for provider in built] == [client, client]
    assert client.is_closed


@pytest.mark.asyncio
async def test_a_closed_catalog_builds_afresh() -> None:
    built: list[_Recorded] = []
    catalog = ModelCatalog(factories={"a": _Factory(built=built)})
    first = catalog["a"]
    await catalog.aclose()
    assert catalog["a"] is not first
    await catalog.aclose()
    assert [provider.closes for provider in built] == [1, 1]


@pytest.mark.asyncio
async def test_a_provider_failing_to_close_leaves_none_of_the_rest_open() -> None:
    built: list[_Recorded] = []
    catalog = ModelCatalog(
        factories={"a": _Factory(built=built), "b": _Factory(built=built)},
    )
    client = catalog.http
    catalog["a"]
    catalog["b"]
    built[0].fails = True
    with pytest.raises(RuntimeError, match=r"^a would not close$"):
        await catalog.aclose()
    assert [provider.closes for provider in built] == [1, 1]
    assert client.is_closed


class _Recorded(AsyncResource):
    """A provider serving no model, which notes what it saw when it closed."""

    def __init__(self, *, name: str, resources: ProviderSharedResources) -> None:
        self._name = name
        self._resources = resources
        self.closes = 0
        self.fails = False
        self.closed_with: httpx2.AsyncClient | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def resources(self) -> ProviderSharedResources:
        return self._resources

    @property
    def usage(self) -> tuple[UsageWindow, ...]:
        return ()

    def __getitem__(self, model: str) -> Model:
        raise KeyError(model)

    def __iter__(self) -> Iterator[str]:
        return iter(())

    @override
    async def aclose(self) -> None:
        self.closes += 1
        # The client a provider still uses is open until every provider closes.
        client = self._resources.http
        self.closed_with = None if client.is_closed else client
        if self.fails:
            raise RuntimeError(f"{self._name} would not close")


class _Factory:
    """Builds ``_Recorded`` providers, keeping each it built."""

    def __init__(self, *, built: list[_Recorded]) -> None:
        self._built = built

    def __call__(self, *, name: str, resources: ProviderSharedResources) -> _Recorded:
        provider = _Recorded(name=name, resources=resources)
        self._built.append(provider)
        return provider


@pytest.mark.parametrize(
    "field",
    [
        "thinking_effort",
        "thinking_budget",
        "thinking_output",
        "service_tier",
        "cache_ttl_sec",
        "manage_context_server_side",
    ],
)
def test_each_choice_is_picked_by_how_it_is_spelled(field: str) -> None:
    model = ModelCatalog()["anthropic"]["opus-5.5"]
    values = offered(model.capability, field)
    assert chosen(model.settings, field) in values
    for value in values:
        assert pick(model.capability, field, f" {spelled(value).upper()} ") == value
        assert chosen(choose(model.settings, field, value), field) == value


@pytest.mark.parametrize(
    ("field", "value"),
    [("speed", 1), ("cache_ttl_sec", "300"), ("manage_context_server_side", 1)],
)
def test_choosing_what_no_field_holds_is_refused(field: str, value: object) -> None:
    settings = ModelCatalog()["anthropic"]["opus-5.5"].settings
    with pytest.raises(ValueError, match=f"^{field}={value!r} is no choice$"):
        choose(settings, field, value)


def test_a_repo_whose_org_names_a_provider_says_so_only_when_one_hosts_repos() -> None:
    """Without SelfHosted, a refused ``provider/model`` is the provider's refusal."""
    catalog = ModelCatalog(
        factories={"anthropic": partial(_Serving, serves=frozenset({"opus"}))},
    )
    with pytest.raises(UnknownModelError, match=r"^anthropic has no 'gone'\.$"):
        catalog.model("anthropic/gone", current="anthropic")


@pytest.mark.parametrize(
    ("account", "line"),
    [
        (None, "scripted/wide-1.0+300k+nothink"),  # codespell:ignore nothink
        ("work", "work@scripted/wide-1.0+300k+nothink"),  # codespell:ignore nothink
    ],
)
def test_a_memory_names_the_account_its_model_signs_in_as(
    account: str | None,
    line: str,
) -> None:
    provider = _windowed()
    model = provider["wide+300k"]
    other = provider["narrow"]
    memory = ModelMemory.from_model(
        model,
        {
            "scripted/narrow-1.0": CachedModel(
                capability=other.capability,
                settings=other.settings,
            ),
            "scripted/wide-1.0": CachedModel(
                capability=model.capability,
                settings=model.settings,
            ),
        },
        account=account,
    )
    assert (memory.model, memory.account) == (line, account)
    assert memory.cache == (
        "scripted/narrow-1.0+fullcontext+nothink",  # codespell:ignore nothink
    )


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
