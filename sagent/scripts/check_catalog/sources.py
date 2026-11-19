"""Compare a provider catalog with what its vendor publishes.

Each ``check_*`` reads the vendor's own pages through ``fetch`` and returns one
``Finding`` per fact it compares. A fact a page no longer states, because its
layout changed or it never said, is reported ``unstated``, never a silent match.

Typical usage example:

  findings = check_openai(table, fetch=fetch, today=date.today())
  [f for f in findings if f.status == "mismatch"]
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING, Literal

import html
import re
import time

from sagent.model.cost import ServiceTier, TokenPrice


if TYPE_CHECKING:
    from sagent.model.capability import ModelCapability
    from sagent.providers.table import ModelTable


__all__ = [
    "Fetch",
    "Finding",
    "Status",
    "check_anthropic",
    "check_google",
    "check_openai",
    "waivers",
]


type Fetch = Callable[[str], str]
type Status = Literal["ok", "mismatch", "unstated", "waived"]
type _Facts = dict[str, tuple[object, object | None]]


@dataclass(frozen=True, slots=True, kw_only=True)
class Finding:
    """One fact as the catalog states it, beside what its source says."""

    provider: str
    model_id: str
    field: str
    catalog: str
    source: str | None
    """What the source says; ``None`` when it says nothing about the fact."""

    url: str
    waiver: str = ""
    """Why the catalog departs from the source on purpose; empty for none."""

    @property
    def status(self) -> Status:
        """Whether the two agree, disagree, disagree on purpose, or cannot be told."""
        if self.source is None:
            return "unstated"
        if self.catalog == self.source:
            return "ok"
        return "waived" if self.waiver else "mismatch"


def check_anthropic(table: ModelTable, *, fetch: Fetch, today: date) -> list[Finding]:
    """Return the findings for every row of an Anthropic catalog section.

    Args:
      table: The section's rows.
      fetch: Returns a URL's text.
      today: The day whose prices are compared.

    Returns:
      findings: Each fact of each row, beside its model page and guides.

    """
    docs = "https://platform.claude.com/docs/en"
    guide = f"{docs}/build-with-claude/thinking.md"
    budgets = _claude_budgets(fetch(guide))
    efforts = _claude_efforts(fetch(f"{docs}/build-with-claude/effort.md"))
    fast = _claude_fast(fetch(f"{docs}/about-claude/pricing.md"))
    vision = fetch(f"{docs}/build-with-claude/vision.md")
    # "MB" is 2**20 bytes: 134,217,728 bytes were taken and one more refused
    # (413), measured 2026-11-18; the image cap is read the same way.
    mib = 1 << 20
    request_mb = _count(
        _match(
            r"^\| Messages, Token Counting\s*\|\s*(\d+)\s+MB",
            fetch(f"{docs}/api/overview.md"),
        ),
    )
    # The guide puts a no-break space between a number and its unit.
    image_mb = _count(
        _match(r"(\d+)\s+MB \(base64-encoded\) on the Claude API", vision),
    )
    edges = _claude_image_edges(vision)
    findings: list[Finding] = []
    names: set[str] = set()
    for row in table.rows:
        url = f"{docs}/models/{row.wire_model_id.removeprefix('claude-')}/overview.md"
        page = fetch(url)
        cells = _cells(page)
        # A page without a title names no model, whatever stands in for it.
        name = _match(r"^title: (.+)$", page) or ""  # pragma: no mutate
        names.add(name)
        thinking = budgets.get(name)
        facts: _Facts = {
            "knowledge_cutoff": (
                row.knowledge_cutoff,
                _month(cells.get("Reliable knowledge cutoff")),
            ),
            "max_request_tokens": (
                row.limits.max_request_tokens,
                _count(cells.get("Context window")),
            ),
            "max_response_tokens": (
                row.limits.max_response_tokens,
                _count(cells.get("Max output")),
            ),
            "thinking_budget": (
                row.thinking_budget,
                None if thinking is None else thinking.budgets,
            ),
            "thinking_budget_default": (
                row.default.thinking_budget,
                None if thinking is None else thinking.unasked,
            ),
            # Where neither side lets thinking be off, the cap means nothing.
            **(
                {
                    "thinking_off_max_effort": (
                        row.thinking_off_max_effort or "any",
                        thinking.off_max_effort,
                    ),
                }
                if thinking is not None
                and ("none" in thinking.budgets or row.thinking_off_max_effort)
                else {}
            ),
            "thinking_effort": (
                row.thinking_effort,
                _claude_effort_levels(name, row.wire_model_id, efforts=efforts),
            ),
            "thinking_effort_default": (
                row.default.thinking_effort,
                _claude_default_effort(cells.get("Default effort")),
            ),
        }
        facts |= _claude_prices(row, cells=cells, today=today)
        facts |= _claude_fast_prices(row, fast=fast.get(name), today=today)
        facts |= {
            "max_request_bytes": (
                row.limits.max_request_bytes,
                None if request_mb is None else request_mb * mib,
            ),
            # A base64 image of n bytes holds 3n/4 raw ones, the cap sagent holds.
            "max_image_bytes": (
                row.limits.max_image_bytes,
                None if image_mb is None else image_mb * mib * 3 // 4,
            ),
            "max_image_edge_px": (
                row.limits.max_image_edge_px,
                _claude_image_edge(name, edges=edges),
            ),
        }
        findings += _findings("anthropic", row, url=url, facts=facts)
    # The thinking guide's table names every model the API serves; a page that
    # names no model leaves each one it lists maybe that one.
    return findings + _uncatalogued(
        "anthropic",
        set(budgets) - names,
        url=guide,
        known="" not in names,
    )


def check_openai(table: ModelTable, *, fetch: Fetch, today: date) -> list[Finding]:
    """Return the findings for every row of an OpenAI catalog section.

    Args:
      table: The section's rows.
      fetch: Returns a URL's text.
      today: The day whose prices are compared.

    Returns:
      findings: Each fact of each row, beside its model page and the price page.

    """
    docs = "https://developers.openai.com/api/docs"
    pricing = f"{docs}/pricing.md"
    prices = _openai_prices(fetch(pricing))
    extended = _openai_extended(fetch(f"{docs}/guides/prompt-caching.md"))
    findings: list[Finding] = []
    served: set[str] = set()
    for row in table.rows:
        url = f"{docs}/models/{row.wire_model_id}.md"
        page = fetch(url)
        # An alias's page is its target's; the price table names only the target.
        model = _match(r"^Model ID: `([^`]+)`", page) or row.wire_model_id
        served |= {row.wire_model_id, model}
        levels, default = _openai_efforts(page)
        cap = _count(_match(r"^- Maximum input tokens: ([\d,]+)", page))
        window = _count(_match(r"^- ([\d,]+) context window", page))
        cutoff = _match(r"^- (\w{3} \d{1,2}, \d{4}) knowledge cutoff", page)
        # Measured 2026-10-27 on 1.05M windows: 921,013 input tokens were served
        # and 921,913 refused though the page says 922,000, so the catalog keeps
        # 1,000 back there; no smaller window was measured short.
        margin = 1_000 if window == 1_050_000 else 0
        facts: _Facts = {
            "knowledge_cutoff": (
                row.knowledge_cutoff,
                None if cutoff is None else _day(cutoff),
            ),
            "context_tokens": (row.limits.context_tokens, window),
            "max_request_tokens": (
                row.limits.max_request_tokens,
                None if cap is None else cap - margin,
            ),
            "max_response_tokens": (
                row.limits.max_response_tokens,
                _count(_match(r"^- ([\d,]+) max output tokens", page)),
            ),
            "thinking_effort": (row.thinking_effort, levels),
            "thinking_effort_default": (row.default.thinking_effort, default),
            "cache_ttl_sec": (
                row.cache_ttl_sec,
                _openai_cache_ttl(model, extended=extended),
            ),
        }
        facts |= _openai_row_prices(row, cards=prices.get(model, {}), today=today)
        findings += _findings("openai", row, url=url, facts=facts)
    # Text models only: the price tables also list audio, image, and realtime
    # models, and each model again under its dated snapshots.
    text = {
        m
        for m in prices
        if re.fullmatch(r"(?:gpt-\d|o\d)[\w.-]*", m)
        and not re.search(
            r"-(?:\d{4}-\d{2}-\d{2}|audio|image|realtime|transcribe|tts|live"
            r"|diarize|translate|whisper)",
            m,
        )
    }
    return findings + _uncatalogued("openai", text - served, url=pricing)


def check_google(table: ModelTable, *, fetch: Fetch, today: date) -> list[Finding]:
    """Return the findings for every row of a Google catalog section.

    Args:
      table: The section's rows.
      fetch: Returns a URL's text.
      today: The day whose prices are compared.

    Returns:
      findings: Each fact of each row, beside its model page, card, and guides.

    """
    docs = "https://aistudio.google.com/docs"
    guide = f"{docs}/thinking.md"
    thinking = _gemini_thinking(fetch(guide))
    prices = _gemini_prices(fetch(f"{docs}/pricing.md"))
    findings: list[Finding] = []
    for row in table.rows:
        url = f"{docs}/models/{row.wire_model_id}.md"
        page = fetch(url)
        # Gemini 4 states one window shared by input and output; earlier models
        # state an input limit, their context then being ``0`` (the input bounds both).
        shared = _count(_match(r"input and output token limit\)\*\* ([\d,]+)", page))
        apart = _count(_match(r"\*\*Input token limit\*\* ([\d,]+)", page))
        facts: _Facts = {
            "knowledge_cutoff": (row.knowledge_cutoff, _gemini_cutoff(page, fetch)),
            "max_request_tokens": (row.limits.max_request_tokens, shared or apart),
            "max_context_tokens": (
                row.limits.max_context_tokens,
                shared if apart is None else 0,
            ),
            "max_response_tokens": (
                row.limits.max_response_tokens,
                _count(
                    _match(r"\*\*(?:Output token limit|Max output)\*\* ([\d,]+)", page),
                ),
            ),
        }
        facts |= _gemini_thinking_facts(
            row,
            row_thinking=thinking.get(row.wire_model_id),
        )
        facts |= _gemini_row_prices(
            row,
            cards=prices.get(row.wire_model_id),
            today=today,
        )
        findings += _findings("google", row, url=url, facts=facts)
    # The thinking guide's table names every model that reasons; an image
    # model in it answers no chat.
    listed = {m for m in thinking if not m.endswith("-image")}
    served = {row.wire_model_id for row in table.rows}
    return findings + _uncatalogued("google", listed - served, url=guide)


def waivers() -> Mapping[tuple[str, str, str], str]:
    """Return the facts each catalog departs from on purpose, with the reason.

    Returns:
      waivers: Each reason, by provider, model, and field.

    """
    claude_code = (
        "Claude Code's default, which anthropic.yaml starts at"  # pragma: no mutate
    )
    codex = "Codex's default, which openai.yaml starts at"  # pragma: no mutate
    image_cap = (
        "10 MB read as 10**6, below the 2**20 reading: unmeasured, and an"  # pragma: no mutate
        " image shrunk a little more costs less than one refused"  # pragma: no mutate
    )
    # Claude Code turns thinking on for the 4.x models, which think only when asked.
    budgets = {
        ("anthropic", model_id, "thinking_budget_default"): claude_code
        for model_id in (
            "opus-4.8",
            "opus-4.7",
            "opus-4.6",
            "opus-4.5",
            "sonnet-4.6",
            "sonnet-4.5",
            "haiku-4.5",
        )
    }
    return (
        budgets
        | {
            # ``*`` is every model of the provider.
            ("anthropic", "*", "max_image_bytes"): image_cap,
            # A model left out of a catalog, by the id its vendor lists it under.
            ("anthropic", "Claude Mythos Preview", "catalogued"): (
                "a research preview served only to invited partners"  # pragma: no mutate
            ),
            ("google", "gemini-3-pro-preview", "catalogued"): (
                "shut down; its page says so"  # pragma: no mutate
            ),
            ("google", "gemini-robotics-er-2-preview", "catalogued"): (
                "an embodied-reasoning model for robots, not a chat model"  # pragma: no mutate
            ),
            ("anthropic", "fable-5.2", "thinking_effort_default"): claude_code,
            ("anthropic", "sonnet-5.5", "thinking_effort_default"): claude_code,
            ("anthropic", "opus-4.7", "thinking_effort_default"): claude_code,
            ("openai", "sol-6.2", "thinking_effort_default"): codex,
            ("openai", "sol-6.1", "thinking_effort_default"): codex,
            ("openai", "sol-5.6", "thinking_effort_default"): codex,
            ("openai", "gpt-5.6", "thinking_effort_default"): codex,
        }
    )


def _waiver(provider: str, model: str, field: str) -> str:
    """Return why ``field`` of ``model`` departs on purpose, its own or the provider's."""
    known = waivers()
    return known.get((provider, model, field)) or known.get((provider, "*", field), "")


# Each is unstated where the catalog's own models are not ``known`` by name.
def _uncatalogued(
    provider: str,
    models: set[str],
    *,
    url: str,
    known: bool = True,
) -> list[Finding]:
    """Return one finding per model the vendor lists at ``url`` and the catalog lacks."""
    return [
        Finding(
            provider=provider,
            model_id=model,
            field="catalogued",
            catalog="absent",
            source="listed" if known else None,
            url=url,
            waiver=_waiver(provider, model, "catalogued"),
        )
        for model in sorted(models)
    ]


def _findings(
    provider: str,
    row: ModelCapability,
    *,
    url: str,
    facts: _Facts,
) -> Iterator[Finding]:
    """Yield one finding per fact, each side as text, waived where on purpose."""
    for field_name, (catalog, source) in facts.items():
        yield Finding(
            provider=provider,
            model_id=row.model_id,
            field=field_name,
            catalog=_text(catalog),
            source=None if source is None else _text(source),
            url=url,
            waiver=_waiver(provider, row.model_id, field_name),
        )


def _text(value: object) -> str:
    """Return ``value`` as one comparable string: sets sorted, floats shortest."""
    if isinstance(value, frozenset):
        return ",".join(sorted(str(v) for v in value))
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _match(pattern: str, text: str) -> str | None:
    """Return the first group of the first line ``pattern`` matches, or ``None``."""
    found = re.search(pattern, text, flags=re.MULTILINE)
    return None if found is None else str(found[1]).strip()


def _all(pattern: str, text: str) -> list[str]:
    """Return the first group of every match of ``pattern``, line-anchored."""
    return [str(m[1]) for m in re.finditer(pattern, text, flags=re.MULTILINE)]


# Each count and rate pattern takes only text ``float`` reads: a count groups its
# thousands, a rate in dollars (``\d+(?:\.\d+)?``) does not.
def _count(text: str | None) -> int | None:
    """Return ``1M``, ``128K``, or ``1,050,000`` (any trailing words) as a count."""
    if text is None:
        return None
    found = re.match(r"(\d[\d,]*(?:\.\d+)?)\s*([KM]?)", text.strip())
    if found is None:
        return None
    scale = {"": 1, "K": 1_000, "M": 1_000_000}[found[2]]
    return round(float(found[1].replace(",", "")) * scale)


def _month(text: str | None) -> str | None:
    """Return ``Jun 2026`` or ``March 2026`` as ``2026-06``; ``None`` for no month."""
    found = None if text is None else re.search(r"([A-Z][a-z]+) (\d{4})", text)
    if found is None:
        return None
    day = _day(f"{found[1]} 1, {found[2]}")
    return None if day is None else day[: len("2026-06")]


def _day(text: str) -> str | None:
    """Return ``Jun 01, 2026`` or ``March 1, 2027`` as ``2026-06-01``; ``None`` for no day."""
    month, _, rest = text.partition(" ")
    try:
        parsed = time.strptime(f"{month[:3]} {rest}", "%b %d, %Y")
    except ValueError:
        return None
    return date(parsed.tm_year, parsed.tm_mon, parsed.tm_mday).isoformat()


def _cells(page: str) -> dict[str, str]:
    """Return a page's two-column table rows, ``| key | value |``, first row winning."""
    cells: dict[str, str] = {}
    for found in re.finditer(
        r"^\|([^|\n]+)\|([^|\n]+)\|\s*$",
        page,
        flags=re.MULTILINE,
    ):
        key = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", found[1]).strip()
        cells.setdefault(key, found[2].strip())
    return cells


def _price_facts(
    prefix: str,
    price: TokenPrice | None,
    stated: Mapping[str, float | None],
) -> _Facts:
    """Return each meter ``stated`` names, beside ``price``'s; ``absent`` if unpriced."""
    return {
        f"{prefix}.{meter}": (
            "absent" if price is None else _meter(price, meter),
            value,
        )
        for meter, value in stated.items()
    }


def _meter(price: TokenPrice, meter: str) -> float:
    """Return one meter's rate, by its field name."""
    rates = {
        "request": price.request,
        "response": price.response,
        "cache_write": price.cache_write,
        "cache_write_1h": price.cache_write_1h,
        "cache_read": price.cache_read,
    }
    return rates[meter]


def _price(
    row: ModelCapability,
    tier: ServiceTier,
    *,
    tokens: int,
    at: date,
) -> TokenPrice | None:
    """Return the row's price of a ``tokens`` prompt at ``tier`` on ``at``, if any."""
    try:
        return row.prices[tier, tokens, at]
    except KeyError:
        return None


# Anthropic.


@dataclass(frozen=True, slots=True, kw_only=True)
class _ClaudeThinking:
    """One model's row of the thinking guide's table."""

    budgets: frozenset[str]
    unasked: str | None
    """The budget the model thinks at when sent no ``thinking``; ``None``, unknown."""

    off_max_effort: str
    """The highest effort ``disabled`` is taken at; ``any`` when the cell names none."""


def _claude_budgets(page: str) -> dict[str, _ClaudeThinking]:
    """Return each model's budget modes, by name, from the thinking guide's table."""
    unasked = {"Thinking off": "none", "Adaptive": "auto", "Extended": "fixed"}
    budgets: dict[str, _ClaudeThinking] = {}
    for found in re.finditer(
        r"^\| (Claude [^|]+?)\s*\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|\s*$",
        page,
        flags=re.MULTILINE,
    ):
        # ``between_tools`` (column 5) has no budget of sagent's to compare.
        omitted, adaptive, enabled, disabled = found[2], found[3], found[4], found[6]
        offered = {
            mode
            for mode, cell in (
                ("auto", adaptive),
                ("fixed", enabled),
                ("none", disabled),
            )
            if "400 error" not in cell
        }
        budgets.setdefault(
            found[1],
            _ClaudeThinking(
                budgets=frozenset(offered),
                unasked=next(
                    (
                        b
                        for cell, b in unasked.items()
                        if omitted.strip().startswith(cell)
                    ),
                    None,
                ),
                off_max_effort=_match(r"at `(\w+)` effort or below", disabled) or "any",
            ),
        )
    return budgets


def _claude_image_edges(page: str) -> tuple[int | None, int | None]:
    """Return the vision guide's long edge for 4.7 and later models, and for the rest."""
    later = _count(
        _match(r"^\| [\w-]+\s*\| Claude 4\.7 and later models\s*\| ([\d,]+) px", page),
    )
    other = _count(_match(r"^\| [\w-]+\s*\| All other models\s*\| ([\d,]+) px", page))
    return later, other


def _claude_image_edge(
    name: str,
    *,
    edges: tuple[int | None, int | None],
) -> int | None:
    """Return the long edge a model's resolution tier downscales to; ``None`` unstated."""
    version = re.fullmatch(r"Claude \w+ (\d+)(?:\.(\d+))?", name)
    if version is None:
        return None
    later, other = edges
    # A missing minor read as 0 or 1 falls the same side of 4.7: (4, 0) and (4, 1).
    minor = int(version[2] or 0)  # pragma: no mutate
    return later if (int(version[1]), minor) >= (4, 7) else other


@dataclass(frozen=True, slots=True, kw_only=True)
class _ClaudeEfforts:
    """The effort guide: which ids take effort, and which names take each top level."""

    ids: tuple[str, ...]
    xhigh: frozenset[str]
    max: frozenset[str]


def _claude_efforts(page: str) -> _ClaudeEfforts:
    """Return the effort guide's supported ids and its ``xhigh`` and ``max`` names."""
    # Only the front matter's model list: platform lists follow it.
    front = _match(r"(?s)\A(.*?)supportedPlatforms:", page)
    return _ClaudeEfforts(
        ids=() if front is None else tuple(_all(r"^\s+- (claude-[\w-]+)$", front)),
        xhigh=_claude_names(_match(r"^\| `xhigh`\s*\|([^|]+)\|", page)),
        max=_claude_names(_match(r"^\| `max`\s*\|([^|]+)\|", page)),
    )


def _claude_names(text: str | None) -> frozenset[str]:
    """Return the names in ``... Available on A, B, and C.``."""
    listed = None if text is None else _match(r"Available on (.*?)\.?$", text)
    if listed is None:
        return frozenset()
    return frozenset(
        name.strip() for name in re.split(r",\s*(?:and\s+)?|\s+and\s+", listed)
    )


def _claude_effort_levels(
    name: str,
    wire_model_id: str,
    *,
    efforts: _ClaudeEfforts,
) -> frozenset[str] | None:
    """Return the levels a model takes; ``{none}`` when it takes no effort."""
    if not efforts.ids:
        return None
    # Pinned snapshots are listed dated: ``claude-opus-4-5-20251101``.
    if not any(
        i == wire_model_id or i.startswith(f"{wire_model_id}-2") for i in efforts.ids
    ):
        return frozenset({"none"})
    levels = {"low", "medium", "high"}
    levels |= {"xhigh"} if name in efforts.xhigh else set()
    levels |= {"max"} if name in efforts.max else set()
    return frozenset(levels)


def _claude_default_effort(cell: str | None) -> str | None:
    """Return a backquoted level bare, and ``Not supported`` as ``none``."""
    if cell is None:
        return None
    return "none" if cell == "Not supported" else _match(r"`(\w+)`", cell)


def _claude_prices(
    row: ModelCapability,
    *,
    cells: Mapping[str, str],
    today: date,
) -> _Facts:
    """Return the base rates of a model page's price table, by prompt band."""
    meters = {
        "request": "Input",
        "response": "Output",
        "cache_write": "5m cache write",
        "cache_write_1h": "1h cache write",
        "cache_read": "Cache read",
    }
    # A missing cell holds no rate, whatever stands in for it.
    blank = ""  # pragma: no mutate
    stated = {meter: cells.get(cell, blank) for meter, cell in meters.items()}
    small = {m: _first_rate(text, index=0) for m, text in stated.items()}
    price = _price(
        row,
        "auto",
        tokens=1,  # pragma: no mutate -- any prompt under the band prices alike.
        at=today,
    )
    facts = _price_facts("price.auto", price, small)
    over = _count(_match(r"over ([\d,]+) tokens", stated["request"]))
    if over is not None:
        large = {m: _first_rate(text, index=1) for m, text in stated.items()}
        price = _price(
            row,
            "auto",
            tokens=over + 1,  # pragma: no mutate -- any prompt past the band alike.
            at=today,
        )
        facts |= _price_facts("price.auto.long", price, large)
    return facts


def _first_rate(text: str, *, index: int) -> float | None:
    """Return the ``index``-th ``$X / MTok`` rate of a cell, or ``None``."""
    rates = _all(r"\$(\d+(?:\.\d+)?) / MTok", text)
    return float(rates[index]) if len(rates) > index else None


def _claude_fast(page: str) -> dict[str, tuple[float, float]]:
    """Return fast mode's input and output rates, by model name."""
    fast: dict[str, tuple[float, float]] = {}
    for found in re.finditer(
        r"^\| (Claude [^|]+?)\s*\| \$(\d+(?:\.\d+)?) USD / MTok\s*"
        r"\| \$(\d+(?:\.\d+)?) USD / MTok\s*\|",
        page,
        flags=re.MULTILINE,
    ):
        for name in found[1].split(" / "):
            fast[name.strip()] = (float(found[2]), float(found[3]))
    return fast


def _claude_fast_prices(
    row: ModelCapability,
    *,
    fast: tuple[float, float] | None,
    today: date,
) -> _Facts:
    """Return fast mode's rates beside the row's ``priority`` card."""
    price = _price(
        row,
        "priority",
        tokens=1,  # pragma: no mutate -- fast mode prices every prompt alike.
        at=today,
    )
    if fast is None:
        return {"price.priority": ("absent" if price is None else "priced", "absent")}
    request, response = fast
    return _price_facts(
        "price.priority",
        price,
        {"request": request, "response": response},
    )


# OpenAI.


type _OpenAICards = dict[tuple[ServiceTier, date], dict[str, float]]


def _openai_prices(page: str) -> dict[str, _OpenAICards]:
    """Return each model's rate cards, by tier and start date, from the price tables."""
    tiers: dict[str, ServiceTier] = {
        "Standard": "auto",
        "Flex": "flex",
        "Fast": "priority",
        "Ultrafast": "ultrafast",
        "Ultrafast promotion": "ultrafast",
        "Ultrafast list": "ultrafast",
    }
    starts = _match(r"times the Standard rate starting (\w+ \d{1,2}, \d{4})", page)
    day = None if starts is None else _day(starts)
    listed = date.max if day is None else date.fromisoformat(day)
    cards: dict[str, _OpenAICards] = {}
    for section in re.split(r"^### ", page, flags=re.MULTILINE)[1:]:
        heading = _match(r"\A(.+) pricing data$", section)
        if heading not in tiers:
            continue
        at = listed if heading == "Ultrafast list" else date.min
        for model, rates in _openai_table_rows(section):
            cards.setdefault(model, {})[tiers[heading], at] = rates
    return cards


def _openai_table_rows(section: str) -> Iterator[tuple[str, dict[str, float]]]:
    """Yield each model row of a section's tables, its rates keyed by meter."""
    columns = {
        "Short context input": "request",
        "Input": "request",
        "Short context output": "response",
        "Output": "response",
        "Short context cached input": "cache_read",
        "Cached input": "cache_read",
        "Short context 30m write": "cache_write",
        "Long context input": "long.request",
        "Long context output": "long.response",
        "Long context cached input": "long.cache_read",
        "Long context 30m write": "long.cache_write",
    }
    # Sagent sends no ``prompt_cache_options.ttl``, so it never pays the 12h write.
    unread = {"Model", "Short context 12h write", "Long context 12h write"}
    # Every table opens with its header row, so no data row meets an empty header.
    header: list[str] = []  # pragma: no mutate -- replaced before it is read.
    for line in section.splitlines():
        if not line.startswith("|") or line.startswith("| ---"):
            continue
        # Cells hold no ``X``, so stripping it too changes nothing.
        inner = line.strip("|")  # pragma: no mutate
        cells = [cell.strip() for cell in inner.split("|")]
        if cells[0] == "Model":
            for name in cells:
                if name not in columns and name not in unread:
                    raise ValueError(
                        f"OpenAI's price tables have a new column {name!r}.",
                    )
            header = cells
            continue
        model = re.sub(r"\s*\(.*\)$", "", cells[0])
        # A short-context ``-`` is a meter not charged (no cache discount, no
        # write fee); a long-context ``-`` is a band the tier does not publish;
        # any other cell that is no ``$rate`` states nothing.
        rates: dict[str, float] = {}
        for name, cell in zip(header, cells, strict=True):
            rate = re.fullmatch(r"\$(\d+(?:\.\d+)?)", cell)
            if name not in columns or (cell == "-" and name.startswith("Long")):
                continue
            if cell == "-":
                rates[columns[name]] = 0.0
            elif rate is not None:
                rates[columns[name]] = float(rate[1])
        yield model, rates


def _openai_extended(page: str) -> frozenset[str] | None:
    """Return the models the caching guide gives extended retention; ``None`` unstated."""
    listed = _match(r"Extended retention is supported by (.+)\.$", page)
    return None if listed is None else frozenset(_all(r"`([\w.-]+)`", listed))


def _openai_cache_ttl(
    model: str,
    *,
    extended: frozenset[str] | None,
) -> frozenset[float] | None:
    """Return how long an unasked cache lasts: 30 minutes if retention is extended."""
    if extended is None:
        return None
    version = re.match(r"gpt-(\d+)(?:\.(\d+))?", model)
    # GPT-5.6 and later keep at least 30 minutes, listed or not. A missing minor
    # read as 0 or 1 falls the same side of 5.6: (5, 0) and (5, 1).
    recent = version is not None and (
        int(version[1]),
        int(version[2] or 0),  # pragma: no mutate
    ) >= (5, 6)
    return frozenset({1800.0 if recent or model in extended else 300.0})


def _openai_efforts(page: str) -> tuple[frozenset[str] | None, str | None]:
    """Return the effort levels a model page lists, and the one it marks default."""
    # A model that does not reason says so by lacking this bullet.
    if not re.search(r"^- Reasoning token support$", page, flags=re.MULTILINE):
        return frozenset({"none"}), "none"
    flat = " ".join(page.replace("`", "").split())
    found = re.search(
        r"effort:? supports:? ([^.]+)\.|supports reasoning\.effort: ([^.]+)\.",
        flat,
    )
    if found is None:
        return None, None
    listed = found[1] or found[2]
    # OpenAI's ``minimal`` is sagent's ``min``.
    names = {"minimal": "min"}
    levels = _all(r"\b(none|minimal|low|medium|high|xhigh|max)\b", listed)
    default = _match(r"\b(\w+) \(default\)", listed)
    return (
        frozenset(names.get(level, level) for level in levels),
        # ``names.get(None, None)`` is ``None`` too.
        None if default is None else names.get(default, default),  # pragma: no mutate
    )


def _openai_row_prices(
    row: ModelCapability,
    *,
    cards: _OpenAICards,
    today: date,
) -> _Facts:
    """Return each published card beside the row's, and the row's unpublished tiers."""
    facts: _Facts = {}
    for (tier, at), rates in cards.items():
        when = max(at, today)
        label = f"price.{tier}" if at == date.min else f"price.{tier}@{at.isoformat()}"
        short = {m: v for m, v in rates.items() if not m.startswith("long.")}
        smallest = _price(
            row,
            tier,
            tokens=1,  # pragma: no mutate -- any short prompt prices alike.
            at=when,
        )
        facts |= _price_facts(label, smallest, short)
        long = {
            m.removeprefix("long."): v
            for m, v in rates.items()
            if m.startswith("long.")
        }
        if long:
            price = _price(
                row,
                tier,
                tokens=272_001,  # pragma: no mutate -- any long prompt prices alike.
                at=when,
            )
            facts |= _price_facts(f"{label}.long", price, long)
    published = {tier for tier, _ in cards}
    for tier in sorted(row.prices.service_tiers - published - {"default"}):
        facts[f"price.{tier}"] = ("priced", "absent")
    return facts


# Google.


def _gemini_cutoff(page: str, fetch: Fetch) -> str | None:
    """Return a model page's cutoff, or else the one its model card states."""
    stated = _match(r"Knowledge cutoff\s*\{% /section %\}\s*\*\s*(\w+ \d{4})", page)
    if stated is not None:
        return _month(stated)
    card = _match(r"\[Model card\]\((https://deepmind\.google/[^)]+)\)", page)
    if card is None:
        return None
    raw = fetch(card)
    stated = _card_cutoff(raw)
    # Point releases defer their limitations to the base model's PDF card.
    limits = re.search(
        r"Known Limitations.*?href=\"?(https://[^\"\s>]+\.pdf)",
        raw,
        re.DOTALL,
    )
    if stated is None and limits is not None:
        stated = _card_cutoff(fetch(limits[1]))
    return _month(stated)


def _card_cutoff(raw: str) -> str | None:
    """Return the ``Month YYYY`` a model card's prose gives as its cutoff."""
    text = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", raw)).split())
    found = re.search(
        r"knowledge cutoff (?:date )?(?:for .{0,60}? )?(?:is|was) (\w+ \d{4})",  # pragma: no mutate[case-insensitive]
        text,
        flags=re.IGNORECASE,
    )
    return None if found is None else str(found[1])


def _gemini_thinking(page: str) -> dict[str, tuple[str, frozenset[str]]]:
    """Return each model's default thinking and its levels, by model id."""
    names = {"minimal": "min"}
    rows: dict[str, tuple[str, frozenset[str]]] = {}
    for found in re.finditer(
        r"^\| (gemini-[\w.-]+) \| ([^|]+?) \| ([^|]+?) \|$",
        page,
        flags=re.MULTILINE,
    ):
        levels = frozenset(names.get(v, v) for v in found[3].split(", "))
        rows[found[1]] = (found[2], levels)
    return rows


def _gemini_thinking_facts(
    row: ModelCapability,
    *,
    row_thinking: tuple[str, frozenset[str]] | None,
) -> _Facts:
    """Return a row's levels and default thinking beside the thinking guide's."""
    if row_thinking is None:
        return {"thinking_effort": (row.thinking_effort, None)}
    default, levels = row_thinking
    if not row.effort_as_level:
        # A budget row is steered by token count; the guide says only on or off.
        return {
            "thinking_off_by_default": (
                row.default.thinking_budget == "none",
                default == "Off",
            ),
        }
    names = {"minimal": "min"}
    level = _match(r"\((\w+)\)", default)
    return {
        "thinking_effort": (row.thinking_effort, levels),
        "thinking_effort_default": (
            row.default.thinking_effort,
            level and names.get(level, level),
        ),
    }


type _GeminiCards = dict[str, list[tuple[float, str]]]


def _gemini_prices(page: str) -> dict[str, _GeminiCards]:
    """Return each model's Standard paid rates, by meter, from the price page."""
    meters = {
        "Input price": "request",
        "Output price": "response",
        "Context caching price": "cache_read",
    }
    cards: dict[str, _GeminiCards] = {}
    for section in re.split(r"^## ", page, flags=re.MULTILINE)[1:]:
        # The models a section prices are named above its rule; later mentions
        # are cross-references ("priced the same as ...").
        head = _match(r"(?s)\A(.*?)\{% hr /%\}", section) or section
        tab = r'(?s)\{% tab title="Standard" %\}(.*?)\{% /tab %\}'
        rates: _GeminiCards = {}
        for block in (_match(tab, section) or section).split("\n  ---"):
            items = re.split(r"^  \* ", block, flags=re.MULTILINE)[1:]
            if len(items) < 3:
                continue
            feature = re.sub(r" \(.*", "", items[0].strip())
            if feature in meters:
                rates[meters[feature]] = _gemini_rates(" ".join(items[2].split()))
        for model in _all(r"\[`(gemini-[\w.-]+)`\]", head):
            cards.setdefault(model, rates)
    return cards


def _gemini_rates(text: str) -> list[tuple[float, str]]:
    """Return a paid cell's per-token rates, each tagged ``base``, ``long``, or a date."""
    rates: list[tuple[float, str]] = []
    for found in re.finditer(r"\$(\d+(?:\.\d+)?)([^$]*)", text):
        qualifier = found[2]
        if "per hour" in qualifier or qualifier.lstrip().startswith("(audio"):
            continue
        starting = _match(r"starting (\w+ \d{1,2}, \d{4})", qualifier)
        if starting is not None:
            # A start that is no date says nothing of when the rate holds.
            day = _day(starting)
            if day is not None:
                rates.append((float(found[1]), day))
        elif re.search(r">\s*200k", qualifier):
            rates.append((float(found[1]), "long"))
        else:
            rates.append((float(found[1]), "base"))
    return rates


def _gemini_row_prices(
    row: ModelCapability,
    *,
    cards: _GeminiCards | None,
    today: date,
) -> _Facts:
    """Return a row's base, long-prompt, and dated rates beside the price page's."""
    if cards is None:
        return {"price.auto": ("priced", None)}
    starts = sorted(
        {tag for rates in cards.values() for _, tag in rates} - {"base", "long"},
    )
    # The base rate is the one in force until the first dated change.
    base_day = (
        min(
            today,
            date.fromisoformat(starts[0])
            - timedelta(days=1),  # pragma: no mutate -- any day before it alike.
        )
        if starts
        else today
    )
    bands: list[tuple[str, date, int]] = [
        ("base", base_day, 1),  # pragma: no mutate -- any short prompt alike.
        ("long", base_day, 200_001),  # pragma: no mutate -- any long prompt alike.
    ]
    bands += [(s, date.fromisoformat(s), 1) for s in starts]  # pragma: no mutate
    facts: _Facts = {}
    for tag, at, tokens in bands:
        # Only the meters this band or date prices: Gemini states no long-prompt
        # output rate for a model that has none, rather than a zero.
        stated: dict[str, float | None] = {
            meter: next(rate for rate, t in rates if t == tag)
            for meter, rates in cards.items()
            if any(t == tag for _, t in rates)
        }
        if stated:
            label = "price.auto" if tag == "base" else f"price.auto.{tag}"
            price = _price(row, "auto", tokens=tokens, at=at)
            facts |= _price_facts(label, price, stated)
    return facts
