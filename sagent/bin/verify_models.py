#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Verify provider model limits.

Checks that every provider's catalog entries have correct
max_request_tokens and max_response_tokens by querying live APIs or
scraping official documentation pages.

Usage::

    uv --quiet --project . run python -m sagent.bin.verify_models
    uv --quiet --project . run python -m sagent.bin.verify_models --provider google
    uv --quiet --project . run python -m sagent.bin.verify_models --provider openai

Every model is looked up and reported by its vendor wire id
(``claude-opus-5-5``), not the catalog name (``opus-5.5``). A run that
verifies no row of a requested provider fails: a missing key or an
unreachable API is not a pass.

Sources:
- Google: GET /v1beta/models (returns inputTokenLimit, outputTokenLimit)
- OpenAI: scrapes https://developers.openai.com/api/docs/models/<id>.md
- Anthropic: GET /v1/models/<id> (returns max_tokens, max_input_tokens)

Cross-reference: https://github.com/taylorwilsdon/llm-context-limits
'''
# fmt: on

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import argparse
import asyncio
import os
import re
import sys

import httpx2

from sagent import providers
from sagent.lib.codec import from_plain
from sagent.providers.anthropic.api import Anthropic
from sagent.providers.google.api import Google
from sagent.providers.openai.api import OpenAI
from sagent.types.providers import ModelResolver


if TYPE_CHECKING:
    from collections.abc import Mapping

    from sagent.catalog.table import ModelCatalog
    from sagent.types.capability import ModelCapability


@dataclass(frozen=True, slots=True, kw_only=True)
class LiveLimits:
    """Token limits a live API reported for one model."""

    max_request_tokens: int
    max_response_tokens: int


# Source: https://ai.google.dev/api/models#method:-models.list
# Returns inputTokenLimit and outputTokenLimit per model.


async def fetch_google(api_key: str) -> dict[str, LiveLimits]:
    """Fetch model limits from the Google Generative Language API.

    Args:
      api_key: Google API key.

    Returns:
      limits: Map of model ID to its token limits; partial when a page fails.

    """
    api = "https://generativelanguage.googleapis.com/v1beta/models"
    out: dict[str, LiveLimits] = {}
    page_token = ""
    # The key rides a header: in the query string it would be echoed by every
    # HTTPStatusError message, which this tool prints.
    headers = {"x-goog-api-key": api_key}
    async with httpx2.AsyncClient(timeout=30) as client:
        while True:
            params = {"pageToken": page_token} if page_token else {}
            try:
                r = await client.get(api, headers=headers, params=params)
                r.raise_for_status()
            except httpx2.HTTPError as e:
                _out(f"  [warn] models list: {_describe(e)}")
                return out
            body = from_plain(r.json(), dict[str, object])
            for raw_model in from_plain(body.get("models"), list[object], default=[]):
                model = from_plain(raw_model, dict[str, object])
                short = from_plain(model.get("name"), str, default="").removeprefix(
                    "models/",
                )
                inp = from_plain(model.get("inputTokenLimit"), int, default=0)
                outp = from_plain(model.get("outputTokenLimit"), int, default=0)
                if inp and outp:
                    out[short] = LiveLimits(
                        max_request_tokens=inp,
                        max_response_tokens=outp,
                    )
            page_token = from_plain(body.get("nextPageToken"), str, default="")
            if not page_token:
                return out


# Source: https://developers.openai.com/api/docs/models/<model>.md
# The markdown rendering states "N context window", "Maximum input tokens: N"
# where the model has one, and "N max output tokens". The HTML page omits the
# input cap, and the catalog stores that cap, not the window.


async def fetch_openai(model_ids: list[str]) -> dict[str, LiveLimits]:
    """Scrape model limits from OpenAI documentation pages.

    Args:
      model_ids: Model identifiers to look up.

    Returns:
      limits: Map of model ID to its token limits.

    """
    out: dict[str, LiveLimits] = {}
    doc = "https://developers.openai.com/api/docs/models"
    async with httpx2.AsyncClient(timeout=30, follow_redirects=True) as client:
        for mid in model_ids:
            url = f"{doc}/{mid}.md"
            try:
                r = await client.get(url)
                r.raise_for_status()
            except httpx2.HTTPError as e:
                _out(f"  [warn] {mid}: {_describe(e)}")
                continue
            limits = _parse_openai_page(r.text)
            if limits:
                out[mid] = limits
            else:
                _out(f"  [warn] {mid}: could not parse limits from {url}")
    return out


# Source: GET /v1/models/{model_id}
# Returns max_tokens (max output) and max_input_tokens.


async def fetch_anthropic(
    api_key: str,
    model_ids: list[str],
) -> dict[str, LiveLimits]:
    """Fetch model limits from the Anthropic API.

    Args:
      api_key: Anthropic API key.
      model_ids: Model identifiers to query.

    Returns:
      limits: Map of model ID to its token limits.

    """
    out: dict[str, LiveLimits] = {}
    api = "https://api.anthropic.com/v1/models"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    async with httpx2.AsyncClient(timeout=30) as client:
        for mid in model_ids:
            try:
                r = await client.get(f"{api}/{mid}", headers=headers)
                r.raise_for_status()
            except httpx2.HTTPStatusError as e:
                if e.response.status_code == 404:
                    _out(f"  [warn] {mid}: not found in API")
                else:
                    _out(f"  [warn] {mid}: {_describe(e)}")
                continue
            except httpx2.HTTPError as e:
                _out(f"  [warn] {mid}: {_describe(e)}")
                continue
            data = from_plain(r.json(), dict[str, object])
            max_input = from_plain(data.get("max_input_tokens"), int, default=0)
            max_output = from_plain(data.get("max_tokens"), int, default=0)
            if max_input and max_output:
                out[mid] = LiveLimits(
                    max_request_tokens=max_input,
                    max_response_tokens=max_output,
                )
            else:
                _out(f"  [warn] {mid}: missing limits in API response")
    return out


def compare(
    provider_name: str,
    known: Mapping[str, ModelCapability],
    live: dict[str, LiveLimits],
) -> int:
    """Compare catalog entries against live API limits.

    A catalog row the live source did not report is listed as unverified;
    if no row at all was verified, that is itself an error.

    Args:
      provider_name: Display name for log output.
      known: Resolved model catalog, keyed by vendor wire id.
      live: LiveLimits fetched from the live API, keyed by vendor wire id.

    Returns:
      error_count: Number of problems found.

    """
    errors = 0
    verified = 0
    all_ids = sorted(set(known) | set(live))
    for mid in all_ids:
        k = known.get(mid)
        lv = live.get(mid)
        if k is None:
            _out(f"  {provider_name}.{mid}: in API but not in the catalog")
            if lv:
                _out(
                    f"    API: req={lv.max_request_tokens:,}"
                    f" resp={lv.max_response_tokens:,}",
                )
            errors += 1
            continue
        if lv is None:
            _out(f"  {provider_name}.{mid}: not verified")
            continue
        verified += 1
        # The untagged context is the one the vendor API reports; ``+1m``
        # is an opt-in the model list does not enumerate.
        base = k.context[""]
        k_req = base.max_request_tokens
        k_resp = base.max_response_tokens
        if k_req != lv.max_request_tokens:
            _out(
                f"  {provider_name}.{mid}: max_request_tokens"
                f" code={k_req:,} api={lv.max_request_tokens:,}",
            )
            errors += 1
        if k_resp != lv.max_response_tokens:
            _out(
                f"  {provider_name}.{mid}: max_response_tokens"
                f" code={k_resp:,} api={lv.max_response_tokens:,}",
            )
            errors += 1
    if known and not verified:
        errors += 1
    if errors or verified < len(known):
        _out(f"  {provider_name}: {verified}/{len(known)} models verified")
    else:
        _out(f"  {provider_name}: all {len(known)} models OK")
    return errors


def audit_catalogs() -> int:
    """Check every provider catalog for self-consistency, offline.

    Catches the drift a live query cannot: an empty price catalog raises at
    first bill rather than at import; a zero window silently disables the
    compaction trigger.

    Returns:
      error_count: Number of problems found.

    """
    errors = 0
    for name in sorted(providers.PROVIDER_NAMES):
        cls = getattr(providers, name, None)
        if not isinstance(cls, ModelResolver):
            continue
        rows = {
            model_id: cls.catalog.resolve(model_id)[0]
            for model_id in cls.catalog.models
        }
        for mid, cap in rows.items():
            if not cap.prices:
                _out(f"  {name}.{mid}: no price rows -- spend() would raise")
                errors += 1
            for tag, lim in cap.context.items():
                where = f"{name}.{mid}{tag}"
                if lim.max_request_tokens <= 0:
                    _out(f"  {where}: max_request_tokens is 0")
                    errors += 1
                if lim.max_response_tokens <= 0:
                    _out(f"  {where}: max_response_tokens is 0")
                    errors += 1
    if not errors:
        _out("  catalogs: all rows OK")
    return errors


def main() -> int:
    """Run the program; return the process exit code."""
    return asyncio.run(_run())


def _out(msg: str) -> None:
    sys.stdout.write(msg + "\n")


def _describe(e: httpx2.HTTPError) -> str:
    """Name a request failure without echoing its URL."""
    if isinstance(e, httpx2.HTTPStatusError):
        return f"HTTP {e.response.status_code}"
    return type(e).__name__


# The input cap is "Maximum input tokens" where the page states one, else the context
# window.
def _parse_openai_page(page: str) -> LiveLimits | None:
    """Extract the input cap and max output tokens from an OpenAI doc page."""
    cleaned = re.sub(r"<!--.*?-->", " ", page, flags=re.DOTALL)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    request = re.search(r"Maximum\s+input\s+tokens:?\s+([\d,]+)", cleaned) or (
        re.search(r"([\d,]+)\s+context\s+window", cleaned)
    )
    out = re.search(r"([\d,]+)\s+max\s+output\s+tokens", cleaned)
    if request and out:
        return LiveLimits(
            max_request_tokens=_num(request.group(1)),
            max_response_tokens=_num(out.group(1)),
        )
    return None


def _num(s: str) -> int:
    """Parse a comma- or underscore-grouped integer literal."""
    return int(s.replace(",", "").replace("_", ""))


async def _run() -> int:
    """Verify all providers' model catalogs against live APIs."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--provider",
        choices=["google", "openai", "anthropic", "all"],
        default="all",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Only run the offline catalog audit; skip every network query.",
    )
    args = parser.parse_args()
    target = args.provider
    _out("Catalog audit (offline):")
    total_errors = audit_catalogs()
    if args.offline:
        if total_errors:
            _out(f"\n{total_errors} problem(s) found.")
        return 1 if total_errors else 0

    if target in ("all", "google"):
        _out("Google (API query):")
        key = os.environ.get("GOOGLE_API_KEY")
        if not key:
            _out("  [error] GOOGLE_API_KEY not set")
            total_errors += 1
        else:
            live = await fetch_google(key)
            total_errors += compare("Google", _canonical_models(Google.catalog), live)

    if target in ("all", "openai"):
        _out("OpenAI (doc scrape):")
        known = _canonical_models(OpenAI.catalog)
        live = await fetch_openai(list(known))
        total_errors += compare("OpenAI", known, live)

    if target in ("all", "anthropic"):
        _out("Anthropic (API query):")
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            _out("  [error] ANTHROPIC_API_KEY not set")
            total_errors += 1
        else:
            known = _canonical_models(Anthropic.catalog)
            live = await fetch_anthropic(key, list(known))
            total_errors += compare("Anthropic", known, live)

    if total_errors:
        _out(f"\n{total_errors} problem(s) found.")
    else:
        _out("\nAll limits verified.")
    return 1 if total_errors else 0


def _canonical_models(catalog: ModelCatalog) -> dict[str, ModelCapability]:
    """Return one resolved row per model, keyed by the id its vendor serves it as."""
    rows = (catalog.resolve(model_id)[0] for model_id in catalog.models)
    return {row.wire_model_id or row.model_id: row for row in rows}


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
