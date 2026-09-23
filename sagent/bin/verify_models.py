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

Sources:
- Google: GET /v1beta/models (returns inputTokenLimit, outputTokenLimit)
- OpenAI: scrapes https://developers.openai.com/api/docs/models/<id>
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
from sagent.lib.custom_json import DictCodec, IntCodec, ListCodec, StrCodec
from sagent.providers.anthropic.api import Anthropic
from sagent.providers.google.api import Google
from sagent.providers.openai.api import OpenAI
from sagent.types.providers import ModelCatalog, ModelResolver


if TYPE_CHECKING:
    from collections.abc import Mapping

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
      limits: Map of model ID to its token limits.

    """
    api = "https://generativelanguage.googleapis.com/v1beta/models"
    async with httpx2.AsyncClient(timeout=30) as client:
        r = await client.get(f"{api}?key={api_key}")
        r.raise_for_status()
        out: dict[str, LiveLimits] = {}
        body = DictCodec.coerce(r.json())
        for raw_model in ListCodec.coerce(body.get("models")):
            model = DictCodec.coerce(raw_model)
            short = StrCodec.coerce(model.get("name"), default="").removeprefix(
                "models/",
            )
            inp = IntCodec.coerce(model.get("inputTokenLimit"))
            outp = IntCodec.coerce(model.get("outputTokenLimit"))
            if inp and outp:
                out[short] = LiveLimits(
                    max_request_tokens=inp,
                    max_response_tokens=outp,
                )
        return out


# Source: https://developers.openai.com/api/docs/models/<model>
# Each page SSR-renders "N context window" and "N max output tokens".


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
            url = f"{doc}/{mid}"
            try:
                r = await client.get(url)
                r.raise_for_status()
                limits = _parse_openai_page(r.text)
                if limits:
                    out[mid] = limits
                else:
                    _out(f"  [warn] {mid}: could not parse limits from {url}")
            except httpx2.HTTPStatusError as e:
                _out(f"  [warn] {mid}: HTTP {e.response.status_code}")
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
                data = DictCodec.coerce(r.json())
                max_input = IntCodec.coerce(data.get("max_input_tokens"))
                max_output = IntCodec.coerce(data.get("max_tokens"))
                if max_input and max_output:
                    out[mid] = LiveLimits(
                        max_request_tokens=max_input,
                        max_response_tokens=max_output,
                    )
                else:
                    _out(f"  [warn] {mid}: missing limits in API response")
            except httpx2.HTTPStatusError as e:
                if e.response.status_code == 404:
                    _out(f"  [warn] {mid}: not found in API")
                else:
                    _out(f"  [warn] {mid}: HTTP {e.response.status_code}")
    return out


def compare(
    provider_name: str,
    known: Mapping[str, ModelCapability],
    live: dict[str, LiveLimits],
) -> int:
    """Compare catalog entries against live API limits.

    Args:
      provider_name: Display name for log output.
      known: Resolved model catalog.
      live: LiveLimits fetched from the live API.

    Returns:
      error_count: Number of mismatches found.

    """
    errors = 0
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
            continue
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
    if not errors:
        _out(f"  {provider_name}: all {len(known)} models OK")
    return errors


def audit_catalogs() -> int:
    """Check every provider catalog for self-consistency, offline.

    Catches the drift a live query cannot: a row whose ``model_id`` does
    not match its key ships the wrong id on the wire; an empty price
    catalog raises at first bill rather than at import; a zero window
    silently disables the compaction trigger.

    Returns:
      error_count: Number of problems found.

    """
    errors = 0
    for name in sorted(providers.PROVIDER_NAMES):
        cls = getattr(providers, name, None)
        if not isinstance(cls, ModelResolver):
            continue
        rows = {
            model_id: cls.catalog.resolve(model_id)[0] for model_id in cls.catalog.rows
        }
        for mid, cap in rows.items():
            if mid in {"default", "utility"}:
                if cap.model_id not in rows:
                    _out(
                        f"  {name}.{mid}: target {cap.model_id!r} is not a catalog key",
                    )
                    errors += 1
            elif cap.model_id != mid:
                _out(f"  {name}.{mid}: model_id is {cap.model_id!r}, not the key")
                errors += 1
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


def _parse_openai_page(html: str) -> LiveLimits | None:
    """Extract context window and max output tokens from an OpenAI doc page."""
    cleaned = re.sub(r"<!--.*?-->", " ", html, flags=re.DOTALL)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    ctx = re.search(r"([\d,]+)\s+context\s+window", cleaned)
    out = re.search(r"([\d,]+)\s+max\s+output\s+tokens", cleaned)
    if ctx and out:
        return LiveLimits(
            max_request_tokens=_num(ctx.group(1)),
            max_response_tokens=_num(out.group(1)),
        )
    return None


def _num(s: str) -> int:
    """Parse a comma- or underscore-grouped integer literal."""
    return int(s.replace(",", "").replace("_", ""))


async def _run() -> int:
    """Verify all providers' model catalogs against live APIs."""
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n", 2)[2],
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
        key = os.environ.get("GOOGLE_API_KEY", "")
        if not key:
            _out("  [skip] GOOGLE_API_KEY not set")
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
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            _out("  [skip] ANTHROPIC_API_KEY not set")
        else:
            known = _canonical_models(Anthropic.catalog)
            live = await fetch_anthropic(key, list(known))
            total_errors += compare("Anthropic", known, live)

    if total_errors:
        _out(f"\n{total_errors} mismatch(es) found.")
    else:
        _out("\nAll limits verified.")
    return 1 if total_errors else 0


def _canonical_models(catalog: ModelCatalog) -> dict[str, ModelCapability]:
    """Return one resolved row per canonical model ID."""
    return {
        capability.model_id: capability
        for model_id in catalog.rows
        if (capability := catalog.resolve(model_id)[0]).model_id == model_id
    }


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
