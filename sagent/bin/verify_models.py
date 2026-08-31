#!/bin/sh
# ruff: noqa: EXE003, D300 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Verify CAPABILITIES limits against provider APIs and docs.

Checks that every provider's CAPABILITIES entries have correct
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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import argparse
import asyncio
import os
import re
import sys

import httpx2

from sagent import providers
from sagent.providers.anthropic.api import Anthropic
from sagent.providers.google.api import Google
from sagent.providers.openai.api import OpenAI
from sagent.types.model import Limits, ModelCapability


def _out(msg: str) -> None:
    sys.stdout.write(msg + "\n")


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelLimits:
    """Token limits for a single model."""

    max_request_tokens: int
    max_response_tokens: int


# Source: https://ai.google.dev/api/models#method:-models.list
# Returns inputTokenLimit and outputTokenLimit per model.


async def fetch_google(api_key: str) -> dict[str, ModelLimits]:
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
        out: dict[str, ModelLimits] = {}
        for m in r.json().get("models", []):
            short = m.get("name", "").removeprefix("models/")
            inp = m.get("inputTokenLimit", 0)
            outp = m.get("outputTokenLimit", 0)
            if inp and outp:
                out[short] = ModelLimits(
                    max_request_tokens=inp,
                    max_response_tokens=outp,
                )
        return out


# Source: https://developers.openai.com/api/docs/models/<model>
# Each page SSR-renders "N context window" and "N max output tokens".


async def fetch_openai(model_ids: list[str]) -> dict[str, ModelLimits]:
    """Scrape model limits from OpenAI documentation pages.

    Args:
      model_ids: Model identifiers to look up.

    Returns:
      limits: Map of model ID to its token limits.

    """
    out: dict[str, ModelLimits] = {}
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


def _parse_openai_page(html: str) -> ModelLimits | None:
    """Extract context window and max output tokens from an OpenAI doc page."""
    cleaned = re.sub(r"<!--.*?-->", " ", html)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    ctx = re.search(r"([\d,]+)\s+context\s+window", cleaned)
    out = re.search(r"([\d,]+)\s+max\s+output\s+tokens", cleaned)
    if ctx and out:
        return ModelLimits(
            max_request_tokens=_num(ctx.group(1)),
            max_response_tokens=_num(out.group(1)),
        )
    return None


# Source: GET /v1/models/{model_id}
# Returns max_tokens (max output) and max_input_tokens.


async def fetch_anthropic(
    api_key: str,
    model_ids: list[str],
) -> dict[str, ModelLimits]:
    """Fetch model limits from the Anthropic API.

    Args:
      api_key: Anthropic API key.
      model_ids: Model identifiers to query.

    Returns:
      limits: Map of model ID to its token limits.

    """
    out: dict[str, ModelLimits] = {}
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
                data = r.json()
                max_input = data.get("max_input_tokens", 0)
                max_output = data.get("max_tokens", 0)
                if max_input and max_output:
                    out[mid] = ModelLimits(
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


def _num(s: str) -> int:
    """Parse a comma- or underscore-grouped integer literal."""
    return int(s.replace(",", "").replace("_", ""))


def compare(
    provider_name: str,
    known: Mapping[str, ModelCapability],
    live: dict[str, ModelLimits],
) -> int:
    """Compare CAPABILITIES entries against live API limits.

    Args:
      provider_name: Display name for log output.
      known: CAPABILITIES mapping from the provider class.
      live: Limits fetched from the live API.

    Returns:
      error_count: Number of mismatches found.

    """
    errors = 0
    all_ids = sorted(set(known) | set(live))
    for mid in all_ids:
        k = known.get(mid)
        lv = live.get(mid)
        if k is None:
            _out(f"  {provider_name}.{mid}: in API but not in CAPABILITIES")
            if lv:
                _out(
                    f"    API: req={lv.max_request_tokens:,}"
                    f" resp={lv.max_response_tokens:,}"
                )
            errors += 1
            continue
        if lv is None:
            continue
        # The untagged context is the one the vendor API reports; ``+1m``
        # is an opt-in the model list does not enumerate.
        limits = k.context_limits
        base = limits if isinstance(limits, Limits) else limits[""]
        k_req = base.max_request_tokens
        k_resp = base.max_response_tokens
        if k_req != lv.max_request_tokens:
            _out(
                f"  {provider_name}.{mid}: max_request_tokens"
                f" code={k_req:,} api={lv.max_request_tokens:,}"
            )
            errors += 1
        if k_resp != lv.max_response_tokens:
            _out(
                f"  {provider_name}.{mid}: max_response_tokens"
                f" code={k_resp:,} api={lv.max_response_tokens:,}"
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
        catalog = getattr(cls, "CAPABILITIES", None)
        if not isinstance(catalog, Mapping):
            continue
        rows = cast(Mapping[str, ModelCapability], catalog)
        for mid, cap in rows.items():
            if cap.model_id != mid:
                _out(f"  {name}.{mid}: model_id is {cap.model_id!r}, not the key")
                errors += 1
            if not cap.prices:
                _out(f"  {name}.{mid}: no price rows -- spend() would raise")
                errors += 1
            limits = cap.context_limits
            per_tag = {"": limits} if isinstance(limits, Limits) else limits
            for tag, lim in per_tag.items():
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


async def _run() -> int:
    """Verify all providers' CAPABILITIES against live APIs.

    Returns:
      exit_code: 0 if all limits match, 1 otherwise.

    """
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
            total_errors += compare("Google", Google.CAPABILITIES, live)

    if target in ("all", "openai"):
        _out("OpenAI (doc scrape):")
        live = await fetch_openai(list(OpenAI.CAPABILITIES))
        total_errors += compare("OpenAI", OpenAI.CAPABILITIES, live)

    if target in ("all", "anthropic"):
        _out("Anthropic (API query):")
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            _out("  [skip] ANTHROPIC_API_KEY not set")
        else:
            live = await fetch_anthropic(key, list(Anthropic.CAPABILITIES))
            total_errors += compare("Anthropic", Anthropic.CAPABILITIES, live)

    if total_errors:
        _out(f"\n{total_errors} mismatch(es) found.")
    else:
        _out("\nAll limits verified.")
    return 1 if total_errors else 0


def main() -> int:
    """The main function. Return the process exit code."""
    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
