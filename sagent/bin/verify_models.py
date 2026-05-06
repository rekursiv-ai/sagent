#!/bin/sh
# ruff: noqa: EXE003, D300  -- Polyglot: #!/bin/sh + triple-single-quotes are intentional.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen python3 "$0" "$@"

Verify KNOWN_MODELS limits against provider APIs and docs.

Checks that every provider's KNOWN_MODELS entries have correct
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

import argparse
import asyncio
import os
import re
import sys

import httpx

from sagent.providers.anthropic import Anthropic
from sagent.providers.google import Google
from sagent.providers.openai import OpenAI


def _out(msg: str) -> None:
    sys.stdout.write(msg + "\n")


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelLimits:
    """Token limits for a single model."""

    max_request_tokens: int
    max_response_tokens: int


# -- Google (API) -----------------------------------------------------
# Source: https://ai.google.dev/api/models#method:-models.list
# Returns inputTokenLimit and outputTokenLimit per model.

_GOOGLE_API = "https://generativelanguage.googleapis.com/v1beta/models"


async def fetch_google(api_key: str) -> dict[str, ModelLimits]:
    """Fetch model limits from the Google Generative Language API.

    Args:
      api_key: Google API key.

    Returns:
      limits: Map of model ID to its token limits.

    """
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{_GOOGLE_API}?key={api_key}")
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


# -- OpenAI (scrape per-model doc page) --------------------------------
# Source: https://developers.openai.com/api/docs/models/<model>
# Each page SSR-renders "N context window" and "N max output tokens".

_OPENAI_DOC = "https://developers.openai.com/api/docs/models"


async def fetch_openai(model_ids: list[str]) -> dict[str, ModelLimits]:
    """Scrape model limits from OpenAI documentation pages.

    Args:
      model_ids: Model identifiers to look up.

    Returns:
      limits: Map of model ID to its token limits.

    """
    out: dict[str, ModelLimits] = {}
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        for mid in model_ids:
            url = f"{_OPENAI_DOC}/{mid}"
            try:
                r = await client.get(url)
                r.raise_for_status()
                limits = _parse_openai_page(r.text)
                if limits:
                    out[mid] = limits
                else:
                    _out(f"  [warn] {mid}: could not parse limits from {url}")
            except httpx.HTTPStatusError as e:
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


# -- Anthropic (API) ---------------------------------------------------
# Source: GET /v1/models/{model_id}
# Returns max_tokens (max output) and max_input_tokens.

_ANTHROPIC_API = "https://api.anthropic.com/v1/models"


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
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        for mid in model_ids:
            try:
                r = await client.get(f"{_ANTHROPIC_API}/{mid}", headers=headers)
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
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    _out(f"  [warn] {mid}: not found in API")
                else:
                    _out(f"  [warn] {mid}: HTTP {e.response.status_code}")
    return out


# -- Comparison --------------------------------------------------------


def _num(s: str) -> int:
    return int(s.replace(",", "").replace("_", ""))


def compare(
    provider_name: str,
    known: Mapping[str, object],
    live: dict[str, ModelLimits],
) -> int:
    """Compare KNOWN_MODELS entries against live API limits.

    Args:
      provider_name: Display name for log output.
      known: KNOWN_MODELS mapping from the provider class.
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
            _out(f"  {provider_name}.{mid}: in API but not in KNOWN_MODELS")
            if lv:
                _out(
                    f"    API: req={lv.max_request_tokens:,}"
                    f" resp={lv.max_response_tokens:,}"
                )
            errors += 1
            continue
        if lv is None:
            continue
        k_req = getattr(k, "max_request_tokens", 0)
        k_resp = getattr(k, "max_response_tokens", 0)
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


# -- Main --------------------------------------------------------------


async def main() -> int:
    """Verify all providers' KNOWN_MODELS against live APIs.

    Returns:
      exit_code: 0 if all limits match, 1 otherwise.

    """
    parser = argparse.ArgumentParser(
        description="Verify KNOWN_MODELS limits against provider APIs.",
    )
    parser.add_argument(
        "--provider",
        choices=["google", "openai", "anthropic", "all"],
        default="all",
    )
    args = parser.parse_args()
    target = args.provider
    total_errors = 0

    if target in ("all", "google"):
        _out("Google (API query):")
        key = os.environ.get("GOOGLE_API_KEY", "")
        if not key:
            _out("  [skip] GOOGLE_API_KEY not set")
        else:
            live = await fetch_google(key)
            total_errors += compare("Google", Google.KNOWN_MODELS, live)

    if target in ("all", "openai"):
        _out("OpenAI (doc scrape):")
        live = await fetch_openai(list(OpenAI.KNOWN_MODELS))
        total_errors += compare("OpenAI", OpenAI.KNOWN_MODELS, live)

    if target in ("all", "anthropic"):
        _out("Anthropic (API query):")
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            _out("  [skip] ANTHROPIC_API_KEY not set")
        else:
            live = await fetch_anthropic(key, list(Anthropic.KNOWN_MODELS))
            total_errors += compare("Anthropic", Anthropic.KNOWN_MODELS, live)

    if total_errors:
        _out(f"\n{total_errors} mismatch(es) found.")
    else:
        _out("\nAll limits verified.")
    return 1 if total_errors else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
