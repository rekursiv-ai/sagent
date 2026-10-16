"""Configure an A2Agent chat-completions model without guessing its metadata.

See docs/providers.md for required model limits and account-specific rates.
This example uses the existing OpenAI-compatible transport unchanged.
"""

from __future__ import annotations

from typing import ClassVar

import argparse
import asyncio
import math
import sys

from sagent.catalog.openai import compatible
from sagent.providers.openai.compat import OpenAICompat
from sagent.types.capability import ModelCapability, ModelLimits
from sagent.types.cost import PriceCatalog, PriceKey, TokenPrice
from sagent.types.model import ModelRequest
from sagent.types.providers import ModelCatalog
from sagent.types.runtime import UserMessage


class A2Agent(OpenAICompat):
    """A2Agent endpoint; configure model metadata with ``create_provider``."""

    ENV_VAR: ClassVar[str] = "A2AGENT_API_KEY"
    BASE_URL: ClassVar[str] = "https://api.a2agent.me/v1"


def create_provider(
    model_id: str,
    *,
    limits: ModelLimits,
    prices: TokenPrice,
    base_url: str = A2Agent.BASE_URL,
) -> A2Agent:
    """Build a single-model provider using explicitly supplied limits and prices.

    Args:
      model_id: Exact model ID available to the API key.
      limits: Confirmed input and output token limits for this route.
      prices: Rates for the key's group, including every cache meter.
      base_url: Full API base URL, including any version/path prefix.

    Returns:
      provider: Configured provider authenticated via ``A2AGENT_API_KEY``.

    Raises:
      ValueError: Metadata is empty, non-positive, or not a finite price.
      RuntimeError: The API key is missing.

    """
    if not model_id.strip() or model_id != model_id.strip():
        raise ValueError("Supply an exact, non-empty model ID.")
    if limits.max_request_tokens <= 0 or limits.max_response_tokens <= 0:
        raise ValueError("Supply positive input and output token limits.")
    rates = (
        prices.request,
        prices.response,
        prices.cache_read,
        prices.cache_write,
        prices.cache_write_1h,
    )
    if prices.tokens_per_unit <= 0 or any(
        not math.isfinite(rate) or rate < 0 for rate in rates
    ):
        raise ValueError("Supply finite non-negative rates and a positive token unit.")
    row = ModelCapability(
        model_id=model_id,
        context={"": limits},
        prices=PriceCatalog({PriceKey("auto"): prices}),
    )
    provider = A2Agent.from_env(base_url=base_url)
    provider.catalog = ModelCatalog(
        rows={model_id: row, "default": row, "utility": row},
        transport=compatible(),
    )
    return provider


async def main() -> None:
    """Send one prompt using explicit metadata and an environment-only API key."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-limit", type=int, required=True)
    parser.add_argument("--output-limit", type=int, required=True)
    parser.add_argument("--input-price", type=float, required=True)
    parser.add_argument("--output-price", type=float, required=True)
    parser.add_argument("--cache-read-price", type=float, required=True)
    parser.add_argument("--cache-write-price", type=float, required=True)
    parser.add_argument("--cache-write-1h-price", type=float, required=True)
    parser.add_argument("--base-url", default=A2Agent.BASE_URL)
    parser.add_argument("--prompt", default="Reply with OK.")
    args = parser.parse_args()
    try:
        provider = create_provider(
            args.model,
            limits=ModelLimits(
                max_request_tokens=args.input_limit,
                max_response_tokens=args.output_limit,
            ),
            prices=TokenPrice(
                request=args.input_price,
                response=args.output_price,
                cache_read=args.cache_read_price,
                cache_write=args.cache_write_price,
                cache_write_1h=args.cache_write_1h_price,
            ),
            base_url=args.base_url,
        )
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    model = provider.model()
    try:
        response = await model.buffer(
            ModelRequest(
                messages=[UserMessage(text=args.prompt)],
                max_response_tokens=min(256, args.output_limit),
            ),
        )
        sys.stdout.write(f"{response.message.text}\n")
    finally:
        await model.close()


if __name__ == "__main__":
    asyncio.run(main())
