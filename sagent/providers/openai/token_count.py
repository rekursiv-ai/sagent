"""Local token-estimation functions shared by OpenAI wire transports.

Which tokenizer and which image formula a model uses are catalog facts
(:mod:`sagent.catalog.openai`); this module only runs them. An id the
OpenAI catalog does not carry -- an OpenAI-compatible vendor's -- falls back
to the catalog character ratio and the catalog's 512px-tile estimate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sagent.catalog import openai
from sagent.catalog.table import base_model_id


if TYPE_CHECKING:
    import tiktoken

    from sagent.lib.image import get_dimensions
else:
    from wrapt import lazy_import

    tiktoken = lazy_import("tiktoken")
    get_dimensions = lazy_import("sagent.lib.image", "get_dimensions")


def approx_text_tokens(
    text: str,
    *,
    model_id: str,
    approx_chars_per_token: float = 4.0,
) -> int:
    """Count text locally, falling back to the catalog ratio.

    Args:
      text: Text to count, including literal special-token spellings.
      model_id: Model id selecting the tokenizer.
      approx_chars_per_token: Fallback divisor when no tokenizer is known.

    Returns:
      tokens: Local text token estimate.

    """
    encoding = openai.tokenizer(base_model_id(model_id))
    if encoding is None:
        return int(len(text) / approx_chars_per_token)
    return len(tiktoken.get_encoding(encoding).encode_ordinary(text))


def approx_image_tokens(data: bytes, *, model_id: str, max_edge: int) -> int:
    """Estimate image tokens using the model's published formula.

    Args:
      data: Encoded image bytes.
      model_id: Model id selecting patch or tile accounting.
      max_edge: Serialization's image-edge ceiling; zero preserves dimensions.

    Returns:
      tokens: Local image token estimate, or zero for unknown dimensions.

    References:
      https://developers.openai.com/api/docs/guides/images-vision

    """
    dims = get_dimensions(data)
    if dims is None:
        return 0
    width, height = _resized_dims(dims, max_edge)
    return openai.image_tokens(base_model_id(model_id), width, height)


def _resized_dims(dims: tuple[int, int], max_edge: int) -> tuple[int, int]:
    """Count the dimensions that serialization will send, not source tiles."""
    width, height = dims
    longest = max(width, height)
    if max_edge <= 0 or longest <= max_edge:
        return width, height
    scale = max_edge / longest
    return max(1, int(width * scale)), max(1, int(height * scale))
