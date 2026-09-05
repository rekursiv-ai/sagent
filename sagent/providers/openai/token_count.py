"""Local token-estimation functions shared by OpenAI wire transports."""

from __future__ import annotations

from typing import TYPE_CHECKING

import math

from sagent.types.model import base_model_id


if TYPE_CHECKING:
    import tiktoken

    from sagent.lib.image import get_dimensions
else:
    from wrapt import lazy_import

    tiktoken = lazy_import("tiktoken")
    get_dimensions = lazy_import("sagent.lib.image", "get_dimensions")


def approx_text_tokens(text: str, *, model_id: str) -> int:
    """Count text locally, falling back to chars/4 for unknown tokenizers.

    Args:
      text: Text to count, including literal special-token spellings.
      model_id: Model id selecting the tokenizer.

    Returns:
      tokens: Local text token estimate.

    """
    encoding = _tiktoken_encoding(model_id)
    return (
        len(encoding.encode_ordinary(text)) if encoding is not None else len(text) // 4
    )


def approx_image_tokens(data: bytes, *, model_id: str, max_edge: int) -> int:
    """Estimate image tokens using the model family's published formula.

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
    if base_model_id(model_id) == "gpt-6-astra":
        # Measured on eight sizes, 1x1 through 2048x1024, on 2026-09-04.
        patches = math.ceil(dims[0] / 32) * math.ceil(dims[1] / 32)
        return math.floor(1.2 * patches) + 1
    if base_model_id(model_id).startswith("gpt-5.6"):
        # GPT-5.6 preserves source dimensions rather than resizing into tiles.
        return math.ceil(dims[0] / 32) * math.ceil(dims[1] / 32)
    width, height = _resized_dims(dims, max_edge)
    tiles = math.ceil(width / 512) * math.ceil(height / 512)
    return 85 + tiles * 170


def _tiktoken_encoding(model_id: str) -> tiktoken.Encoding | None:
    model_id = base_model_id(model_id)
    try:
        return tiktoken.encoding_for_model(model_id)
    except KeyError:
        # Tiktoken's registry predates dotted GPT-5 releases and GPT-6.
        if model_id.startswith(("gpt-5.", "gpt-6")):
            return tiktoken.get_encoding("o200k_base")
        return None


def _resized_dims(dims: tuple[int, int], max_edge: int) -> tuple[int, int]:
    """Count the dimensions that serialization will send, not source tiles."""
    width, height = dims
    longest = max(width, height)
    if max_edge <= 0 or longest <= max_edge:
        return width, height
    scale = max_edge / longest
    return max(1, int(width * scale)), max(1, int(height * scale))
