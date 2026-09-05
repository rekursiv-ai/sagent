"""Standalone token estimation and transport composition contracts."""

from io import BytesIO

from PIL import Image

import pytest
import tiktoken

from sagent.providers.lib.model_base import ModelDefaults
from sagent.providers.openai import token_count
from sagent.providers.openai.compat import OpenAICompatModel
from sagent.providers.openai.responses import _OpenAIResponsesModel


@pytest.mark.parametrize("model_class", [OpenAICompatModel, _OpenAIResponsesModel])
def test_transports_use_model_defaults_directly(model_class: type) -> None:
    assert model_class.__bases__ == (ModelDefaults,)


@pytest.mark.parametrize("model_id", ["gpt-4o", "gpt-5.6-sol", "gpt-6-astra"])
def test_text_estimation_needs_no_model_instance(model_id: str) -> None:
    text = "東京 function_call(arg=42) <|endoftext|>"
    expected = len(tiktoken.get_encoding("o200k_base").encode_ordinary(text))
    assert token_count.approx_text_tokens(text, model_id=model_id) == expected


def test_unknown_tokenizer_keeps_coarse_fallback() -> None:
    assert token_count.approx_text_tokens("123456789", model_id="unknown-vendor") == 2


@pytest.mark.parametrize(
    ("model_id", "max_edge", "expected"),
    [("gpt-5.6-sol", 0, 6), ("gpt-6-astra", 0, 8), ("unknown-vendor", 32, 255)],
)
def test_image_estimation_needs_no_model_instance(
    model_id: str, max_edge: int, expected: int
) -> None:
    image = BytesIO()
    Image.new("RGB", (33, 65)).save(image, format="PNG")
    assert (
        token_count.approx_image_tokens(
            image.getvalue(), model_id=model_id, max_edge=max_edge
        )
        == expected
    )


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
