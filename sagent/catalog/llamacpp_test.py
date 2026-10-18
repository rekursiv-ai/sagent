"""llama.cpp catalog rows: free, local, and windowed per served model."""

from __future__ import annotations

import pytest

from sagent.catalog import llamacpp
from sagent.types.cost import PriceKey


@pytest.mark.parametrize(
    ("model_id", "wire_id", "window"),
    [
        ("qwen-27b-12gb-3.6", "qwen3.6-27b-12gb", (16_384, 1_024)),
        ("qwen-27b-mtp-64k-3.6", "qwen3.6-27b-mtp-64k", (65_536, 4_096)),
        ("local-0.0", "local", (32_768, 4_096)),
    ],
)
def test_rows_carry_their_wire_id_and_window(
    model_id: str,
    wire_id: str,
    window: tuple[int, int],
) -> None:
    row = llamacpp.models()[model_id]
    limits = row.context[""]
    assert row.wire_model_id == wire_id
    assert (limits.max_request_tokens, limits.max_response_tokens) == window
    assert tuple(row.context) == ("",)


def test_every_row_bills_nothing() -> None:
    rows = llamacpp.models()
    for model_id in ("qwen-27b-12gb-3.6", "qwen-27b-mtp-64k-3.6", "local-0.0"):
        price = rows[model_id].prices[PriceKey("auto")]
        assert (
            price.request,
            price.response,
            price.cache_write,
            price.cache_write_1h,
            price.cache_read,
        ) == (0.0, 0.0, 0.0, 0.0, 0.0)
        assert rows[model_id].service_tier == rows[model_id].prices.service_tiers


def test_roles_name_the_reference_row() -> None:
    rows = llamacpp.models()
    assert rows["default"].model_id == "qwen-27b-12gb-3.6"
    assert rows["utility"].model_id == "qwen-27b-12gb-3.6"


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
