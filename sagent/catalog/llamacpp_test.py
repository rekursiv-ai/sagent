"""llama.cpp rows: free, local, and windowed per served model."""

import pytest

from sagent.catalog.loader import models
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
    row = models("llamacpp")[model_id]
    limits = row.context[""]
    assert row.wire_model_id == wire_id
    assert (limits.max_request_tokens, limits.max_response_tokens) == window
    assert row.context.keys() == {""}


def test_every_row_bills_nothing() -> None:
    for row in models("llamacpp").values():
        assert row.prices.keys() == {PriceKey("auto")}
        assert row.prices[PriceKey("auto")].total == 0.0


def test_roles_name_the_reference_row() -> None:
    table = models("llamacpp")
    assert {table[role].model_id for role in table.roles} == {"qwen-27b-12gb-3.6"}


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
