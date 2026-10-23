"""MiniMax transport restrictions."""

from dataclasses import replace

from sagent.catalog import minimax
from sagent.catalog.openai import compatible


def test_api_adds_priority_to_the_compatible_transport() -> None:
    transport = compatible()
    assert minimax.api() == replace(
        transport,
        service_tier=frozenset({"auto", "priority"}),
    )


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
