"""Defaults, immutability, and names of ``ContextSize``."""

from __future__ import annotations

import dataclasses

import pytest

from sagent.model.capability import literal_members
from sagent.model.context import ContextSize, ContextSizeName


def test_defaults() -> None:
    size = ContextSize()
    assert (size.context_name, size.tokens, size.display_name) == (
        "smallcontext",
        0,
        "0",
    )


def test_fields_are_keyword_only_and_frozen() -> None:
    size = ContextSize(context_name="fullcontext", tokens=1_000_000, display_name="1M")
    assert size.context_name == "fullcontext"
    assert size.tokens == 1_000_000
    assert size.display_name == "1M"
    assert all(field.kw_only for field in dataclasses.fields(size))
    with pytest.raises(dataclasses.FrozenInstanceError):
        dataclasses.replace(size, tokens=2).__setattr__("tokens", 1)


def test_the_size_names_say_they_size_the_context() -> None:
    assert literal_members(ContextSizeName) == (
        "smallcontext",
        "midcontext",
        "fullcontext",
    )


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
