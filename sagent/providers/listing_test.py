"""Tests for ``Listing``: what the catalog knows of a provider beyond building it."""

from __future__ import annotations

from functools import partial
from pathlib import Path

from sagent.providers.anthropic.api import Anthropic
from sagent.providers.anthropic.cli import AnthropicCLI
from sagent.providers.listing import Listing


def test_a_provider_listed_plainly_signs_in_as_no_one() -> None:
    listed = Listing.of(Anthropic)

    assert (listed.section, listed.factory, listed.former_names) == (
        "anthropic",
        Anthropic,
        (),
    )
    assert listed.default_login is None
    assert listed.signed_in is None
    assert listed.preferred is False


def test_a_provider_that_signs_in_is_built_with_the_file_it_is_given() -> None:
    def default() -> Path:
        return Path("/logins/default.json")

    listed = Listing.signing_in(
        AnthropicCLI,
        default=default,
        former_names=("Old",),
        preferred=True,
    )
    assert listed.preferred is True
    assert listed.signed_in is not None
    built = listed.signed_in(Path("/logins/work.json"))

    assert (listed.section, listed.factory, listed.former_names) == (
        "anthropic-cli",
        AnthropicCLI,
        ("Old",),
    )
    assert listed.default_login is default
    assert isinstance(built, partial)
    assert (built.func, built.keywords) == (
        AnthropicCLI,
        {"credentials": Path("/logins/work.json")},
    )


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
