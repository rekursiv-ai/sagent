"""Tests for persistent automatic web-fetch transport routing."""

from pathlib import Path

import pytest

from sagent.lib.web.fetch.transport_routing import (
    remember_zendriver_domain,
    zendriver_domains,
    zendriver_domains_path,
)


def test_default_path_is_next_to_transport_routing_module() -> None:
    assert zendriver_domains_path() == (Path(__file__).parent / "zendriver-domains.txt")


def test_absent_domain_list_is_empty(tmp_path: Path) -> None:
    assert zendriver_domains(path=tmp_path / "domains.txt") == frozenset()


def test_remembered_domains_are_normalized_sorted_and_deduplicated(
    tmp_path: Path,
) -> None:
    path = tmp_path / "domains.txt"
    remember_zendriver_domain("B.Example", path=path)
    remember_zendriver_domain("a.example", path=path)
    remember_zendriver_domain("b.example", path=path)

    assert path.read_text() == "a.example\nb.example\n"
    assert zendriver_domains(path=path) == frozenset({"a.example", "b.example"})


def test_newline_domain_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid Zendriver domain"):
        remember_zendriver_domain("safe.example\nother.example", path=tmp_path / "x")


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
