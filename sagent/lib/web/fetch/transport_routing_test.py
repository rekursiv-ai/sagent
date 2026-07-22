"""Tests for persistent automatic web-fetch transport routing."""

from pathlib import Path

import pytest

from sagent.lib.web.fetch.transport_routing import (
    remember_zendriver_domain,
    zendriver_domains,
    zendriver_domains_path,
)


def test_default_path_uses_per_user_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    assert zendriver_domains_path() == tmp_path / "loop/web/zendriver-domains.txt"


def test_default_remember_does_not_modify_bundled_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundled = Path(__file__).parent / "zendriver-domains.txt"
    # The bundled default list is optional; read defensively so this asserts
    # "the bundle is never mutated" whether it is present or absent.
    before = bundled.read_bytes() if bundled.exists() else None
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    remember_zendriver_domain("learned.example")

    after = bundled.read_bytes() if bundled.exists() else None
    assert after == before
    assert zendriver_domains_path().read_text() == "learned.example\n"
    assert "learned.example" in zendriver_domains()


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


def test_read_casefolds_existing_entries(tmp_path: Path) -> None:
    # Runtime lookups key on urlparse().hostname (always lowercase), so a
    # bundled or hand-edited mixed-case entry must still match. The writer
    # casefolds; the reader must too.
    path = tmp_path / "domains.txt"
    path.write_text("Foo.Example\nBar.EXAMPLE\n")

    assert zendriver_domains(path=path) == frozenset({"foo.example", "bar.example"})


def test_unreadable_domain_list_degrades_to_empty(tmp_path: Path) -> None:
    # An optional learned-route cache with corrupt (non-UTF-8) contents must
    # not abort every automatic fetch; it degrades to no learned routing.
    path = tmp_path / "domains.txt"
    path.write_bytes(b"good.example\n\xff\xfe not utf-8\n")

    assert zendriver_domains(path=path) == frozenset({"good.example"}) or (
        zendriver_domains(path=path) == frozenset()
    )


def test_remember_preserves_domains_past_the_read_chunk(tmp_path: Path) -> None:
    # The domain list must be read to EOF before rewrite; a fixed read cap would
    # silently drop every domain past it on the next remember.
    path = tmp_path / "domains.txt"
    # 50k * 25 bytes/line ~= 1.25 MiB, safely past the old 1 MiB read cap.
    many = [f"dom-{i:07d}.example.test" for i in range(50_000)]
    path.write_text("".join(f"{d}\n" for d in many))
    assert path.stat().st_size > (1 << 20)

    remember_zendriver_domain("fresh.example", path=path)

    result = zendriver_domains(path=path)
    assert "fresh.example" in result
    assert many[-1] in result
    assert len(result) == len(many) + 1


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
