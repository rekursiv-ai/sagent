"""Tests for check_catalog's command line: flags, the exit code, and fetching."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import sys

import httpx
import pypdfium2 as pdfium
import pytest

from sagent.providers import anthropic, google, openai
from sagent.providers.table import ModelTable
from sagent.scripts.check_catalog import cli
from sagent.scripts.check_catalog.sources import (
    Fetch,
    Finding,
    check_anthropic,
    check_google,
    check_openai,
)


@pytest.mark.parametrize(
    ("sources", "code", "printed"),
    [
        (("a",), 0, "0 mismatch, 0 waived, 0 unstated, 1 ok, 0 unreachable\n"),
        (
            ("b", "a"),
            1,
            (
                "mismatch anthropic m-1.0 f: catalog=a source=b  https://u\n"
                "1 mismatch, 0 waived, 0 unstated, 1 ok, 0 unreachable\n"
            ),
        ),
    ],
)
def test_main_prints_disagreements_and_exits_1_only_for_them(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    sources: tuple[str, ...],
    code: int,
    printed: str,
) -> None:
    calls = _fake_checks(monkeypatch, sources=sources)
    monkeypatch.setattr(sys, "argv", ["cli.py", "--provider", "anthropic"])
    assert cli.main() == code
    assert capsys.readouterr().out == printed
    assert [call.provider for call in calls] == ["anthropic"]


def test_verbose_prints_every_status_and_every_vendor_is_the_default(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = _fake_checks(monkeypatch, sources=("a", None, "w", "b"))
    monkeypatch.setattr(sys, "argv", ["cli.py", "--verbose"])
    assert cli.main() == 1
    lines = capsys.readouterr().out.splitlines()
    assert lines[:4] == [
        "ok       anthropic m-1.0 f: catalog=a source=a  https://u",
        "unstated anthropic m-1.0 f: catalog=a source=None  https://u",
        "waived   anthropic m-1.0 f: catalog=a source=w  https://u  (on purpose)",
        "mismatch anthropic m-1.0 f: catalog=a source=b  https://u",
    ]
    assert lines[-1] == "3 mismatch, 3 waived, 3 unstated, 3 ok, 0 unreachable"
    assert [call.provider for call in calls] == ["anthropic", "openai", "google"]


def test_an_unreachable_page_leaves_its_facts_unstated_and_fails_the_run(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One page that fails to arrive must not hide every other fact."""

    def fetching(table: ModelTable, *, fetch: Fetch, today: date) -> list[Finding]:
        del table, today
        return [
            Finding(
                provider="anthropic",
                model_id="m-1.0",
                field=url,
                catalog="a",
                source=fetch(url) or None,
                url=url,
            )
            for url in ("https://down", "https://up", "https://down")
        ]

    def get(url: str, **kwargs: object) -> httpx.Response:
        del kwargs
        if url == "https://down":
            raise httpx.ConnectTimeout("handshake timed out")
        return httpx.Response(200, content=b"a", request=httpx.Request("GET", url))

    monkeypatch.setattr(cli, "_checks", lambda: {"anthropic": (anthropic, fetching)})
    monkeypatch.setattr(httpx, "get", get)
    monkeypatch.setattr(sys, "argv", ["cli.py"])
    assert cli.main() == 1
    assert capsys.readouterr().out == (
        "unreachable https://down: handshake timed out\n"
        "0 mismatch, 0 waived, 2 unstated, 1 ok, 1 unreachable\n"
    )


def test_main_checks_today_through_one_cached_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake_checks(monkeypatch, sources=("a",))
    monkeypatch.setattr(sys, "argv", ["cli.py", "--provider", "google"])
    cli.main()
    sent = _fake_get(monkeypatch, b"page")
    assert [calls[0].fetch("https://x"), calls[0].fetch("https://x")] == ["page"] * 2
    assert len(sent) == 1
    assert isinstance(calls[0].today, date)


def test_help_shows_the_whole_docstring(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["cli.py", "--help"])
    with pytest.raises(SystemExit):
        cli.main()
    shown = capsys.readouterr().out
    assert "\n\nCompare the shipped provider catalogs" in shown
    assert "Examples:\n  python -m sagent.scripts.check_catalog\n" in shown


def test_an_unknown_vendor_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["cli.py", "--provider", "acme"])
    with pytest.raises(SystemExit, match="2"):
        cli.main()
    assert "invalid choice: 'acme'" in capsys.readouterr().err


def test_a_waived_line_says_why() -> None:
    finding = Finding(
        provider="p",
        model_id="m-1.0",
        field="f",
        catalog="a",
        source="b",
        url="u",
        waiver="on purpose",
    )
    assert cli._line(finding) == (
        "waived   p m-1.0 f: catalog=a source=b  u  (on purpose)"
    )


def test_each_provider_is_checked_against_its_shipped_api_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake_checks(monkeypatch, sources=("a",))
    today = date(2026, 11, 18)

    def fetch(url: str) -> str:
        return url

    unreadable: dict[str, str] = {}
    findings = cli.check(
        ["openai", "google"],
        fetch=fetch,
        today=today,
        unreadable=unreadable,
    )
    assert unreadable == {}
    assert len(findings) == 2
    assert [(call.fetch, call.today) for call in calls] == [(fetch, today)] * 2
    wires = {
        call.provider: {row.wire_model_id for row in call.table.rows} for call in calls
    }
    assert "gpt-6.2-astra" in wires["openai"]
    assert "gemini-3.8-flash" in wires["google"]
    # The API section, not the subscription's: it serves every model priced.
    assert "gpt-5.5" in wires["openai"]


def test_every_vendor_is_wired_to_its_own_catalog_and_check() -> None:
    checks = cli._checks()
    assert {name: package for name, (package, _) in checks.items()} == {
        "anthropic": anthropic,
        "openai": openai,
        "google": google,
    }
    assert [run for _, run in checks.values()] == [
        check_anthropic,
        check_openai,
        check_google,
    ]


def test_fetch_sends_an_agent_follows_redirects_and_decodes_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent = _fake_get(monkeypatch, "plain \u2713 10\u00a0MB".encode())
    # A no-break space reads as a space, so no pattern has to spell both.
    assert cli._fetch("https://x/page.md", unreachable={}) == "plain \u2713 10 MB"
    assert sent == [
        ("https://x/page.md", "Mozilla/5.0 sagent-catalog-check", True, 60.0),
    ]


def test_a_vendor_whose_pages_cannot_be_read_fails_alone(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A page whose layout changed must not hide the other vendors' facts."""
    calls = _fake_checks(monkeypatch, sources=("a",))
    fakes = cli._checks()

    def broken(table: ModelTable, *, fetch: Fetch, today: date) -> list[Finding]:
        del table, fetch, today
        raise ValueError("zip() argument 2 is longer than argument 1")

    monkeypatch.setattr(
        cli,
        "_checks",
        lambda: {**fakes, "openai": (openai, broken)},
    )
    monkeypatch.setattr(sys, "argv", ["cli.py"])
    assert cli.main() == 1
    assert capsys.readouterr().out.splitlines()[-2:] == [
        "unreadable openai: zip() argument 2 is longer than argument 1",
        "0 mismatch, 0 waived, 0 unstated, 2 ok, 0 unreachable, 1 unreadable",
    ]
    assert [call.provider for call in calls] == ["anthropic", "google"]


def test_fetch_records_an_error_status_and_reads_as_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_get(monkeypatch, b"gone", status=404)
    unreachable: dict[str, str] = {}
    assert cli._fetch("https://x/page.md", unreachable=unreachable) == ""
    assert list(unreachable) == ["https://x/page.md"]
    assert "404 Not Found" in unreachable["https://x/page.md"]


def test_fetch_reads_a_pdf_as_its_pages_text_one_line_apart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_get(monkeypatch, _pdf(b"cutoff was", b"January 2025"))
    text = cli._fetch("https://x/card.pdf", unreachable={})
    assert [line.strip() for line in text.split("\n")] == [
        "cutoff was",
        "January 2025",
    ]
    _fake_get(monkeypatch, b"%PDF-1.1 not a real document")
    with pytest.raises(pdfium.PdfiumError):
        cli._fetch("https://x/card.pdf", unreachable={})


def _pdf(*pages: bytes) -> bytes:
    """Return a PDF with one Helvetica line per page, its cross references left out."""
    kids = b" ".join(b"%d 0 R" % (3 + 2 * i) for i in range(len(pages)))
    parts = [
        b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n",
        b"2 0 obj<</Type/Pages/Kids[%s]/Count %d>>endobj\n" % (kids, len(pages)),
        b"9 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n",
    ]
    for i, line in enumerate(pages):
        page, content = 3 + 2 * i, 4 + 2 * i
        stream = b"BT /F1 12 Tf 10 20 Td (%s) Tj ET" % line
        parts += [
            b"%d 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 50]" % page,
            b"/Contents %d 0 R/Resources<</Font<</F1 9 0 R>>>>>>endobj\n" % content,
            b"%d 0 obj<</Length %d>>stream\n%s\nendstream endobj\n"
            % (content, len(stream), stream),
        ]
    return b"".join([*parts, b"trailer<</Root 1 0 R>>\n%%EOF\n"])


def _fake_get(
    monkeypatch: pytest.MonkeyPatch,
    body: bytes,
    *,
    status: int = 200,
) -> list[tuple[str, str, bool, float]]:
    """Replace ``httpx.get`` with one answering ``body``; return what it was asked."""
    sent: list[tuple[str, str, bool, float]] = []

    def get(
        url: str,
        *,
        headers: dict[str, str],
        follow_redirects: bool,
        timeout: float,
    ) -> httpx.Response:
        sent.append((url, headers["User-Agent"], follow_redirects, timeout))
        return httpx.Response(status, content=body, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", get)
    return sent


@dataclass(frozen=True, slots=True, kw_only=True)
class _Call:
    """What one vendor check was called with."""

    provider: str
    table: ModelTable
    fetch: Fetch
    today: date


@dataclass(slots=True, kw_only=True)
class _Recording:
    """A vendor check that records its call and finds one fact per source given.

    A source of ``w`` is waived; ``None`` is unstated.
    """

    provider: str
    sources: tuple[str | None, ...]
    calls: list[_Call] = field(default_factory=list[_Call])

    def __call__(
        self,
        table: ModelTable,
        *,
        fetch: Fetch,
        today: date,
    ) -> list[Finding]:
        self.calls.append(
            _Call(provider=self.provider, table=table, fetch=fetch, today=today),
        )
        return [
            Finding(
                provider=self.provider,
                model_id="m-1.0",
                field="f",
                catalog="a",
                source=source,
                url="https://u",
                waiver="on purpose" if source == "w" else "",
            )
            for source in self.sources
        ]


def _fake_checks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    sources: tuple[str | None, ...],
) -> list[_Call]:
    """Replace each vendor's check with a recording; return the shared call log."""
    calls: list[_Call] = []
    fakes = {
        name: (package, _Recording(provider=name, sources=sources, calls=calls))
        for name, (package, _) in cli._checks().items()
    }
    monkeypatch.setattr(cli, "_checks", lambda: fakes)
    return calls


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
