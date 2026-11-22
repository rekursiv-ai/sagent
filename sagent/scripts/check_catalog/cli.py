"""Compare the shipped provider catalogs with what each vendor publishes.

Reads every model's own page, the vendor's price page, and its thinking and
effort guides, then prints each fact the catalog states that the vendor does not.
A section a login gates is checked only when named, against what its login is
served now. A page that cannot be fetched leaves its facts unstated and is
named. Exits 1 when any fact disagrees without a recorded reason or any page is
unreachable, so it can run on a schedule; it needs the network and is never a
pre-commit hook.

Examples:
  python -m sagent.scripts.check_catalog
  python -m sagent.scripts.check_catalog --provider openai --provider google
  python -m sagent.scripts.check_catalog --provider anthropic --verbose
  python -m sagent.scripts.check_catalog --write

"""

from __future__ import annotations

from datetime import UTC, date, datetime
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import argparse
import functools

import pypdfium2 as pdfium

from sagent.model.exceptions import AuthError
from sagent.providers import anthropic, google, openai
from sagent.providers.table import ModelTable
from sagent.scripts.check_catalog.sources import (
    Fetch,
    Finding,
    check_anthropic,
    check_google,
    check_openai,
)
from sagent.scripts.check_catalog.write import write


if TYPE_CHECKING:
    from importlib.resources.abc import Traversable
    from types import ModuleType

    import httpx2
else:
    from wrapt import lazy_import

    httpx2 = lazy_import("httpx2")


def main() -> int:
    """Check the catalogs asked for, and print what disagrees.

    Returns:
      exit_code: ``1`` if any fact disagrees without a recorded reason, or any
        page is unreachable, else ``0``.

    """
    parser = argparse.ArgumentParser(
        # The docstring is absent only under ``python -OO``, which no test runs.
        description=(__doc__ or "").strip(),  # pragma: no mutate
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(Flags, parser.parse_args())
    # Vendors date prices in their own zone; UTC is within a day of every one.
    now = datetime.now(UTC)  # pragma: no mutate
    unreachable: dict[str, str] = {}
    unreadable: dict[str, str] = {}
    findings = check(
        flags.provider or list(_checks()),
        fetch=functools.cache(functools.partial(_fetch, unreachable=unreachable)),
        today=now.date(),
        unreadable=unreadable,
    )
    wrote: dict[str, int] = {}
    if flags.write:
        findings, wrote = _written(findings)
    shown: set[str] = {"mismatch"}
    if flags.verbose:
        shown |= {"ok", "unstated", "waived"}
    for finding in findings:
        if finding.status in shown:
            print(_line(finding))
    for name, count in wrote.items():
        print(f"wrote {count} fact{'' if count == 1 else 's'} to {name}")
    for url, error in unreachable.items():
        print(f"unreachable {url}: {error}")
    for provider, error in unreadable.items():
        print(f"unreadable {provider}: {error}")
    counts = {
        s: sum(f.status == s for f in findings)
        for s in ("mismatch", "waived", "unstated", "ok")
    }
    counts["unreachable"] = len(unreachable)
    if unreadable:
        counts["unreadable"] = len(unreadable)
    print(", ".join(f"{n} {s}" for s, n in counts.items()))
    return 1 if counts["mismatch"] or unreachable or unreadable else 0


class Flags(Protocol):
    """Parsed command-line flags."""

    provider: list[str] | None
    verbose: bool
    write: bool


def check(
    providers: list[str],
    *,
    fetch: Fetch,
    today: date,
    unreadable: dict[str, str],
) -> list[Finding]:
    """Return the findings for each named section of its vendor's shipped catalog.

    Args:
      providers: Section names, each a key of ``_every_check``: the API sections
        (``anthropic``, ``openai``, ``google``) and those a login gates.
      fetch: Returns a URL's text.
      today: The day whose prices are compared.
      unreadable: Filled with each vendor whose pages no longer parse, and why;
        its facts are left out, the other vendors' kept.

    Returns:
      findings: Every fact compared, in provider and catalog order.

    """
    findings: list[Finding] = []
    for provider in providers:
        package, run = _every_check()[provider]
        # The shipped document, not a user's override: the check is of what ships.
        table = ModelTable.from_yaml(_document(package), section=provider)
        try:
            findings += run(table, fetch=fetch, today=today)
        except (ValueError, AuthError) as e:
            unreadable[provider] = str(e)
    return findings


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument(
        "--provider",
        action="append",
        choices=list(_every_check()),
        help="Section to check; repeat for several. Default: each whose vendor's"
        " pages state it, none that needs a login.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Also print facts that agree, that the vendor does not state, or that"
        " differ on purpose.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the vendor's facts into the shipped catalog: each value a model's"
        " own row states, each model it lists and the catalog lacks, and each"
        " section's omits; the rest are printed for a person.",
    )


def _checks() -> dict[str, tuple[ModuleType, _Check]]:
    """Return each section the default run checks: those its vendors' pages state."""
    return {
        "anthropic": (anthropic, check_anthropic),
        "openai": (openai, check_openai),
        "google": (google, check_google),
    }


# Run only when named: a check that needs a login fails wherever there is none.
def _every_check() -> dict[str, tuple[ModuleType, _Check]]:
    """Return every section that can be checked, the signed-in ones too."""
    return {
        **_checks(),
    }


def _document(package: ModuleType) -> Traversable:
    """Return the catalog ``package`` ships: ``google`` holds ``google.yaml``."""
    return files(package) / f"{package.__name__.rpartition('.')[2]}.yaml"


class _Check(Protocol):
    def __call__(
        self,
        table: ModelTable,
        *,
        fetch: Fetch,
        today: date,
    ) -> list[Finding]: ...


# Vendors refuse a client's default agent; some Gemini model cards redirect to a PDF.
# ``""``, unreachable.
#
# A no-break space reads as a space.
def _fetch(url: str, *, unreachable: dict[str, str]) -> str:
    """Return ``url``'s body as text, a PDF's as its pages' text."""
    try:
        response = httpx2.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 sagent-catalog-check"},
            follow_redirects=True,
            timeout=60.0,
        )
        response.raise_for_status()
    except httpx2.HTTPError as e:
        unreachable[url] = str(e)
        return ""
    body = response.content
    text = (
        "\n".join(
            page.get_textpage().get_text_range() for page in pdfium.PdfDocument(body)
        )
        if body.startswith(b"%PDF")
        else body.decode()
    )
    # Vendors set a no-break space between a number and its unit.
    return text.replace("\u00a0", " ")


def _line(finding: Finding) -> str:
    """Return a finding as one line: status, where, both sides, and the source."""
    reason = f"  ({finding.waiver})" if finding.status == "waived" else ""
    return (
        f"{finding.status:<8} {finding.provider} {finding.model_id}"
        f" {finding.field}: catalog={finding.catalog}"
        f" source={finding.source}  {finding.url}{reason}"
    )


def _written(findings: list[Finding]) -> tuple[list[Finding], dict[str, int]]:
    """Write each mismatch into its shipped document; return the rest and the counts."""
    by_document: dict[Path, list[Finding]] = {}
    for finding in findings:
        if finding.status == "mismatch":
            package, _ = _every_check()[finding.provider]
            by_document.setdefault(Path(str(_document(package))), []).append(finding)
    done: set[Finding] = set()
    wrote: dict[str, int] = {}
    for path, mismatches in by_document.items():
        written = set(mismatches) - set(write(path, mismatches))
        done |= written
        if written:
            wrote[path.name] = len(written)
    return [f for f in findings if f not in done], wrote
