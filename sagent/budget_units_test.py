"""The budget is denominated in TOKENS, and nothing re-derives it.

Every bound in this package answers one question -- "will this fit the
model's context?" -- whose unit is the token. Any code that converts that
budget into another unit needs a ratio it cannot know: a chars-per-line,
a chars-per-match, a flat chars-per-token. Each such guess was wrong by a
factor that depended on the CONTENT, so the same tool returned 46% of its
budget on narrow source and 4092% on wide JSONL.

These tests are structural because the defect is structural: a
conversion added anywhere re-opens the hole, and it will pass every
behavioral test written against the content its author had in mind.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Final

import ast

import pytest


_CWD: Final = Path(__file__).resolve().parent

# Files exempt from the ratio ban, each for a stated reason. Kept minimal on
# purpose: an exemption that no longer covers a real division is a licence
# nobody is using, and it silently re-opens the hole if one is added later.
_RATIO_ALLOWED: Final = frozenset(
    {
        # Providers with no local tokenizer: the ratio IS their estimator.
        "providers/google/api.py",
        "providers/google/cli.py",
        # tiktoken-less compat vendors (Kimi, Qwen, MiniMax) fall back to it.
        "providers/openai/token_count.py",
        # The single no-agent fallback, and the test seam mirroring it.
        "agent/state.py",
        "testing.py",
    }
)


def test_every_ratio_exemption_is_load_bearing() -> None:
    """An exemption must name a file that exists AND still needs one.

    The catalogs moved to ``sagent.catalog`` and took their measured
    divisors with them, leaving three entries naming files that no longer
    exist plus three that no longer divide -- six licences covering nothing.
    """
    stale = sorted(rel for rel in _RATIO_ALLOWED if not (_CWD / rel).exists())
    assert not stale, f"exemption names a file that does not exist: {stale}"
    unused = sorted(
        rel
        for rel in _RATIO_ALLOWED
        if not _ratio_divisions((_CWD / rel).read_text(encoding="utf-8"), rel)
    )
    assert not unused, f"exemption is no longer needed; drop it: {unused}"


_BANNED_NAMES: Final = (
    "_ASSUMED_CHARS_PER_LINE",
    "_ASSUMED_CHARS_PER_MATCH",
    "TOOL_RESULT_MAX_CHARS",
    "PERSIST_EXEMPT_TOOLS",
    "_FALLBACK_LINE_LIMIT",
    "_FALLBACK_MAX_RESULTS",
    "_FALLBACK_KEEP_FIRST",
    "PREVIEW_CHARS",
)


@cache
def _sources() -> tuple[tuple[str, str], ...]:
    """Return ``(relative_path, text)`` for every shipped non-test module.

    ``examples/`` is excluded: its models are deliberately offline and
    tokenizer-free, so a ratio there IS the estimator, not a conversion
    away from one.

    Dot-prefixed directories are skipped because this walk is over a
    WORKING TREE, not the shipped package: a developer venv at
    ``sagent/.venv`` put 38k site-packages files in scope, one of which is
    latin-1 by design (joblib's encoding fixture) and raised
    UnicodeDecodeError before any assertion ran. That also covers
    ``.export/``, whose staged copy would otherwise double every hit.

    Cached because all nine tests in this module walk the same tree, and
    re-reading it per test was the module's entire runtime.
    """
    found: list[tuple[str, str]] = []
    for path in sorted(_CWD.rglob("*.py")):
        rel = path.relative_to(_CWD)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if path.name.endswith("_test.py") or rel.parts[0] == "examples":
            continue
        found.append((str(rel), path.read_text(encoding="utf-8")))
    return tuple(found)


@pytest.mark.parametrize("banned", _BANNED_NAMES)
def test_the_deleted_constants_stay_deleted(banned: str) -> None:
    """No module may reintroduce a char-, line-, or match-shaped cap.

    Each of these named a bound in the wrong unit. ``PERSIST_EXEMPT_TOOLS``
    is here for the same reason: it exempted ``Read`` from disk offload on
    the grounds that Read bounded itself, which stopped being true the
    moment that bound was expressed in lines.
    """
    offenders = [rel for rel, text in _sources() if banned in text]
    assert not offenders, (
        f"{banned} is back in {offenders}. It expresses a budget in a unit"
        " the provider does not enforce; bound by tokens instead."
    )


def _ratio_divisions(text: str, rel: str) -> list[int]:
    """Line numbers where ``text`` divides a string length by a constant.

    ``len(sequence) // n`` takes a FRACTION of a collection -- a retry step,
    a midpoint. Only ``len(<str>) // <const>`` is the tokenizer guess, so
    both a literal divisor and a textually-named argument are required.
    """
    lines: list[int] = []
    for node in ast.walk(ast.parse(text, rel)):
        if not isinstance(node, ast.BinOp) or not isinstance(
            node.op, (ast.FloorDiv, ast.Div)
        ):
            continue
        if not isinstance(node.left, ast.Call):
            continue
        fn = node.left.func
        if not (isinstance(fn, ast.Name) and fn.id == "len"):
            continue
        if not isinstance(node.right, ast.Constant):
            continue
        arg = node.left.args[0] if node.left.args else None
        if isinstance(arg, ast.Name) and arg.id in (
            "text",
            "content",
            "body",
            "unit",
            "s",
        ):
            lines.append(node.lineno)
    return lines


def test_no_module_divides_by_a_chars_per_token_ratio() -> None:
    """``len(text) // 4`` is a tokenizer guess wearing arithmetic.

    Measured against real session traffic, the true ratio runs 3.25 to
    4.15 depending on content class -- so a flat 4 undercounts the densest
    content by 14%, and that content is tool results, which dominate.
    Call ``approx_text_tokens`` and let the model answer.
    """
    offenders = [
        f"{rel}:{line}"
        for rel, text in _sources()
        if rel not in _RATIO_ALLOWED
        for line in _ratio_divisions(text, rel)
    ]
    assert not offenders, (
        f"a chars-per-token division survives at {offenders}; use"
        " approx_text_tokens so the ACTIVE model's tokenizer decides"
    )


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
