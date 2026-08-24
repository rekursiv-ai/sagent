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

from collections.abc import Iterator
from pathlib import Path
from typing import Final

import ast

import pytest


_PACKAGE: Final = Path(__file__).resolve().parent

# Files exempt from the ratio ban, each for a stated reason.
_RATIO_ALLOWED: Final = frozenset(
    {
        # Declares the ratio as catalog DATA, per model, measured.
        "providers/anthropic/catalog.py",
        "providers/google/catalog.py",
        "providers/openai/catalog.py",
        # Providers with no local tokenizer: the ratio IS their estimator.
        "providers/anthropic/api.py",
        "providers/anthropic/cli.py",
        "providers/anthropic/sub.py",
        "providers/google/api.py",
        "providers/google/cli.py",
        "providers/google/sub.py",
        "providers/openai/compat.py",
        # The type that owns the field, and the two re-attach caps that
        # legitimately bound FILE BYTES read before any tokenizer runs.
        "types/model.py",
        # The single no-agent fallback, and the test seam mirroring it.
        "agent/state.py",
        "testing.py",
    }
)

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


def _sources() -> Iterator[tuple[str, str]]:
    """Yield ``(relative_path, text)`` for every shipped non-test module.

    ``examples/`` is excluded: its models are deliberately offline and
    tokenizer-free, so a ratio there IS the estimator, not a conversion
    away from one.
    """
    for path in sorted(_PACKAGE.rglob("*.py")):
        rel = str(path.relative_to(_PACKAGE))
        if path.name.endswith("_test.py") or ".export/" in rel:
            continue
        if rel.startswith("examples/"):
            continue
        yield rel, path.read_text(encoding="utf-8")


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


def test_no_module_divides_by_a_chars_per_token_ratio() -> None:
    """``len(text) // 4`` is a tokenizer guess wearing arithmetic.

    Measured against real session traffic, the true ratio runs 3.25 to
    4.15 depending on content class -- so a flat 4 undercounts the densest
    content by 14%, and that content is tool results, which dominate.
    Call ``approx_text_tokens`` and let the model answer.
    """
    offenders: list[str] = []
    for rel, text in _sources():
        if rel in _RATIO_ALLOWED:
            continue
        tree = ast.parse(text, rel)
        for node in ast.walk(tree):
            if not isinstance(node, ast.BinOp) or not isinstance(
                node.op, (ast.FloorDiv, ast.Div)
            ):
                continue
            if not isinstance(node.left, ast.Call):
                continue
            fn = node.left.func
            if not (isinstance(fn, ast.Name) and fn.id == "len"):
                continue
            # ``len(sequence) // n`` takes a FRACTION of a collection --
            # a retry step, a midpoint. Only ``len(<str>) // <const>`` is
            # the tokenizer guess, so require a literal divisor AND an
            # argument that is textual by name.
            if not isinstance(node.right, ast.Constant):
                continue
            arg = node.left.args[0] if node.left.args else None
            name = arg.id if isinstance(arg, ast.Name) else ""
            if name in ("text", "content", "body", "unit", "s"):
                offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, (
        f"a chars-per-token division survives at {offenders}; use"
        " approx_text_tokens so the ACTIVE model's tokenizer decides"
    )


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
