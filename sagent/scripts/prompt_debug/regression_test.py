r"""Integration regression guard for proven prompt/config fixes.

Each built-in case pins a behavior a ``debug-prompt`` experiment proved fixed.
This re-runs every case on the system prompt the sagent CLI assembles, through
:func:`harness.run_case`, and fails when the flagged rate exceeds the case's
tolerance.

It spawns live model turns, so its markers deselect it by default; run this
file explicitly with ``RUN_REAL_LLM=1``.

Each provider is built as the sagent CLI builds it, from a ``ModelCatalog`` on
its own login; a provider without credentials skips.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import asyncio

import pytest

from sagent.model.exceptions import AuthError
from sagent.providers.catalog import ModelCatalog
from sagent.scripts.prompt_debug.harness import Case, load_cases, run_case


if TYPE_CHECKING:
    from sagent.scripts.prompt_debug.harness import CaseResult


pytestmark = [
    pytest.mark.network_anthropic,
    pytest.mark.cli_real_llm,
]


async def _guard(case: Case, provider: str) -> CaseResult:
    """Run ``case`` on the provider's default model; skip when it cannot sign in."""
    async with ModelCatalog() as catalog:
        try:
            return await run_case(case, model=catalog[provider]["default"])
        except AuthError as e:
            raise pytest.skip.Exception(f"{provider} unavailable: {e}") from e


@pytest.mark.parametrize(
    "provider",
    ["anthropic", "openai"],
)
@pytest.mark.parametrize(
    "case",
    [pytest.param(case, id=case.name) for case in load_cases()],
)
def test_prompt_does_not_regress(case: Case, provider: str) -> None:
    """The guarded behavior stays within tolerance on the real system prompt."""
    result = asyncio.run(_guard(case, provider))
    assert result.passed, (
        f"{case.name} [{provider}]: flagged {result.flagged}"
        f"/{len(result.verdicts)} of n={case.n}"
        f" (tolerance {case.max_flag_rate:.2f}) --"
        " the system prompt may have regressed."
    )


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
