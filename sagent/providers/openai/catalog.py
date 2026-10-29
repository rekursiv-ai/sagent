"""The models each section of the OpenAI catalog serves."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sagent.providers import table


if TYPE_CHECKING:
    from sagent.providers.table import ModelTable


__all__ = [
    "models",
]


def models(section: str) -> ModelTable:
    """Return the models one section of ``openai.yaml`` serves.

    Args:
      section: The section, as ``openai-sub``.

    Returns:
      table: Its models, each setting the section does not send fixed.

    """
    return table.models("sagent.providers.openai", "openai.yaml", provider=section)
