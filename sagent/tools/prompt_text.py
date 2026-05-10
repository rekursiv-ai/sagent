"""Utilities for text copied from untrusted files into prompts."""

from __future__ import annotations

import html


def escape_prompt_text(text: str) -> str:
    """Escape text before embedding it inside prompt wrapper markup.

    Args:
      text: Untrusted text loaded from user-controlled files or directives.

    Returns:
      escaped: Text with XML-ish wrapper metacharacters escaped.

    """
    return html.escape(text, quote=False)
