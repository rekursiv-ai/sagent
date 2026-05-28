"""Duration formatting helpers.

Used by retry banners (``agent.retry``) and renderer suspension text
(``repl.render``) to share one humanization rule for wait durations.
"""

from __future__ import annotations


def humanize_duration(seconds: float) -> str:
    """Format ``seconds`` as ``Xh Ym`` / ``Ym Zs`` / ``Zs``.

    Args:
      seconds: Non-negative duration to render; negatives are clamped to 0.

    Returns:
      label: Concise human-readable duration.

    """
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"
