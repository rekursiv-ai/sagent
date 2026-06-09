"""Per-role sagent Agent factories.

Each role module exposes a single ``build()`` function returning a
configured :class:`sagent.agent.Agent`. The system prompt is loaded
from the sibling ``<role>.md`` file at build time.
"""
