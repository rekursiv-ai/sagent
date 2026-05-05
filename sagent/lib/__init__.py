"""Standalone cross-module utilities.

Modules here MUST have zero dependency on the rest of ``sagent`` so they
can be imported freely without pulling in the agent / provider / tool
runtime. A util belongs in ``lib/`` when (a) it's a simple function and
(b) it's used in more than one place.
"""
