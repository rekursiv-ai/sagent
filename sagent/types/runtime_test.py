"""Tests for ``types.runtime`` event documentation."""

from __future__ import annotations

from sagent.types.runtime import Recompact


def test_recompact_event_docstring_describes_compact_alias() -> None:
    assert Recompact.__doc__ is not None
    assert "alias" in Recompact.__doc__.lower()
    assert "/compact" in Recompact.__doc__
    assert "reload" not in Recompact.__doc__.lower()


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
