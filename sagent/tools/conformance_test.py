"""Cross-seam conformance checks over every default tool.

The display-knob bugs all lived in a gap between two well-tested seams:
``coerce_kwargs`` was only ever exercised against locally-declared stub
classes, and the renderer only against ``RecordingPrinter``. Each side
passed; nothing instantiated a REAL tool and pushed it through the real
machinery. These tests cross that gap.
"""

from __future__ import annotations

from typing import get_type_hints

import inspect

import pytest

from sagent.bin.cli import DEFAULT_TOOLS, resolve_tools
from sagent.tools.tool_spec import CLI_SETTABLE, coerce_kwargs
from sagent.types.tools import Tool


def _settable_keys(cls: type) -> list[str]:
    """Return the constructor parameters opted into ``--tool``.

    An unresolvable forward reference means no knob can be declared
    either, so an empty list is the honest answer.
    """
    try:
        hints = get_type_hints(cls.__init__, include_extras=True)
    except (NameError, TypeError):
        return []
    return [
        name
        for name, p in inspect.signature(cls.__init__).parameters.items()
        if name != "self"
        and p.kind is p.KEYWORD_ONLY
        and CLI_SETTABLE in getattr(hints.get(name), "__metadata__", ())
    ]


@pytest.mark.parametrize("name", DEFAULT_TOOLS)
def test_tool_instance_keeps_its_model_facing_description(name: str) -> None:
    """A display knob must never shadow the ``Tool`` protocol's fields.

    Providers serialize ``tool.description`` off the INSTANCE
    (``agent/background.py`` snapshots it), so a same-named constructor
    parameter ships its own value to the model in place of the tool's
    entire guidance prose.
    """
    tool = resolve_tools([name])[0]
    # A ``property`` is a legitimate way to compute prose per request
    # (WebSearch stamps the current date); what must never happen is an
    # INSTANCE attribute shadowing the class's prose with a knob value.
    # A slotted tool has no ``__dict__`` at all, which is the strongest
    # form of the same guarantee.
    shadowed = getattr(tool, "__dict__", {}).get("description")
    assert not isinstance(shadowed, str), (
        f"{name}: a constructor parameter shadows the model-facing description"
    )
    assert len(tool.description) > 40, "description looks like a knob, not prose"


@pytest.mark.parametrize("name", DEFAULT_TOOLS)
def test_tool_satisfies_the_protocol(name: str) -> None:
    tool = resolve_tools([name])[0]
    assert isinstance(tool, Tool)


@pytest.mark.parametrize("name", DEFAULT_TOOLS)
def test_every_settable_knob_round_trips_from_the_command_line(name: str) -> None:
    """Each declared knob must survive ``--tool NAME.key=value``.

    The knobs are declared through PEP 695 aliases, which
    ``coerce_kwargs`` must resolve; a stub class declaring an inline
    ``Literal`` takes a different branch and hides the failure.
    """
    cls = type(resolve_tools([name])[0])
    for key in _settable_keys(cls):
        default = inspect.signature(cls.__init__).parameters[key].default
        coerced = coerce_kwargs(cls, {key: str(default)})
        assert coerced == {key: default}, f"{name}.{key} did not round-trip"


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
