"""Tests for ``--tool NAME.key=value`` override parsing."""

from __future__ import annotations

from typing import Annotated, Literal

import pytest

from sagent.tools.tool_spec import (
    CLI_SETTABLE,
    ToolSpecError,
    coerce_kwargs,
    parse_tool_overrides,
)


type _Toggle = Literal["on", "off"]
type _NestedToggle = _Toggle


def test_parses_dotted_override() -> None:
    assert parse_tool_overrides(["Bash.output=none"]) == {"Bash": {"output": "none"}}


def test_merges_repeated_flags_for_one_tool() -> None:
    got = parse_tool_overrides(["Bash.output=none", "Bash.input=actual"])
    assert got == {"Bash": {"output": "none", "input": "actual"}}


def test_value_may_contain_equals_and_commas() -> None:
    """Only the FIRST ``=`` splits, so values need no quoting."""
    got = parse_tool_overrides(["Bash.output=a=b,c"])
    assert got == {"Bash": {"output": "a=b,c"}}


@pytest.mark.parametrize("bad", ["Bash", "Bash.output", "=none", ".output=none"])
def test_malformed_override_is_rejected(bad: str) -> None:
    with pytest.raises(ToolSpecError):
        parse_tool_overrides([bad])


def test_coerces_int_and_literal_by_annotation() -> None:
    class _Tool:
        def __init__(
            self,
            *,
            output: Annotated[str, CLI_SETTABLE] = "actual",
            output_lines: Annotated[int, CLI_SETTABLE] = 20,
        ) -> None:
            self.output = output
            self.output_lines = output_lines

    got = coerce_kwargs(_Tool, {"output_lines": "50"})
    assert got == {"output_lines": 50}
    assert isinstance(got["output_lines"], int)


def test_unknown_key_names_the_settable_ones() -> None:
    """A typo must abort loudly; silently ignoring it looks like a no-op."""

    class _Tool:
        def __init__(self, *, output: Annotated[str, CLI_SETTABLE] = "actual") -> None:
            self.output = output

    with pytest.raises(ToolSpecError) as excinfo:
        coerce_kwargs(_Tool, {"otuput": "none"})
    assert "otuput" in str(excinfo.value)
    assert "output" in str(excinfo.value)


def test_bad_literal_value_names_the_choices() -> None:
    class _Tool:
        def __init__(
            self,
            *,
            output: Annotated[Literal["actual", "none"], CLI_SETTABLE] = "actual",
        ) -> None:
            self.output = output

    with pytest.raises(ToolSpecError) as excinfo:
        coerce_kwargs(_Tool, {"output": "bogus"})
    assert "bogus" in str(excinfo.value)
    assert "actual" in str(excinfo.value)


class _AliasTool:
    """Declares its knob through a PEP 695 alias, exactly as tools do."""

    def __init__(
        self,
        *,
        output: Annotated[_Toggle, CLI_SETTABLE] = "on",
        wrap: Annotated[_NestedToggle, CLI_SETTABLE] = "on",
    ) -> None:
        self.output = output
        self.wrap = wrap


def test_pep695_literal_alias_is_settable() -> None:
    """``type X = Literal[...]`` must coerce like an inline ``Literal``.

    A PEP 695 alias binds a ``TypeAliasType``, whose ``get_origin`` is
    ``None`` -- so the ``Literal`` branch misses and the value falls
    through to "not settable". Every display toggle is such an alias, so
    the whole ``--tool`` feature dies on it.
    """
    assert coerce_kwargs(_AliasTool, {"output": "off"}) == {"output": "off"}


def test_pep695_alias_still_rejects_an_invalid_value() -> None:
    """Unwrapping the alias must not lose its membership check."""
    with pytest.raises(ToolSpecError) as excinfo:
        coerce_kwargs(_AliasTool, {"output": "bogus"})
    assert "bogus" in str(excinfo.value)


def test_nested_alias_resolves_to_its_value() -> None:
    """An alias of an alias must unwrap all the way down."""
    assert coerce_kwargs(_AliasTool, {"wrap": "off"}) == {"wrap": "off"}


def test_unmarked_parameter_is_not_settable() -> None:
    """Opt-in: a secret or object graph must never reach the CLI.

    ``Slack.token`` is a ``str`` like any display knob, so a rule based
    on the type alone would happily accept it on argv.
    """

    class _Tool:
        def __init__(self, *, token: str = "", peers: object = None) -> None:
            self.token = token
            self.peers = peers

    with pytest.raises(ToolSpecError) as excinfo:
        coerce_kwargs(_Tool, {"token": "xoxb-secret"})
    assert "(none)" in str(excinfo.value)


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
