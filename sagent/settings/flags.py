"""A command line as settings read it: flags and settings apart, and an action's words.

Every setting the prompt reaches as ``/name VALUE`` the command line reaches
as ``--name VALUE``, but the settings exist only once the agent is built, so
argparse cannot be told them first.

An action, such as ``mcp.add``, is reached as ``/mcp.add ARGS`` and as
``sagent mcp add ARGS``; ``ActionParser`` reads ARGS the same way for both.

Typical usage example:

  rest, given = flags(sys.argv[1:], static={"--resume", "--help", ...})
  args = parser.parse_args(rest)
  for name, value in given:               # [("effort", "high"), ...]
      ...

  parser = ActionParser(("mcp", "remove"), description="Remove an MCP server.")
  parser.add_argument("name")
  parsed = parser.read(args)              # ``None`` when help was asked for.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn, override

import argparse
import shlex

from sagent.settings.exceptions import HelpRequestedError, SettingError


if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from _typeshed import SupportsWrite


__all__ = [
    "ActionParser",
    "flags",
]


class ActionParser(argparse.ArgumentParser):
    """Reads an action's arguments as a command line would; raises, never exits.

    ``argparse``'s own exits on a mistake or ``--help``, which typed at the
    prompt would end the session.

    Args:
      name: The action's name, such as ``("mcp", "add")``.
      description: What the action does, for its help.

    """

    def __init__(self, name: tuple[str, ...], *, description: str) -> None:
        super().__init__(
            prog=f"sagent {' '.join(name)}",
            description=description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            allow_abbrev=False,
        )

    @property
    def options(self) -> frozenset[str]:
        """Return every option string this parser takes, such as ``--headless``."""
        return frozenset(o for action in self._actions for o in action.option_strings)

    def read(self, args: str) -> argparse.Namespace | None:
        """Return ``args``, split as a shell would, parsed; ``None`` when help is asked.

        Args:
          args: What followed the action's name, as one string.

        Returns:
          parsed: What was given; ``None`` for ``-h`` or ``--help`` before ``--``.

        Raises:
          SettingError: ``args`` does not split, or does not fit the parser.

        """
        try:
            words = shlex.split(args)
        except ValueError as e:
            self.error(str(e))
        try:
            return self.parse_args(words)
        except HelpRequestedError:
            return None

    # Help is asked for as argparse reads it, before it checks what is required, so
    # ``--help`` with nothing else still gets it; after ``--`` it is an argument.
    @override
    def print_help(self, file: SupportsWrite[str] | None = None) -> NoReturn:
        del file
        raise HelpRequestedError

    @override
    def error(self, message: str) -> NoReturn:
        raise SettingError(f"{message}; see {self.prog} --help")

    @override
    def exit(self, status: int = 0, message: str | None = None) -> NoReturn:
        # ``argparse`` exits only after ``error``, or for help ``print_help`` raises.
        raise SettingError(message or f"see {self.prog} --help")


def flags(
    argv: Sequence[str],
    *,
    static: Collection[str],
) -> tuple[list[str], list[tuple[str, str]]]:
    """Return what argparse should parse, and each setting given with its value.

    A word ``--NAME`` or ``--NAME=VALUE`` argparse does not know is a setting;
    without ``=``, the next word is its value. Everything else is left in
    order for argparse.

    Args:
      argv: The command line, without the program.
      static: Every option string argparse knows, such as ``--resume``.

    Returns:
      rest: The words argparse parses.
      given: Each setting's name, without ``--``, and its value, in order.

    Raises:
      ValueError: A setting has no name, or no value.

    """
    rest: list[str] = []
    given: list[tuple[str, str]] = []
    words = iter(argv)
    for word in words:
        if word == "--":
            rest.extend([word, *words])
            break
        option, equals, value = word.partition("=")
        if not option.startswith("--") or option in static:
            rest.append(word)
            continue
        if option == "--":
            raise ValueError(f"{word!r} names no setting")
        if not equals:
            value = next(words, "--")
            if value.startswith("--"):
                raise ValueError(f"{option} needs a value")
        given.append((option.removeprefix("--"), value))
    return rest, given
