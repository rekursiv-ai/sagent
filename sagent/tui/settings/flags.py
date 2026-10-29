"""A command line split in two: the flags argparse knows, and settings.

Every setting the prompt reaches as ``/name VALUE`` the command line reaches
as ``--name VALUE``, but the settings exist only once the agent is built, so
argparse cannot be told them first.

Typical usage example:

  rest, given = flags(sys.argv[1:], static={"--resume", "--help", ...})
  args = parser.parse_args(rest)
  for name, value in given:               # [("effort", "high"), ...]
      ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Collection, Sequence


__all__ = [
    "flags",
]


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
