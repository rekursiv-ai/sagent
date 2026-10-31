"""What settings raise: a refusal the user reads, and a request for help.

Typical usage example:

  try:
      said = await setting.apply(args)
  except SettingError as e:
      ...  # Say why, having changed nothing.

  try:
      parsed = parser.parse_args(words)
  except HelpRequestedError:
      ...  # Return the help rather than print it.
"""

from __future__ import annotations

from sagent.runtime.exceptions import UserFacingError


__all__ = [
    "HelpRequestedError",
    "SettingError",
]


class SettingError(UserFacingError, ValueError):
    """What a user gave a setting does not fit it; the message says what would."""


class HelpRequestedError(Exception):
    """``-h`` or ``--help`` was given: an action's help is wanted, not its work."""
