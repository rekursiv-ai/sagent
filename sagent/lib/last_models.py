"""Cross-session memory of the last model_id used per provider.

Stored at ``~/.sagent/last-models.json`` as a flat
``{provider_class_name: model_id}`` map. Updated on every
``Agent.swap_model`` (and at initial agent construction when a
``ModelSpec`` is supplied). Looked up by the ``/model`` slash
command when the user changes provider without naming a model -- so
``/model provider=OpenAISubscription`` resumes the last OpenAI model
the user picked, falling back to ``OpenAISubscription.DEFAULT_MODEL``
when this provider hasn't been used before.

Concurrent agents racing writes: the last writer wins, via
``atomic_write_bytes``. No locking; the file is tiny and the cost
of a lost update is trivial (next swap re-records).
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import json
import logging

from sagent.lib.atomic_file import atomic_write_bytes


logger = logging.getLogger(__name__)

_PATH = Path.home() / ".sagent" / "last-models.json"


def load() -> dict[str, str]:
    """Return the persisted ``provider → model_id`` map.

    Returns:
      mapping: Empty dict when the file is missing, malformed, or the
          top-level value isn't a JSON object.

    """
    try:
        raw = _PATH.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return {}
    try:
        data: object = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Corrupt %s; treating as empty.", _PATH)
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        key: value
        for key, value in cast(dict[object, object], data).items()
        if isinstance(key, str) and isinstance(value, str)
    }


def get(provider: str) -> str | None:
    """Look up the last model_id used for ``provider``.

    Args:
      provider: Provider class name (e.g. ``"OpenAISubscription"``).

    Returns:
      model_id: Persisted model_id, or ``None`` when unseen.

    """
    return load().get(provider)


def record(provider: str, model_id: str) -> None:
    """Persist that ``provider`` was last used with ``model_id``.

    No-op when the entry already matches (skips the disk write).

    Args:
      provider: Provider class name.
      model_id: The model_id to associate.

    """
    if not provider or not model_id:
        return
    current = load()
    if current.get(provider) == model_id:
        return
    current[provider] = model_id
    try:
        atomic_write_bytes(_PATH, json.dumps(current, indent=2).encode("utf-8"))
    except OSError:
        logger.warning("Could not persist last-models to %s", _PATH, exc_info=True)
