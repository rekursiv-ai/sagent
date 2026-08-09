"""Cross-session memory of the last model_id used per provider.

Stored at ``data_dir("rekursiv-ai")/sagent/last-models.json`` as a flat
``{provider_class_name: model_id}`` map. Updated on every
``Agent.swap_model`` (and at initial agent construction when a
``ModelRecipe`` is supplied). Looked up by the ``/model`` slash
command when the user changes provider without naming a model -- so
``/model provider=OpenAISubscription`` resumes the last OpenAI model
the user picked, falling back to ``OpenAISubscription.DEFAULT_MODEL``
when this provider hasn't been used before.

Concurrent writers (multiple agent processes / threads) are serialised
through an ``fcntl.flock`` on a sibling lock file so the
load->modify->write sequence is atomic across both. Without the lock,
two writes racing on different provider keys would clobber each other.
"""

from __future__ import annotations

from typing import cast

import fcntl
import json
import logging
import os

from sagent.lib.atomic_file import atomic_write_bytes
from sagent.lib.userdirs import data_dir


logger = logging.getLogger(__name__)


def load() -> dict[str, str]:
    """Return the persisted ``provider → model_id`` map.

    Returns:
      mapping: Empty dict when the file is missing, malformed, or the
          top-level value isn't a JSON object.

    """
    try:
        raw = (data_dir("rekursiv-ai") / "sagent" / "last-models.json").read_text(
            encoding="utf-8"
        )
    except (FileNotFoundError, OSError):
        return {}
    try:
        data: object = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(
            "Corrupt %s; treating as empty.",
            data_dir("rekursiv-ai") / "sagent" / "last-models.json",
        )
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
    Persistence is best-effort: any ``OSError`` raised while creating
    the sagent data directory, acquiring the lock file, or writing
    the data is logged at WARNING and swallowed -- callers never crash
    on a locked-down home directory.

    Args:
      provider: Provider class name.
      model_id: The model_id to associate.

    Raises:
      ValueError: When ``provider`` or ``model_id`` is empty.

    """
    if not provider or not model_id:
        raise ValueError(
            "last_models.record requires non-empty provider and model_id; "
            f"got provider={provider!r}, model_id={model_id!r}."
        )
    try:
        (data_dir("rekursiv-ai") / "sagent" / "last-models.json").parent.mkdir(
            parents=True, exist_ok=True
        )
        lock_path = (
            data_dir("rekursiv-ai") / "sagent" / "last-models.json"
        ).with_suffix(
            (data_dir("rekursiv-ai") / "sagent" / "last-models.json").suffix + ".lock"
        )
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            current = load()
            if current.get(provider) == model_id:
                return
            current[provider] = model_id
            atomic_write_bytes(
                data_dir("rekursiv-ai") / "sagent" / "last-models.json",
                json.dumps(current, indent=2).encode("utf-8"),
            )
        finally:
            os.close(fd)
    except OSError:
        logger.warning(
            "Could not persist last-models to %s",
            data_dir("rekursiv-ai") / "sagent" / "last-models.json",
            exc_info=True,
        )
