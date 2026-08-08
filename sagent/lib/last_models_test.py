"""Tests for ``lib.last_models``: cross-session per-provider model memory."""

from __future__ import annotations

import json
import threading

import pytest

from sagent.lib import last_models
from sagent.lib.userdirs import data_dir


def test_record_then_get_roundtrip() -> None:
    last_models.record("OpenAISubscription", "gpt-x")
    assert last_models.get("OpenAISubscription") == "gpt-x"


def test_load_missing_file_returns_empty() -> None:
    assert last_models.load() == {}


def test_record_concurrent_writes_preserve_all_keys() -> None:
    """Concurrent writes from many threads must not lose other-key entries."""
    providers = [f"P{i}" for i in range(20)]

    def worker(name: str) -> None:
        last_models.record(name, f"model-{name}")

    threads = [threading.Thread(target=worker, args=(p,)) for p in providers]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    final = last_models.load()
    # Without inter-process/thread locking, load->modify->write races can
    # clobber prior records. A correct implementation either serialises
    # writes (flock) or retries until all 20 keys land.
    missing = [p for p in providers if final.get(p) != f"model-{p}"]
    assert not missing, f"lost records under concurrent writes: {missing}"


def test_record_skips_when_value_unchanged() -> None:
    path = data_dir("rekursiv-ai") / "sagent" / "last-models.json"
    last_models.record("Prov", "m1")
    mtime_before = path.stat().st_mtime_ns
    last_models.record("Prov", "m1")
    assert path.stat().st_mtime_ns == mtime_before


def test_load_ignores_non_string_entries() -> None:
    path = data_dir("rekursiv-ai") / "sagent" / "last-models.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"ok": "v", "bad_int": 1, "bad_list": ["x"]}))
    loaded = last_models.load()
    assert loaded == {"ok": "v"}


def test_record_empty_provider_raises() -> None:
    with pytest.raises(ValueError, match="non-empty provider"):
        last_models.record("", "m")


def test_record_empty_model_id_raises() -> None:
    with pytest.raises(ValueError, match="non-empty provider"):
        last_models.record("Prov", "")


def test_record_swallows_locked_down_sagent_dir(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``record`` is best-effort: a non-writable ``~/.sagent`` must not crash."""
    ro_root = data_dir("rekursiv-ai") / "sagent"
    ro_root.mkdir(parents=True, exist_ok=True)
    ro_root.chmod(0o500)
    try:
        last_models.record("Prov", "m1")
    finally:
        ro_root.chmod(0o700)
    assert "Could not persist last-models" in caplog.text


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
