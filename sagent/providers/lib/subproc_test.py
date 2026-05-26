"""Tests for ``providers.lib.subproc``."""

from __future__ import annotations

from pathlib import Path
from typing import cast, override

import asyncio
import contextlib
import json
import logging

import pytest

from sagent.providers.lib.subproc import (
    Subproc,
    SubprocessTransportError,
)


def test_default_read_idle_timeout_is_one_minute() -> None:
    proc = Subproc(["python3", "-c", "pass"])
    assert proc._read_timeout_sec == 60.0


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_write_and_read_json_round_trip(tmp_path: Path) -> None:
    """Round-trip one NDJSON line through a Python echo subprocess."""
    proc = Subproc(["python3", "-c", "import sys; sys.stdout.write(sys.stdin.read())"])
    await proc.start()
    await proc.write_line(json.dumps({"a": 1}))
    assert proc._proc is not None
    assert proc._proc.stdin is not None
    proc._proc.stdin.close()
    msg = await proc.read_json_line()
    assert msg == {"a": 1}
    await proc.close()
    del tmp_path


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_read_json_line_skips_non_json_when_requested() -> None:
    """``skip_non_json`` discards banner lines until the first ``{``."""
    script = (
        "import sys;"
        "sys.stdout.write('Loaded cached credentials.\\n');"
        "sys.stdout.write('Hook registry initialized\\n');"
        "sys.stdout.write('{\"id\":1}\\n');"
        "sys.stdout.flush()"
    )
    proc = Subproc(["python3", "-c", script])
    await proc.start()
    msg = await proc.read_json_line(skip_non_json=True)
    assert msg == {"id": 1}
    await proc.close()


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_read_json_line_raises_on_malformed_when_strict() -> None:
    """Default (``skip_non_json=False``) raises ``ValueError`` on a non-JSON line."""
    proc = Subproc(
        [
            "python3",
            "-c",
            "import sys; sys.stdout.write('not json\\n'); sys.stdout.flush()",
        ]
    )
    await proc.start()
    with pytest.raises(ValueError, match="non-JSON line"):
        _ = await proc.read_json_line()
    await proc.close()


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_eof_returns_none() -> None:
    """``read_json_line`` returns ``None`` once stdout has closed."""
    proc = Subproc(["python3", "-c", "pass"])
    await proc.start()
    assert await proc.read_json_line() is None
    await proc.close()


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_read_json_line_timeout_is_transport_error() -> None:
    """A silent stdout stall surfaces as a transport failure."""
    proc = Subproc(
        ["python3", "-c", "import time; time.sleep(60)"],
        read_timeout_sec=0.01,
    )
    await proc.start()
    try:
        with pytest.raises(SubprocessTransportError, match="stdout idle timeout"):
            _ = await proc.read_json_line()
    finally:
        await proc.close()


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_stderr_tail_captures_diagnostics() -> None:
    """Stderr is drained into the bounded ring buffer for diagnostics."""
    proc = Subproc(
        ["python3", "-c", "import sys; sys.stderr.write('boom\\n'); sys.stderr.flush()"]
    )
    await proc.start()
    # Wait for the child to exit so the drain task has fully consumed stderr.
    assert proc._proc is not None
    _ = await proc._proc.wait()
    # The drain task scheduled the final readline before EOF; let it run.
    if proc._stderr_task is not None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(asyncio.shield(proc._stderr_task), timeout=1.0)
    assert "boom" in proc.stderr_tail
    await proc.close()


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_close_removes_tmpdir(tmp_path: Path) -> None:
    """``close`` removes the owned tmpdir even if the subprocess already exited."""
    owned = tmp_path / "owned"
    owned.mkdir()
    (owned / "marker").write_text("x", encoding="utf-8")
    proc = Subproc(["python3", "-c", "pass"], tmpdir=owned)
    await proc.start()
    await proc.close()
    assert not owned.exists()


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_close_is_idempotent() -> None:
    """A second call to ``close`` is a no-op."""
    proc = Subproc(["python3", "-c", "pass"])
    await proc.start()
    await proc.close()
    await proc.close()


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_write_after_subprocess_exit_raises_runtime_error() -> None:
    """Writing to a closed-stdin subprocess surfaces a clean ``RuntimeError``."""
    proc = Subproc(["python3", "-c", "pass"])
    await proc.start()
    # Wait for the child to exit.
    assert proc._proc is not None
    _ = await proc._proc.wait()
    with pytest.raises(SubprocessTransportError, match="subprocess stdin closed"):
        await proc.write_line("x")
    await proc.close()


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_start_cancellation_closes_subprocess() -> None:
    """Cancellation during ``start`` tears down the partially-started child."""
    proc = Subproc(["python3", "-c", "import time; time.sleep(60)"])
    task = asyncio.create_task(proc.start())
    await asyncio.sleep(0)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert not proc.is_alive
    await proc.close()


def test_unused_argument_holders() -> None:
    """Static placeholder to satisfy basedpyright's import-tracking on ``cast``."""
    _ = cast


class _BoomDrainSubproc(Subproc):
    """Subproc whose stderr drainer raises on first scheduling."""

    @override
    async def _drain_stderr(self) -> None:
        raise RuntimeError("simulated drainer crash")


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_stderr_drain_failure_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A crash inside ``_drain_stderr`` surfaces at ``ERROR`` immediately.

    The drainer is fire-and-forget: ``close()`` joins it eventually and
    logs at ``debug``, but the failure may be hours old by then. Wire
    a ``log_task_exception`` callback so the failure shows up in logs
    at the moment it happens.
    """
    logger_name = "sagent.providers.lib.subproc"
    proc = _BoomDrainSubproc(["python3", "-c", "import time; time.sleep(60)"])
    with caplog.at_level(logging.DEBUG, logger=logger_name):
        await proc.start()
        # Yield so the drainer task gets scheduled and raises.
        for _ in range(5):
            await asyncio.sleep(0)
            if any(
                r.name == logger_name and r.levelname == "ERROR" for r in caplog.records
            ):
                break
        await proc.close()
    errs = [
        r for r in caplog.records if r.name == logger_name and r.levelname == "ERROR"
    ]
    assert errs, (
        "drainer crash was not surfaced at ERROR; "
        f"records={[(r.levelname, r.getMessage()) for r in caplog.records]!r}"
    )
    assert errs[0].exc_info is not None, "drainer ERROR must carry a traceback"
    assert "stderr" in errs[0].getMessage().lower()


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
