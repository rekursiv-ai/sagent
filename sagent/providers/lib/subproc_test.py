"""Tests for ``providers.lib.subproc``."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import asyncio
import contextlib
import json

import pytest

from sagent.providers.lib.subproc import _Subproc


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_write_and_read_json_round_trip(tmp_path: Path) -> None:
    """Round-trip one NDJSON line through a Python echo subprocess."""
    proc = _Subproc(["python3", "-c", "import sys; sys.stdout.write(sys.stdin.read())"])
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
    proc = _Subproc(["python3", "-c", script])
    await proc.start()
    msg = await proc.read_json_line(skip_non_json=True)
    assert msg == {"id": 1}
    await proc.close()


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_read_json_line_raises_on_malformed_when_strict() -> None:
    """Default (``skip_non_json=False``) raises ``ValueError`` on a non-JSON line."""
    proc = _Subproc(
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
    proc = _Subproc(["python3", "-c", "pass"])
    await proc.start()
    assert await proc.read_json_line() is None
    await proc.close()


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_stderr_tail_captures_diagnostics() -> None:
    """Stderr is drained into the bounded ring buffer for diagnostics."""
    proc = _Subproc(
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
    proc = _Subproc(["python3", "-c", "pass"], tmpdir=owned)
    await proc.start()
    await proc.close()
    assert not owned.exists()


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_close_is_idempotent() -> None:
    """A second call to ``close`` is a no-op."""
    proc = _Subproc(["python3", "-c", "pass"])
    await proc.start()
    await proc.close()
    await proc.close()


@pytest.mark.real_sleep
@pytest.mark.asyncio
async def test_write_after_subprocess_exit_raises_runtime_error() -> None:
    """Writing to a closed-stdin subprocess surfaces a clean ``RuntimeError``."""
    proc = _Subproc(["python3", "-c", "pass"])
    await proc.start()
    # Wait for the child to exit.
    assert proc._proc is not None
    _ = await proc._proc.wait()
    with pytest.raises(RuntimeError, match="subprocess stdin closed"):
        await proc.write_line("x")
    await proc.close()


def test_unused_argument_holders() -> None:
    """Static placeholder to satisfy basedpyright's import-tracking on ``cast``."""
    _ = cast


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
