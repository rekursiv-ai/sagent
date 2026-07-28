#!/bin/sh
# ruff: noqa: EXE003, D300, T201 -- Polyglot shell/Python repro; prints diagnostics.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Reproduce Ctrl+D teardown with a serviced child blocked mid-model-call.

The idle-child case cancels clean (proven). The untested trigger: a serviced
child wedged inside a live ``stream`` call when the REPL tears down. A real HTTP
stream unwinds on cancel differently than a StubModel's instant return -- this
repro blocks ``stream`` on an Event to mimic it.

Examples:
  ./repro_ctrld_hang.py

'''
# fmt: on

from __future__ import annotations

from collections.abc import Callable
from typing import override

import asyncio
import time

from sagent import types
from sagent.agent import agent_test
from sagent.agent.agent import Agent
from sagent.agent.state import agent_registry
from sagent.repl.run_repl import _background_tasks_for_repl_cancel
from sagent.tools.agent_spawn import AgentSpawn
from sagent.tools.core import current_agent_var


class BlockingModel(agent_test.StubModel):
    """A model whose ``stream`` blocks forever until cancelled."""

    @override
    async def stream(
        self,
        request: types.model.ModelRequest,
        publish: Callable[[types.runtime.RuntimeEvent], None] | None = None,
    ) -> types.model.ModelResponse:
        del request, publish
        # Block as a live provider stream would while awaiting bytes.
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


async def _main() -> None:
    parent = Agent(model=agent_test.StubModel(), tools=[], name="Agent")
    # parent replies instantly and idles; child blocks mid-stream.
    parent.model = parent.model  # keep parent responsive
    spawn = AgentSpawn()
    parent_drive = asyncio.create_task(parent.serve_forever())
    await asyncio.sleep(0.05)

    # Spawn a serviced child whose model blocks mid-stream.
    token = current_agent_var.set(parent)
    try:
        # Build the child by hand with the blocking model, then hand it in.
        child = Agent(model=BlockingModel(), tools=[], name="worker")
        # Route through the serviced spawn path directly; the repro's whole
        # point is to exercise this internal path with a mid-stream-blocked child.
        res = spawn._spawn_serviced(  # noqa: SLF001 -- repro targets the internal serviced-spawn path
            child, "worker", "do work", notify_on_asleep=True
        )
        print("SPAWN:", res.content.split(".")[0])
    finally:
        current_agent_var.reset(token)
    await asyncio.sleep(0.2)  # let child enter the blocking stream
    print("registry:", sorted(agent_registry))
    child_rt_busy = child.runtime.model_call is not None
    print("child mid-model-call:", child_rt_busy)

    # Mimic run_repl Ctrl+D teardown.
    t0 = time.monotonic()
    parent.shutdown(force=True)
    bg = _background_tasks_for_repl_cancel(parent)
    print("bg tasks REPL cancels (excludes serviced child):", len(bg))
    for t in bg:
        t.cancel()
    if bg:
        await asyncio.gather(*bg, return_exceptions=True)
    try:
        await asyncio.wait_for(parent_drive, timeout=3.0)
        print(f"root teardown COMPLETED in {time.monotonic() - t0:.2f}s")
    except TimeoutError:
        print("HANG: root serve_forever did not return")

    # Now what asyncio.run does at process exit: cancel remaining tasks.
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    print("pending after root teardown:", [t.get_name() for t in pending])
    for t in pending:
        t.print_stack(limit=6)
    t1 = time.monotonic()
    for t in pending:
        t.cancel()
    try:
        await asyncio.wait_for(
            asyncio.gather(*pending, return_exceptions=True), timeout=3.0
        )
        print(f"pending cancelled in {time.monotonic() - t1:.2f}s -> NO HANG")
    except TimeoutError:
        print("HANG: mid-stream serviced child did NOT cancel within 3s")


def main() -> int:
    """Run the teardown repro; return the process exit code."""
    asyncio.run(_main())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
