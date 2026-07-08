"""Arena: run a team of autonomous agents on one maze arm (mesh or tree).

Owns the concurrency the tick-free design needs: each agent is its own
``agent.run()`` task; spawning launches another task into the live team; and a single
**shutdown barrier** stops everyone cleanly the instant the maze is solved (or the
interaction budget is spent), freezing the event log so nothing lands after the win.

mesh  : any agent may message anyone, broadcast, and spawn  -> a recursive team.
tree  : workers reach only the coordinator (it relays); only the coordinator spawns
        -> a flat star, every cross-agent fact double-hops through the hub.
told   : the system prompt states the topology.   discover : it doesn't (illegal sends
        are dropped; the agent must infer the structure).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import asyncio
import contextlib

from examples.agent_maze.engine import Engine
from examples.agent_maze.tools import CommsTool, SpawnTool, WorldTool
from examples.agent_maze.world import SpawnMeta
from sagent.agent import Agent
from sagent.tools.core import agent_label_var
from sagent.types.model import Model
from sagent.types.runtime import UserMessage


SEED = "a0"


def _system(label: str, role: str, *, mesh: bool, told: bool, coordinator: str) -> str:
    core = (
        "You are in a FOGGY maze. A LOCK has TWO same-letter plates far apart (out of "
        "each other's sight). It opens ONLY when two DIFFERENT agents stand on the two "
        "plates and BOTH press, EACH naming the other, within a short window — a lone or "
        "mistimed or mis-named press just wastes one of your few charges. Tools: `world` "
        "(look / move x y / press partner), `comms` (talk)"
    )
    can_spawn = mesh or label == coordinator
    core += ", `spawn` (x y: add a helper).\n" if can_spawn else ".\n"
    if role == "seed":
        core += (
            "You are the FIRST agent, ALONE — you can open nothing by yourself. Your very "
            "first job is to SPAWN several helpers on empty tiles next to you, then spread "
            "the team out to find plates and pair up. ASSIGN pairs explicitly (e.g. 'a1 & "
            "a2 take lock a; a3 & a4 take lock b') so exactly two agents converge on each "
            "lock. "
        )
    else:
        core += (
            "You were just spawned to help. Find a plate, announce which plate/lock you are "
            "on, and pair with whoever holds the matching plate. "
        )
    if told and mesh:
        core += "You may message ANYONE, broadcast discoveries, and spawn helpers.\n"
    elif told and label == coordinator:
        core += (
            f"You are the COORDINATOR '{coordinator}': workers report only to you and you "
            "relay; only you may spawn. Direct the team.\n"
        )
    elif told:
        core += (
            f"Message ONLY the coordinator '{coordinator}' (it relays); you cannot spawn "
            "or reach peers directly.\n"
        )
    else:
        core += "Work out who you can reach and whether you can spawn.\n"
    core += (
        "Be TERSE: a couple of short messages to coordinate, then ACT (move/press) — "
        "don't narrate or chat. Keep acting until ALL locks are open; do not stop early "
        "while any lock remains."
    )
    return core


class Arena:
    """One arm of the demo: a team of autonomous agents racing to open every lock."""

    def __init__(
        self,
        rows: list[str],
        meta: SpawnMeta,
        make_model: Callable[[], Model],
        *,
        mesh: bool = True,
        told: bool = True,
        model_id: str = "",
        max_agents: int = 8,
        rounds: int = 28,
        budget_t: int = 140,
    ) -> None:
        self.engine = Engine(rows, model=model_id)
        self.meta = meta
        # A FRESH model (its own provider/SDK) per agent: Agent.shutdown() closes the
        # agent's own model, so one agent finishing must not tear down a sibling's SDK.
        self.make_model = make_model
        self.mesh = mesh
        self.told = told
        self.rounds = rounds
        self.budget_t = budget_t
        self.world = WorldTool(self.engine)
        self.comms = CommsTool(self.engine, mesh=mesh, coordinator=SEED)
        self.spawn = SpawnTool(
            self.engine,
            self.spawn_child,
            mesh=mesh,
            coordinator=SEED,
            max_agents=max_agents,
        )
        self.agents: dict[str, Agent] = {}
        self.tasks: dict[str, asyncio.Task[None]] = {}
        self._counter = 0

    def _tools_for(self, label: str) -> list[Any]:
        tools: list[Any] = [self.world, self.comms]
        if self.mesh or label == SEED:
            tools.append(self.spawn)
        return tools

    def _launch(self, label: str, role: str) -> None:
        agent = Agent(
            model=self.make_model(),
            system=_system(
                label, role, mesh=self.mesh, told=self.told, coordinator=SEED
            ),
            tools=self._tools_for(label),
            max_tool_call_rounds=self.rounds,
        )
        self.agents[label] = agent
        wake = (
            "You wake, alone." if role == "seed" else "You were just spawned — act now."
        )

        async def drive() -> None:
            agent_label_var.set(label)
            gen = agent.run(UserMessage(text=self.engine.feedback(label, wake)))
            try:
                async for _ev in gen:
                    if self.engine.all_locks_open() or self.engine.t >= self.budget_t:
                        break
            except asyncio.CancelledError:
                pass
            finally:
                with contextlib.suppress(Exception):
                    await gen.aclose()

        self.tasks[label] = asyncio.create_task(drive())

    def spawn_child(self, parent: str, xy: tuple[int, int]) -> str:
        """Embody + launch a new helper task (called by the SpawnTool, under the lock)."""
        self._counter += 1
        label = f"a{self._counter}"
        self.engine.add_agent(label, xy, parent=parent)
        self._launch(label, "helper")
        return label

    async def run(self, *, wall_s: float = 300.0) -> Engine:
        loop = asyncio.get_event_loop()
        self.engine.add_agent(SEED, self.meta["seed_spawn"])
        self._launch(SEED, "seed")
        deadline = loop.time() + wall_s
        # poll external solve/budget/task state (not a single event), then barrier-stop
        while (  # noqa: ASYNC110
            not self.engine.all_locks_open()
            and self.engine.t < self.budget_t
            and not all(t.done() for t in self.tasks.values())
            and loop.time() < deadline
        ):
            await asyncio.sleep(0.15)
        await self._shutdown()
        return self.engine

    async def _shutdown(self) -> None:
        """Quiesce every agent. The engine froze on solve (no post-win events), so this
        just stops the tasks; each agent closes its OWN model. Re-gather in a loop because
        a task mid-spawn can create a new drive task after the first cancel sweep.
        """
        for _ in range(5):
            tasks = list(self.tasks.values())
            for agent in self.agents.values():
                with contextlib.suppress(Exception):
                    agent.shutdown()  # sync: closes the agent's own model + bg jobs
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if all(t.done() for t in self.tasks.values()):
                break
