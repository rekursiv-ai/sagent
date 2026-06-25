# Agent-mesh maze — decentralized vs centralized coordination

A runnable [sagent](https://github.com/rekursiv-ai/sagent) example of the thing sagent makes
easy that most agent stacks don't: **agents that spawn agents and talk in every direction**
— recursive spawn + any-to-any messaging + broadcast — instead of a rigid
supervisor→worker tree.

**One agent** is dropped into a foggy 2D maze that it cannot solve alone: there are several
**locks**, and a lock opens only when its **two same-letter plates** (in different corridors,
out of sight of each other) are **pressed on the same turn** by two different agents. So the
first agent must **spawn a team** — it doesn't know how many it needs — explore (some corridors are
dead-ends), pair plates up by talking, and synchronize the presses.

The *same maze, same task* runs two ways; only the **topology** differs:

| arm | spawn | messaging |
|---|---|---|
| **mesh** (decentralized) | **any** agent may spawn (recursive, parallel team growth) | any agent → any agent, plus broadcast |
| **tree** (centralized) | **only the first agent** may spawn (serial, one per turn) | workers may message only the first agent, which relays one at a time |

Each turn an agent does **one** of `MOVE` / `SPAWN` / `PRESS` / `SEND` (messaging costs a
whole turn — comms is a real resource). The webpage replays both arms side by side, with the
synced conversation under each maze and an explainer of *why* the tree chokes.

## What's actually unique here (honest)

We stress-tested this against the field (OpenAI Agents SDK / Swarm, AutoGen, LangGraph,
CrewAI). The honest finding: **no single one of these primitives is unique to sagent** —
they're loosely sprinkled across frameworks. AutoGen's core already has any-to-any pub/sub;
runtime spawning exists in a few places; OpenAI's handoffs even share context *better* than
sagent (a sagent child boots empty by design).

**The real differentiator is the seamless *combination*:** sagent is the stack where
recursive-budgeted-spawn + any-to-any + broadcast + detached mid-execution preemption are all
native and compose in one model — so the *decentralized strategy is actually available* and
easy to wire, instead of being three subsystems stitched from three places. This demo makes
that visible: the mesh's recursive spawn grows a team fast and its broadcasts pair plates in
parallel, while the tree's lone first agent serializes **both** the spawning **and** the relaying and
falls behind. We explicitly do **not** claim "decentralization is sagent-only" (it isn't).

## Why it's a fair contrast (not a strawman)

This was built adversarially, and several earlier designs were thrown out for *being* unfair —
the lab notebook (`WORKLOG.md`) has the receipts:

- **The coordination must genuinely need communication.** Plates are out of sight and presses
  must be simultaneous with limited charges, so you can't camp or brute-force — partners *must*
  talk. (An earlier "stand on the plate" version let agents win without coordinating; cut.)
- **No free spawn edge.** The spawner *chooses* the tile (an invalid choice wastes the turn);
  spawned agents still have to walk. Spawning isn't a teleport.
- **A pure-Python baseline** (no LLM) proves the maze structurally rewards the decentralized
  strategy, and an **abstract LLM test** shows the gap is the topology (one coordinator
  serializing K independent coordinations is `O(K)`; the mesh is `O(1)`), not a fixed handicap.

## Run it

```bash
# Replay the captured run in a local webpage (no API key needed):
uv run python -m examples.agent_maze.run
#   -> serves http://localhost:8001 (prints an `ssh -L` line for remote viewing)

# Re-capture all four conditions (mesh/tree x told/discover) and write data.js:
uv run python -m examples.agent_maze.run --live   # needs ~/.config/sagent/anthropic_api_key
```

You'll see two mazes animate: agents **appear as the first agent spawns them**, spread through the
fog (dead-ends included), **broadcast to find their plate-partner**, and press together — while
the tree's first agent visibly bottlenecks. Each arm has its own scrubber (synced by default; grab
either to control them independently), a terse synced conversation, and `[…]` to expand any
agent's full reasoning.

## told vs discover (a switch on the page)

A capture stores **both** modes, so the page has a told/discover toggle — no re-run needed.

- **told**: each agent's prompt states its topology (the human designer picked the structure).
  The contrast is **speed** — the mesh opens both locks faster than the tree's serial relay.
- **discover**: the topology is hidden; illegal sends are silently dropped and an agent must
  *infer* it ("my peer-messages aren't landing — I must route through the first agent"). The mesh
  adapts; the tree often **fails**, its workers burning turns on dropped peer-messages (the
  recognition tax, visible as red dropped-arrows).

## How it works

- **Lockstep controller** (`lock_lockstep.py`): every agent acts once per tick and all actions
  resolve together, so a simultaneous press is exact and the per-tick trace drives the replay.
  (An autonomous `serve_forever` variant was tried first; its logical clock made "press within a
  window" too unreliable — see `WORKLOG.md`.)
- **World** (`world.py`): a deterministic foggy maze — paired-plate locks, agent-chosen spawn
  with validation, fog-of-war view, per-tick trace. No LLM, no sagent imports.
- Default model is `claude-sonnet-4-6` — haiku can't reliably run the explore → spawn → pair →
  synchronize chain (it makes both partners pile on one plate); set
  `LLM_MAZE_MODEL=claude-haiku-4-5` for cheaper, flakier runs. The shipped `web/data.js` is a
  captured live run, **cherry-picked best-mesh / worst-tree per mode** (same honest discipline as
  demo-1's clean run); re-runs vary, so the picks sell the effect while the worklog records the
  full distribution.

## Files

- `world.py` — the maze engine (grid, fog, paired-plate locks, spawn validation, trace).
- `lock_lockstep.py` — the lockstep controller: spawn-from-one, the 4 actions, topology routing,
  capture.
- `run.py` — the driver: replay server by default, `--live` to re-capture, `--discover` to hide
  the topology.
- `web/index.html` + `web/data.js` — the animated side-by-side replay + the explainer.
- `WORKLOG.md` — the full lab notebook: the thesis, the adversarial reviews, every rejected
  design and why, and the validation data.
