# Agent maze — decentralized vs centralized coordination

A runnable [sagent](https://github.com/rekursiv-ai/sagent) example of the thing sagent makes
easy that most agent stacks don't: **autonomous agents that spawn agents and talk in every
direction** — recursive spawn + any-to-any messaging + broadcast — instead of a rigid
supervisor→worker tree.

**One agent** is dropped into a foggy 2D maze it cannot solve alone. There are several
**locks**; a lock opens only when its **two same-letter plates** (in different corridors, out
of each other's sight) are pressed by **two different agents who each name the other** as their
partner, within a short window. So the first agent must **spawn a team** — it doesn't know how
many it needs — explore (some corridors are dead-ends), **discover who its partner is and agree
the pairing by talking**, and time the co-press. None of that is possible without communication.

The *same maze, same task* runs two ways; only the **topology** differs:

| arm | spawn | messaging |
|---|---|---|
| **mesh** (decentralized) | **any** agent may spawn — a recursive, parallel team | any agent → any agent, plus broadcast |
| **tree** (centralized) | **only the coordinator** may spawn — a flat star | workers may message only the coordinator, which relays one at a time |

There is **no global turn**. Each agent is a real sagent `Agent` running autonomously and acting
through tools — `world` (`look` / `move` / `press partner`), `comms` (`say` / `broadcast`), and
`spawn`. The **World is a reactive feedback service** on a *logical interaction clock* (so the
press window is latency-independent and the run is reproducible, unlike wall-clock). The webpage
replays both arms from an event stream, side by side, with a coordination-cost metrics strip and
an explainer of *why* the tree chokes.

## What's actually unique here (honest)

We stress-tested this against the field (OpenAI Agents SDK / Swarm, AutoGen, LangGraph, CrewAI).
The honest finding: **no single one of these primitives is unique to sagent** — any-to-any
pub/sub, runtime spawning, and shared context each exist somewhere.

**The real differentiator is the seamless *combination*:** sagent is the stack where a real
reasoning `Agent` can recursively spawn more agents, message any peer or broadcast, and do it all
as native, composable primitives — so the *decentralized strategy is actually available* to build,
instead of three subsystems stitched from three places. This demo makes that visible: the mesh's
recursive spawn grows a team fast and its direct messaging pairs plates in parallel, while the
tree's lone coordinator serializes **both** the spawning **and** the relaying and falls behind —
or, with the topology hidden, never converges at all. We explicitly do **not** claim
"decentralization is sagent-only" (it isn't).

## Why it's a fair contrast (not a strawman)

- **Coordination genuinely needs communication.** Plates are out of sight, so you cannot know
  *which* agent is on the matching plate without asking; `press` must *name* that partner; and the
  arm is live only a short logical window, so you must agree the moment. Standing on a plate, or
  pressing blindly, opens nothing.
- **Broadcast isn't free.** A broadcast fans out to N peers and is counted as N delivered
  messages, so blasting everyone self-penalises on the cost metric.
- **Multiple locks force real assignment** — the team must divide into pairs; one broadcast can't
  trivially partition everyone.
- The full adversarial design history (and the rejected designs) is in `WORKLOG.md`.

## Run it

```bash
# Replay the captured run in a local webpage (no API key needed):
uv run python -m examples.agent_maze.run
#   -> serves http://localhost:8001 (prints an `ssh -L` line for remote viewing)

# Re-capture all four conditions (mesh/tree x told/discover) and write data.js:
uv run python -m examples.agent_maze.run --live --locks 3 --k 2
#   -> needs ~/.config/sagent/anthropic_api_key
```

You'll see two mazes animate from the event stream: agents **appear as they're spawned**, spread
through the fog, exchange messages (green delivered / red dropped) to find their plate-partner, and
press together — while the tree's coordinator visibly bottlenecks. Each arm shows its
**coordination cost per lock opened** with a ✓ solved / ✗ STUCK badge, plus messages, failed
presses, team size, and total interactions.

## told vs discover (a switch on the page)

A capture stores **both** modes, so the page has a told/discover toggle — no re-run needed.

- **told**: each agent's prompt states its topology. The contrast is **cost** — the mesh opens
  every lock at a low cost-per-lock; the tree's serial relay costs far more.
- **discover**: the topology is hidden; illegal sends are silently dropped and an agent must
  *infer* the structure. The mesh adapts; the tree often **fails outright** (STUCK), its workers
  never inferring they must route through the hub.

## How it works

- **`engine.py`** — the reactive engine on a logical interaction clock. Resolves one action at a
  time under an async lock; appends an event log (a static scene header + per-event
  `{seq,t,agent,kind,…}`) that drives both the metrics and the replay. The one coordination point
  is `press(partner)`: it arms a plate naming a partner for `PRESS_WINDOW` logical interactions and
  latches the lock when both plates are mutually armed by two distinct agents standing on them.
- **`world.py`** — the deterministic foggy maze (grid, BFS navigation, paired-plate locks,
  agent-chosen spawn validation, fog-of-war view). No LLM, no sagent imports.
- **`arena.py`** — runs one arm: each agent is its own `agent.run()` task; spawning launches another
  task into the live team; mesh/tree comms policy + told/discover prompts; and a **shutdown barrier**
  that stops every agent cleanly the instant the maze is solved, so nothing lands after the win.
- **`tools.py`** — the sagent tools: `WorldTool`, `CommsTool` (on the real inbox primitive),
  `SpawnTool`.
- **`capture.py`** — runs the four arms, computes the metrics, and writes the event-stream
  `web/data.js`, keeping best-mesh / worst-tree over `k` runs.
- Default model is `claude-sonnet-4-6`. The shipped `web/data.js` is a captured live run,
  **cherry-picked best-mesh / worst-tree per mode** (the same honest discipline as demo-1); re-runs
  vary, so the picks sell the effect while the worklog records the distribution.

## Files

- `engine.py` — reactive engine: logical clock, event log, the press rendezvous.
- `world.py` — the maze engine (grid, fog, paired-plate locks, spawn validation).
- `arena.py` — one arm: concurrent autonomous agents, mesh/tree topology, shutdown barrier.
- `tools.py` — `WorldTool` / `CommsTool` / `SpawnTool`.
- `capture.py` — runs the four arms, metrics, writes `web/data.js`.
- `run.py` — driver: replay server by default, `--live` to re-capture.
- `web/index.html` + `web/data.js` — the animated event-stream replay + explainer.
- `WORKLOG.md` — the lab notebook: thesis, adversarial reviews, rejected designs, validation data.
