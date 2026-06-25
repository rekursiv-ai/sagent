# Agent-mesh maze — peer mesh vs hub-and-spoke

A runnable [sagent](https://github.com/rekursiv-ai/sagent) example of the thing sagent
makes easy that most agent stacks don't: **agents that spawn agents and talk in every
direction** — any peer messaging any peer, broadcasting, and interrupting — instead of a
rigid supervisor→worker tree.

Three agents are dropped into a **foggy 2D maze** and must find a diamond and carry it to
the exit. The *same task, same prompt, same tools* is run two ways:

| arm | wiring | how info moves |
|---|---|---|
| **peer mesh** | any agent → any agent + broadcast (`AgentSend`) | a discovery reaches everyone in **one hop** |
| **hub-and-spoke tree** | workers may message only a coordinator, which relays | every cross-worker fact **double-hops** through one agent |

The webpage replays both side by side. The tell is the **comm graph** under each maze:
the mesh draws a **full triangle** (everyone talks to everyone); the tree draws a
**Λ-star** through the hub — there is no worker↔worker edge.

## What's actually unique here (honest)

We stress-tested this against the field (OpenAI Agents SDK / Swarm, AutoGen, LangGraph,
CrewAI). The honest finding: **no single one of these primitives is unique to sagent** —
they're loosely sprinkled across frameworks. AutoGen's core already has any-to-any pub/sub
messaging; runtime spawning exists in a few places; OpenAI's handoffs even share context
*better* than sagent does (sagent's spawned child boots empty by design).

**Two things are genuinely hard to get elsewhere:**
1. **Mid-execution preemption that preserves in-flight work.** Interrupt a busy agent and its
   running tool keeps going *detached*, then splices its result back. Every other framework
   **cancels** the work (AutoGen), **blocks** on it, or **replays it from scratch**
   (LangGraph). No clean equivalent anywhere — hard even in hand-rolled code.
2. **Budgeted recursive spawn as a first-class primitive** (`AgentSpawn` + `max_depth`,
   children inherit the spawn tool). Elsewhere it's roll-your-own.

**The real differentiator is the seamless *combination*:** sagent is the only stack where
recursive-budgeted-spawn + any-to-any + broadcast + detached-preemption are all native and
compose in one model. Everywhere else you'd stitch three subsystems from three places. This
demo is built to make that combination visible. We explicitly do **not** claim
"decentralization is sagent-only" (it isn't) or "spawned agents inherit memory" (sagent is
weaker there).

## Run it

```bash
# Replay the captured run in a local webpage (no API key needed):
uv run python -m examples.agent_mesh_maze.run
#   -> serves http://localhost:8000 (prints an `ssh -L` line for remote viewing)

# Re-run both arms live (needs ~/.config/sagent/anthropic_api_key) and re-capture:
uv run python -m examples.agent_mesh_maze.run --live
```

You'll see two mazes animate in lockstep: agents (coloured dots) explore the fog, the
diamond/junk/exit render, **message arrows accumulate**, and each agent's **full
transcript** (reasoning + tool calls + inbox) collapses open for inspection.

## How it works

- Each maze runner is a **persistent** sagent agent (`AgentSpawn(persistent=true)` shape;
  here built directly and `serve_forever`-ed) with its own inbox, registered by label.
- They perceive + act via a custom **`world`** tool (`look` / `go_to` / `pick` / `drop`),
  and coordinate via a custom **`comms`** tool (`say` / `broadcast`) built on sagent's
  inbox primitive. Routine messages **queue** (read at the recipient's next decision, so
  they don't thrash movement); an `urgent` message **preempts** an in-flight move.
- A `Sim` coordinator owns a **logical clock** that only ticks on movement/actions, so
  LLM thinking-time produces no dead animation frames and the tick count measures real
  work. Default model is `claude-haiku-4-5` (~$0.07/arm).

## Honest notes

- **What's demonstrated is the *wiring*.** The comm graph + the captured data show the
  real difference: the mesh uses all six directed message pairs; the tree only
  hub-touching ones. That any-to-any topology is what sagent uniquely makes easy.
- **The performance gap is regime-dependent — and small on this task.** On the simple
  single-diamond maze both arms finish in ~55 ticks: with only a couple of discoveries to
  share, the tree's double-hop latency barely costs anything. A *measurable* mesh
  advantage needs **churn or scale** — real-time N-plate synchronisation, many agents
  (the hub can't relay to everyone), or a tight time barrier — which is the next iteration
  (see `WORKLOG.md`). We do **not** claim the mesh is universally faster.
- The shipped `web/data.js` is a captured live run; re-runs vary (haiku is stochastic).

## Files

- `world.py` — the deterministic foggy-maze engine (grid, fog, items, dig, plates, budget, trace).
- `sim.py` — the `Sim` logical-clock coordinator + the `world` perceive/act tool.
- `comms.py` — the `comms` tool (`say`/`broadcast`, mesh-vs-tree policy, queued vs urgent).
- `run.py` — the driver: replay server by default, `--live` to re-capture both arms.
- `web/index.html` + `web/data.js` — the self-contained, animated side-by-side replay.
- `WORKLOG.md` — the full lab notebook: the hub-vs-mesh thesis, spike-confirmed sagent
  primitives, every finding, and the open contrast-tuning decision.
