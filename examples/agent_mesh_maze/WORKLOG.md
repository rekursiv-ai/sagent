# WORKLOG — agent-mesh maze demo (sagent multi-directional orchestration)

**One line:** A 2D escape-maze where the same task, given the same prompt, is solved by
two agent *topologies* — a **hub-and-spoke tree** (peers route every message through a
coordinator) vs a **peer mesh** (any agent talks to any agent, broadcasts, interrupts).
The mesh wins, visibly, and you watch *why* — the coordinator node drowns in arrows while
the mesh distributes. Demo 2 of the sagent "unique strength" series.

> Demo 1 (`examples/self_escalating_solver`) showed **vertical** self-mutation: one agent
> changing *itself* (`AgentSelf`, cross-vendor). Demo 2 shows **horizontal** orchestration:
> an agent deciding to *become a team* — `AgentSpawn` + `AgentSend` + interrupt — where the
> payoff is **multi-directional conversation**, not a parent-child tree.

---

## 0. The thesis (what this demo proves)

Most agent frameworks — and the default mental model — are a **tree**: a supervisor/parent
spawns workers, workers return results to the parent, the parent aggregates. Hub-and-spoke.
sagent's pitch is the **mesh**: peers spawn, message any peer directly, broadcast, and
preempt each other. The demo is a maze engineered so that:

- a **static** decomposition favours neither topology, but
- **constant mid-run churn** (fog, dead-end walls, roving hazards, the exit appearing,
  who-holds-what) punishes the tree's round-trips and rewards the mesh's direct talk.

The honest claim is **not** "mesh beats tree everywhere" — a tree is fine for static,
cleanly-decomposable work. The claim is "**when the situation keeps changing, a mesh that
re-coordinates directly beats a tree that funnels every update through a hub**," and the
maze is built to live in that regime. (Honesty guardrail carried over from demo 1's cost
caveat — see §8.)

The win is **quantitative**: ticks-to-solve, messages sent, wasted effort, budget spent,
agents extracted. Not just success/fail.

---

## 0.5 ADVERSARIAL REVIEW — what is *actually* sagent-unique? (2026-06-25)

We stress-tested the thesis against the competitive landscape (web-researched: OpenAI
Agents SDK / Swarm, Microsoft AutoGen v0.4 / AG2, LangGraph, CrewAI, plain code). The
honest verdict reshaped the whole demo:

**No single ingredient is unique to sagent — they are loosely sprinkled across other
frameworks:**
- **Recursive runtime spawn:** near-impossible in OpenAI SDK / LangGraph / CrewAI (static,
  pre-declared topology); moderate in AutoGen (runtime factories) — but **no framework
  except sagent has a depth-*budget* concept**. A budgeted recursive spawn exists in the
  wild only in OpenAI *Codex* (a product, not a library) and sagent.
- **Any-to-any + broadcast:** AutoGen v0.4 core genuinely HAS it (`send_message` +
  `publish_message` pub/sub, no supervisor). So **this is NOT sagent-only — do not claim it
  is** (a knowledgeable reviewer rebuts instantly the moment you say it).
- **Context inheritance on spawn:** sagent is *weaker* — its child boots empty with a
  parent-written prompt; OpenAI's handoff shares full history *by default*. **Do NOT claim
  this as a sagent advantage; it is backwards.**
- **"Spawn at the fork / no travel":** a property of *our maze world*, not any framework
  (none has a spatial concept). **STRAWMAN — cut it.**

**The one genuinely hard-to-replicate capability:** mid-execution **preemption with a
DETACHED background continuation + result splice-back**. Every mainstream framework cancels
(AutoGen `future.cancel()` → lost work), awaits, or suspend-and-replays (LangGraph re-runs
the node from the top). sagent interrupts the busy agent, lets its in-flight tool keep
running detached, and splices the result back. **No equivalent anywhere; hard even in
hand-rolled code.** Lead with this.

**The honest differentiator = the SEAMLESS COMBINATION.** sagent is the only stack where
recursive-*budgeted*-spawn + any-to-any + broadcast + **detached-preemption** coexist as
native, composable primitives in one model. Elsewhere you bolt AutoGen-core pub/sub onto a
roll-your-own budgeted-recursive-spawn onto a hand-rolled detach-splice protocol — three
subsystems from three places that do not compose. The demo's job is to make that seamless
composition VISIBLE, and to lead with **detached-preemption**, NOT "decentralization beats
hubs" (mostly ergonomics — AutoGen core is decentralized too).

- **DO claim:** detached-preemption (the capability moat); budgeted recursive spawn
  (first-class vs roll-your-own); the seamless one-model composition of all four.
- **Do NOT claim:** spawn-at-location, context-inheritance, "only sagent does any-to-any".

Sources: OpenAI handoffs/multi_agent docs + issue #329 (mid-run interrupt → wontfix); Codex
subagents; AutoGen messaging + tools/cancellation; LangGraph interrupts/Send; langgraph-swarm;
CrewAI source. (Full review in the session transcript.)

## 0.6 REDESIGN — the puzzle that needs all four AT ONCE (lead with the moat)

Scenario: a **BUDGETED PARALLEL HYPOTHESIS SEARCH** (the scientific-discovery parallel from
the design notes below).

- A forking tree of branches (hypotheses); **most are dead-ends (failure modes)**; one deep
  leaf holds the discovery. Probing a branch is a LONG, multi-step action ( = compute ).
- A finite **COMPUTE BUDGET** (spawns + probe-steps). The run starts from ONE lead.
- Winning within budget requires all four, seamlessly:
  1. **Recursive budgeted spawn** — fan out across the tree; a serial search blows the
     budget. `max_depth` IS the compute allocation.
  2. **Broadcast** — the instant a probe finds the discovery OR proves a subtree dead, it
     tells EVERYONE, so nobody burns budget on a now-irrelevant branch.
  3. **Detached preemption** — peers are mid-probe (long actions) when that broadcast lands;
     interrupt them to stop wasting budget, but their in-flight probe's partial findings
     ("this sub-branch is also dead") are PRESERVED (detach + splice back) and feed the
     pruning — not thrown away. ← the moat, made visible.
  4. (any-to-any messaging underlies 2+3 — framed as *ergonomics*, not a unique claim.)
- **Seamlessness is the point:** a few simple tools, and the swarm behaviour emerges.
- **Honest contrast (no strawman):** sagent's detach-splice vs the *universal*
  cancel-on-interrupt (faithfully what AutoGen/LangGraph actually do) → quantify the compute
  WASTED re-doing cancelled probes. Or skip the head-to-head and SHOWCASE the sagent swarm
  plus an honest "to assemble this elsewhere you'd need …" panel.
- **Closing statement** (AI-for-science) carries over from the design notes below.

> NOTE: this supersedes the find-the-diamond demo (committed, phases 1-5) as the headline.
> The world engine / Sim / comms / webpage substrate is reused; the *task* and the *claim*
> change to the budgeted hypothesis search led by detached-preemption.

## 0.7 No-LLM validation — `simulation_optimal_baseline.py` (2026-06-25)

Before spending a token, we validated the breakable-wall escape maze with TWO faithful
pure-Python strategies (no LLM) — a centralized TREE (only the lead spawns, at the hub →
helpers ferry; serial) and a decentralized MESH (any agent spawns in place, parallel) —
and scored them on the same hub maze (4 arms, thick breakable walls, the exit hidden
behind one, all agents must escape; collaborative breaking: k agents on a wall = −k/turn).

- **Result: the maze is structurally sound.** MESH wins every exit position: **21 turns /
  187 compute vs TREE 26.2 / 249 — ~1.25× faster, 1.33× cheaper.** Never loses. (A real
  `_adj_cell` bug that made S/E walls unreachable was found + fixed during this — exactly
  what the validation is for.)
- **Honest nuance — the win is from SPAWN STRUCTURE, not communication.** Adding fog +
  comms-lag (mesh broadcast = 1-turn, tree relay-via-lead = 2-turn) did **not** widen the
  gap. Why: the two solvers coordinate via a *shared algorithm* (round-robin "agent i takes
  wall i%4"), which needs zero communication, and the optimal play is "break all walls in
  parallel" — there is no redundant work to avoid by talking. So the sim measures each
  paradigm's **efficiency CEILING**, and that ceiling already favors decentralized ~1.25×.
- **Where communication actually lives: the ACHIEVABILITY GAP = the LLM test.** Real agents
  don't share a round-robin rule; they coordinate by talking. So the mesh team (broadcast)
  can *reach* its ceiling, while the tree team (everything via the lead, 2-hop) falls
  *short* of its ceiling — it can't divide work / prune dead-ends fast enough. **The comms
  advantage IS that shortfall**, and it only shows once real LLM agents play.
- → `simulation_optimal_baseline.py` therefore serves double duty: (1) proof the maze is
  fair + structurally biased to decentralized, and (2) the **optimal-baseline benchmark**
  the LLM arms are scored against — mesh should reach ~ceiling, tree should fall short, and
  if an arm underperforms its ceiling we read off *why* (redundant exploration? hub stall?)
  and tune that arm's prompt.

## 1. Capability confirmation — sagent CAN do all of this (spike-verified 2026-06-24)

The whole demo rides real, first-class sagent tools (like demo 1 rode `AgentSelf`). Built-in
tools live in `sagent/tools/`: `agent_spawn.py`, `agent_send.py`, `background_task.py`.

| What we need | sagent primitive | Status |
|---|---|---|
| Spawn → **tree** | `AgentSpawn` (non-persistent): spawn child → run to completion → **return output to parent**; children never see each other | ✅ |
| Spawn → **mesh** | `AgentSpawn(persistent=true)`: runs child via `serve_forever()`, returns a **label** immediately; addressable concurrent peer | ✅ live |
| **Any-to-any** messaging | `AgentSend(to=<label>, content)`: routes via the **global `agent_registry`** — any agent → any agent. Each agent's prompt is auto-injected with the live roster ("agents you can message: …"). Lands in target's **inbox**. | ✅ live |
| **Interrupt / preempt** | the `AgentSendMessage` `AgentSend` posts is **preempting**: it interrupts the target's current cohort and *detaches* in-flight tools (they keep running) so it reacts now. (Also Queued / Deferred variants.) | ⚠️ code-confirmed; live test deferred to build |
| **Spawn cost / budget** | `max_depth` ("Spawn budget: depth X/Y — N generations available") + auto-bundled `BackgroundTask` (list/cancel children) | ✅ |
| **Report-back** | `notify_on_asleep`: parent inbox auto-gets "[child is idle] \<last text\>" when a child goes quiet | ✅ live |

### The spike (de-risk, haiku, 2 runs) — `scratchpad/spike_mesh.py`

A lead spawned two **persistent** peers; **bob** queried **alice** peer-to-peer, alice
replied peer-to-peer, bob reported. Confirmed live: persistent spawn, **sibling-to-sibling**
`AgentSend` (NOT routed through the parent), roster discovery, inbox delivery, idle-notify.

**THE GOTCHA (this is why we spiked):** only **persistent** agents register in
`agent_registry`. The non-persistent root ("lead") that does the spawning is **not
addressable** — `bob → AgentSend(to='lead')` failed: *"Unknown agent: 'lead'. Active:
['alice','bob']"*. The parent still hears children via the idle-notify forwarder, but an
explicit child→coordinator `AgentSend` needs the coordinator **registered**.

→ **Design consequence:** every mesh participant must be a **persistent** agent. The
"coordinator" is just another registered persistent peer (or register it explicitly). A thin
non-persistent bootstrap spawns everyone and steps back.

### Two known gaps / nuances

- **No native one-to-all broadcast.** `AgentSend` is point-to-point. "Broadcast" = fan-out N
  sends (honest: shows the mesh isn't free, and the count is a visible metric) **or** a
  ~15-line custom `Broadcast` tool that pushes to every registry inbox. **OPEN DECISION (§9).**
- **Preempt not yet live-tested.** Code says `AgentSendMessage` preempts + detaches in-flight
  tools; confirm during the build when agents have long "move" actions to interrupt.

### Cost (spike-measured)

**~$0.0023 per agent decision on `claude-haiku-4-5`** (stable across 2 runs; $0.016/run for
3 agents). Projected maze run ≈ 5 agents × ~12 decisions ≈ **$0.12–0.17**; two arms ≈
**$0.25–0.35/run**; a handful of capture runs ≈ ~$1. Sonnet ≈ 5× — reserve for if haiku
can't coordinate the *harder* maze. **Replay (default mode) is free.** Use haiku for the bulk;
escalate to sonnet only on a measured coordination-quality limitation.

---

## 2. World model — tick-based, deterministic, replayable, animatable

Chosen for the same reasons demo 1 shipped a replay: deterministic, cheap, captureable, and
trivially animatable. (Free-running fully-async agents are closer to sagent's "real" channel
model but flaky and expensive to capture — rejected for the demo surface.)

- **A world clock.** Each tick: every agent reads its inbox → thinks → emits an **intent**;
  the world resolves moves + delivers messages, then advances.
- **Macro-intents, not tile-by-tile.** Agents issue high-level intents ("go to plate B",
  "press", "dig wall at (4,7)", "broadcast: exit at (7,3)"); the **world** pathfinds/animates
  the steps. LLM calls go to *coordination*, not A\* — far cheaper, and the comms stay the
  star. (The animation still renders tile-by-tile movement.)
- **Fog of war.** Each agent sees only nearby tiles + its own inventory. No single agent —
  coordinator included — has the whole map. This is what turns "nice-to-have comms" into
  "you cannot solve this without talking," and makes the *multi-directional* part real.
- **Capture → `web/data.js`** (per-tick positions, messages, events, inventories) → animated
  webpage. Replay by default, `--live` to re-capture (the demo-1 pattern).

---

## 3. The two arms — same prompt, same tools, different comms POLICY

Both arms use **persistent** agents (concurrent movers). The difference is the messaging
policy — a legitimate sagent topology each, NOT a strawman:

- **Tree / hub-and-spoke (baseline).** Persistent peers are told (prompt + restricted roster,
  or a send tool that only accepts `to=<coordinator>`) to message **only the coordinator**,
  which relays. Real coordination — but every cross-agent update **double-hops** through the
  hub, the hub **serializes**, and the hub's context **bloats** with everyone's chatter. This
  is the "parent-child, inefficient" baseline the user specified: *difficult, not impossible*.
- **Mesh.** Persistent peers + full roster → **any-to-any** `AgentSend` + **broadcast** +
  **preempt**. One hop.

Same task prompt for both (like demo 1's identical prompt across arms). Only the comms policy
(which labels each agent may message / whether broadcast is offered) differs.

---

## 4. The mechanics — each isolates a DIFFERENT multi-directional primitive

Not redundant: broadcast-query / preempt-sync / broadcast / peer-gossip are four faces of
multi-directional comms, each with its own visible **tree tax**.

| Mechanic | Mesh primitive it showcases | The tree tax |
|---|---|---|
| **Distributed inventory — "who's holding the diamond?"** (agents pick up keys/diamonds/junk while exploring; the vault needs a specific item; some held items are useless) | **broadcast-query + peer handoff** | tree: the hub must pull *everyone's* inventory incl. the junk and sift centrally → context bloat + serialization. mesh: one query pulls only the holder → direct handoff |
| **N plates held down at once** (vault opens only while all N plates are simultaneously held; one body can't) | **preempt-sync + interrupt** | re-sync after a hazard = peer→hub→all peers (double-hop) every time; mesh aborts/re-presses in one preempt-broadcast |
| **All converge to one exit (early-halt)** (everyone must reach the exit; whoever finds it first should stop the others searching) | **broadcast (one-to-all)** | finder→hub→all (2 hops, serialized) vs finder→all in one hop; early-stop value is maximized by the direct broadcast |
| **Diggable walls, wasted effort** (some walls dig through faster with more diggers, but some are solid rock / dead ends → wasted effort) | **peer gossip (many-to-many)** | siblings are blind to each other through the hub → two agents dig the same dead wall; mesh: "solid rock here, skip / broke through, come" |

The "could be useless" wrinkle (junk inventory, dead-end walls) is the key: it's what makes
**broadcast-pull** (ask only when you need it) beat **tree-push** (everyone reports everything
to the hub). Fog-of-knowledge: the roster tells agents *who exists*, never *who has what*.

---

## 5. The economy — spawn cost + extract-everyone barrier

Requirement: the lead should want to spawn *enough but not too many*.

- **Spawn cost.** `max_depth` is a native budget ("N generations of sub-spawning available");
  plus a shared **oxygen/turn budget** drawn down by each spawn and each move.
- **Barrier to complete.** Win = vault open **AND every living agent back at the exit** before
  the budget runs out. Over-spawn → too many bodies to extract in time. Under-spawn → can't
  cover the plates / dig fast enough. Optimum ≈ N (the plate count). Every body the lead
  creates is another body it must get out alive.

This puts the lead's "spawn judgement" on screen, in *both* arms.

---

## 6. The hero visual — a side-by-side race where the topology IS the picture

Two identical mazes, same prompt, animated in lockstep by tick:

- **left = tree:** every message arrow funnels to/from the one coordinator node — a **star**
  that visibly clogs.
- **right = mesh:** arrows fly peer-to-peer; broadcasts ripple outward — a **web**.
- **Live counters:** ticks elapsed, messages sent, wasted digs, items handed off, agents
  extracted, budget left. Mesh finishes first / extracts everyone; the tree double-hops itself
  toward the time budget.

You don't *explain* the bottleneck — you *see* the coordinator drowning in arrows.

---

## 7. Cost budget (from §1)

| | haiku | sonnet |
|---|---|---|
| per agent decision | ~$0.0023 | ~5× |
| one maze run (≈5 agents × ~12 decisions) | ~$0.12–0.17 | ~$0.6–0.9 |
| both arms / run | ~$0.25–0.35 | ~$1.2–1.8 |

Default to haiku; escalate to sonnet only on a *measured* coordination-quality limit. Replay
is free and keyless.

---

## 8. Honesty guardrails (carried from demo 1)

- **Don't overclaim.** Say plainly the maze is built to stress the regime where mesh wins
  (high-frequency re-coordination); a tree is fine for static decompositions.
- **Cherry-pick discipline.** If the shipped `data.js` is a cherry-picked clean run, **say so**
  in the README/UI (demo 1 precedent). Live re-runs are stochastic.
- **Honest metrics.** Report ticks/messages/budget as measured; if the tree happens to win a
  run, don't hide it — note the variance.
- **Real topologies, not a strawman.** Both arms are legitimate sagent patterns; the tree is
  *inefficient under churn*, not crippled.

---

## 9. OPEN DECISIONS (to settle before/early in the build)

1. **Broadcast: fan-out N `AgentSend`s vs a custom `Broadcast` tool.** Lean **fan-out** —
   keeps us on pure stock sagent, the send-count becomes a visible metric, and the animation
   gets real per-edge arrows. (Custom tool = cleaner one call, but hides the cost.)
2. **v1 spine — which mechanic first.** Lean **distributed-inventory ("who has the diamond?")
   + exit-convergence** (both pure broadcast, cleanest tree-tax, instantly readable), then
   layer **N-plates** (the "physically impossible alone" showpiece) and **digging**.
3. **Maze size / agent count.** Pick the smallest that forces ~N=3–4 plates + a few dead-end
   walls and still reads on screen. Drives cost + capture time.
4. **Coordinator in the mesh arm.** Is there a privileged coordinator at all, or a flat
   peer-elected lead? (Flat is the purer mesh; a named lead is easier to animate.)
5. **Live preempt confirmation** — first build-phase de-risk (agent mid-"move" gets an urgent
   peer message; confirm detach + react).

---

### Design notes from review (incorporate)

- **Fairness — NOT turn-capped (avoid the rigged setup).** Agents are async (no
  "1 message per turn" cap), so broadcast isn't a free win. A `broadcast` is N
  fan-out `say`s — counted as N real messages (the `CommsTool` already counts them).
  The tree relays the same fact for ~N messages too (worker→hub + hub→each peer). So
  the mesh edge is NOT message count; it's **latency** (1 hop vs 2), **hub
  serialization** (one LLM must absorb every report and emit every relay), and **hub
  context-bloat** (the coordinator drowns in chatter and slows/dumbs under churn while
  workers stay focused). Win metric = ticks-to-solve + coordination quality, shown
  alongside hops and the hub's growing context. Balance knob if the mesh wins *too*
  easily: charge each message against the shared budget.
- **Retain full transcripts.** Capture every agent's COMPLETE history (reasoning text
  + tool calls + received messages) into `data.js`, tied to the tick timeline — so the
  webpage shows per-agent reasoning panels (demo-1 style, collapsible) and the run is
  fully inspectable. The driver dumps `agent.history` per agent; the webpage renders
  "what agent X was thinking when it broadcast Y".

## 10. Build plan (de-risk order — each phase lands runnable + verified)

0. **[DONE]** Confirm primitives (spike) — §1. ✅
1. **World engine** (no agents): grid, fog, plates, diggable walls, inventory, pathfinding,
   tick loop, budget; deterministic; emits a per-tick trace. Unit-tested headless.
2. **Single persistent agent** drives one body via macro-intents through the world (close the
   agent↔world loop; confirm intent parsing + cost/tick). **Live preempt test here.**
3. **Mesh arm**: N persistent peers + any-to-any `AgentSend` (+ broadcast decision) solve a
   minimal level (inventory + exit-convergence). Capture a trace.
4. **Tree arm**: same peers, hub-only comms policy; same level. Capture.
5. **Webpage**: side-by-side replay from `data.js` (topology arrows + counters). Replay-default,
   `--live` to recapture (demo-1 pattern).
6. **Layer mechanics**: plates (sync+interrupt), digging (peer-gossip). Tune the maze so the
   gap is clear + honest.
7. **Polish + cherry-pick + README**, honest framing, lint/ty/codespell green (demo-1 CI gates).

---

## 11. Findings log (chronological)

### 2026-06-24 — primitives confirmed via spike (haiku)
- `AgentSpawn(persistent=true)` + `AgentSend` give a real any-to-any mesh; sibling↔sibling
  messaging works without routing through the parent. Roster auto-injected; inbox delivery;
  idle-notify to parent. haiku drives the tools reliably.
- **Gotcha:** non-persistent root is NOT in `agent_registry` → unaddressable by peers. All mesh
  participants must be persistent. (`bob → AgentSend(to='lead')` → "Unknown agent: 'lead'".)
- Cost ~$0.0023/decision (haiku). Two transient `ReadError`s auto-retried by sagent's retry.
- Artifacts: `scratchpad/spike_mesh.py` (+ `spike_mesh.log`, `spike_mesh2.log`).

### 2026-06-24 — Phase 1 (world engine) + Phase 2 (agent↔world + live preempt) landed
- World engine (`world.py`): deterministic grid / fog / items / dig / plates / BFS /
  budget / per-tick trace. 8 unit tests.
- Sim coordinator (`sim.py`): a LOGICAL clock that ticks only on movement/actions —
  LLM thinking-time makes no dead animation frames and the tick count measures real
  work. `WorldTool` is the shared perceive+act tool; `go_to` awaits the coordinator
  walking the body there, which is what makes a moving agent "busy" → preemptible.
- **Live preempt CONFIRMED (haiku):** one persistent agent mid-traverse to (13,1)
  got a preempting `AgentSendMessage` redirect → the in-flight `go_to` detached, the
  agent re-planned, and it physically walked to the new target (1,9). go_to calls
  `[(13,1),(1,9)]`; final pos == redirect target. ~$0.0035.
- **Gotchas fixed:** (a) custom tools MUST implement the full sagent contract —
  missing `prompt()` → AttributeError in `_build_system`; now covered by an
  offline-Agent contract test. (b) under `serve_forever`, `agent_label_var` ≠ the
  construction-time `default_id`; `WorldTool._aid` now resolves label → default_id →
  sole-agent fallback.
- Artifacts: `scratchpad/spike_p2.py` (+ logs).

### 2026-06-24 — Phases 3-5 landed: comms, both arms, webpage
- `CommsTool` (`say` / `broadcast`, mesh-vs-tree policy, coordinator-aware, QUEUED
  routine delivery + an `urgent` preempt flag). Tree = workers→coordinator→workers
  (the hub relays one `say` at a time, no broadcast — the honest serialization tax).
- `run.py` driver: builds 3 persistent peers per arm, runs the sim, captures trace +
  full per-agent transcripts + metrics to `web/data.js`. `--live` recaptures both arms.
- Mesh SOLVED the open v1 level (40–55 ticks): a3 broadcasts the exit on sight, a1
  finds + carries the diamond, peers dedup "who has it" via broadcast + a direct say.
- Webpage (`web/index.html`): two mazes animated side-by-side; agents move,
  items/exit render, message arrows accumulate; per-agent transcripts collapse open;
  a per-arm **COMM GRAPH** shows the topology — mesh = full triangle (any-to-any),
  tree = Λ-star on the hub a1 with no a2↔a3 base.

### ⚠️ KEY FINDING — the metric contrast is weak (needs churn/scale; design call)
Both arms solved in ~55 ticks (mesh 55 / tree 56; msgs 14 / 12). The single-diamond
find-and-deliver task is too **low-coordination**: with only ~2 discoveries, the
tree's double-hop latency + hub serialization never bite. The **topology** difference
is real and visible (comm graph + data: mesh has all 6 directed pairs, the tree only
hub-touching ones), but a **performance** gap needs churn or scale. Options (a design
call with the user — they flagged "too obvious an advantage" so it must stay fair):
- **N-plate simultaneity** (the original showpiece): the vault opens only with all
  plates pressed at once → real-time sync the hub can't relay fast enough; a hazard
  knocking an agent off a plate forces an instant re-broadcast. Strongest, biggest build.
- **Scale** (5–8 agents): the hub relays to N-1 workers per discovery and absorbs N-1
  reports → serialization + context-bloat bottleneck. The most honest "hub doesn't scale."
- **Tight extraction barrier + harder maze**: all must reach the exit before a tight
  budget; slow propagation in the tree strands an agent. Delicate to tune fairly.
- Honest framing regardless: the claim is the WIRING (any-to-any vs hub-and-spoke),
  which IS shown; the perf gap is regime-dependent (high churn / many agents), not
  universal — same honesty discipline as demo 1's cost caveat.
- Artifacts: `scratchpad/spike_p3.py`, `capture1.log`, `shot_p5.py`.

---

## 12. Environment & how to run

- Branch `demo/agent-mesh-maze` (off demo-1 HEAD; rebase onto sagent main before the PR).
- Keys: `~/.config/sagent/anthropic_api_key` (file-based). haiku default (~$0.07/arm).
- Replay (no key):  `uv run python -m examples.agent_mesh_maze.run`
- Re-capture both arms:  `uv run python -m examples.agent_mesh_maze.run --live`
- Tests:  `uv run pytest examples/agent_mesh_maze/ -q` (15, all green; ruff + ty + codespell clean)

- Branch: `demo/agent-mesh-maze` (off demo-1 HEAD; rebase onto sagent main before the PR so
  demo 2 is independent).
- Keys: `~/.config/sagent/anthropic_api_key` (file-based, never exported to a CLI subscription —
  demo-1 discipline). haiku default.
- Run (planned): `uv run python -m examples.agent_mesh_maze.run` (replay) / `--live` (recapture).
