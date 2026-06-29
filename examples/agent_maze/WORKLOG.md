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

### 2026-06-25 — no-LLM validation CORRECTED + first LLM sweep + redesign call
**The "mesh 1.25× faster, maze validated" claim was an artifact — retracted.** It only
holds when BOTH arms are forced to a 12-agent team (mesh's best case, the tree's worst:
serial lead-spawn + herd 12 bodies to the exit). Letting **each paradigm pick its own
turn-optimal team size** collapses it: the tree's optimum is a lean ~6-agent team and the
arms ~tie.

| maze | TREE\* | MESH\* | mesh turns | mesh compute |
|---|---|---|---|---|
| arm4/wall8  | 23.2t | 20.0t | 1.16× | **0.46×** (mesh uses ~2× compute) |
| arm8/wall16 | 38.0t | 34.0t | 1.12× | 0.81× |
| arm10/wall20| 45.5t | 41.0t | 1.11× | 0.93× |

True structural edge ≈ **1.1–1.16× turns at a compute PREMIUM** — a latency/cost tradeoff,
not a free win. Two deeper flaws: (1) `simulation_optimal_baseline` is **not optimal** —
the LLM beat it (lean teams exit faster; "all must exit" punishes over-spawning), so the
name oversells it. (2) The **no-fog sim gives every agent GLOBAL knowledge → it tests spawn
mechanics, NOT communication.** The comms thesis lives only in the fog sim, and the current
fog model barely amplifies (1.16→1.18×: one relay-hop of staleness on one discovery event).

**First LLM sweep (haiku, 4 exits, arm4/wall8, n=1/cell):**
`MESH 18.0t (4/4)` vs `TREE 23.0t (4/4)`. Mesh tight (16–21t); tree volatile (19,19,**35**,19
— a catastrophic blowup at exit=2; at exit=3 tree even beat mesh). Mesh agents reliably
self-organize once prompted spawn-first: **spawn a team → split by corridor → broadcast the
exit on sight → peers redirect next turn → gang-break + converge** (the broadcast→redirect IS
the comms thesis, observed live). Prompt fix that unlocked it: make team-building the explicit
FIRST priority ("you start ALONE; a lone agent is hopeless; SPAWN then SPLIT UP").

#### [LESSON · robustness] centralized = single point of COORDINATION failure
The honest LLM differentiator isn't mean speed — it's **variance**. Mesh is low-variance;
the tree is brittle (the lead juggling reports/relays occasionally breaks down → 35t blowup).
The lead is a single point of *coordination* failure, not just a bandwidth bottleneck.
→ promote to `worklog/lessons/` when the redesign confirms it across seeds.

#### DESIGN CALL (user, 2026-06-25) — back to Python land before more agent tests
1. **Make the sim genuinely optimal** per paradigm (optimize team size + strategy → a real
   frontier, so "baseline" means what it says).
2. **Make COMMUNICATION the objective bottleneck for the tree.** Default to LOCAL knowledge
   (kill the global cheat); model the lead's relay as bounded bandwidth (≤ b msgs/turn) so
   tree info-throughput is O(1) in team size while mesh broadcast is O(W). The maze must make
   the optimal plan depend on DISTRIBUTED local discoveries that must be shared — then the
   tree only ties IF its lead already holds precise optimal-strategy knowledge a priori
   (unrealistic). Hypothesis: widen the problem (more fronts K) → mesh advantage GROWS with K.
3. **Then** the agent test on the real dynamic: how an agent REASONS once it understands its
   own spawn topology ("am I in a tree or a graph?") and strategizes accordingly.
- Artifacts: `scratchpad/llm_maze.py` (turn-based controller), `scratchpad/llm_sweep.log`,
  team-size + hardness + fog sweeps (this session).

### 2026-06-25 (cont.) — comms-bottleneck sim: one artifact caught, one FAIR result
Per the design call, rebuilt the sim in Python land to make COMMUNICATION the bottleneck.
Two iterations, the first wrong — caught by an adversarial fairness check:

- **Attempt A — argmax aggregation (`scratchpad/comms_maze.py`): ARTIFACT, rejected.**
  K corridors, each room a token, exit = argmax(tokens) so no agent knows the exit alone.
  Naive run looked great (mesh flat 17t, tree 1.12×→1.65× with K). But it required GLOBAL
  aggregation — *every* agent needs all K tokens — so when I bandwidth-matched the mesh
  agents to the lead, **the gap collapsed to exactly 1.00× at every K.** The "win" was only
  from letting mesh aggregate for free while throttling the lead. Lesson: a task where
  everyone needs everything is topology-invariant; mesh has no fair advantage there.

- **Attempt B — independent pairwise coordination (`scratchpad/pair_maze.py`): FAIR, kept.**
  K corridors paired into K/2 LOCKS; lock p opens only if the agents at corridors 2p and
  2p+1 break ON THE SAME TURN. Partners are out of sight -> synchronizing REQUIRES comms.
  Model is intrinsically fair: every decider finalizes <= b lock-syncs/turn. A mesh agent
  decides its ONE lock (b never binds); the hub decides ALL K/2 (serializes at (K/2)/b).
  Result (b=1, L=5), and it SURVIVES the fairness check because the asymmetry is structural:

  | K  | locks | TREE | MESH | mesh× |
  |----|-------|------|------|-------|
  | 4  | 2     | 11   | 11   | 1.10× |
  | 8  | 4     | 13   | 11   | 1.18× |
  | 12 | 6     | 15   | 11   | 1.36× |
  | 16 | 8     | 17   | 11   | 1.55× |
  | 24 | 12    | 21   | 11   | 1.91× |

  **MESH flat (O(1), all locks sync in parallel); TREE linear (O(K/b)); gap unbounded in K.**
  b-sweep at K=16: b=1→1.55×, b=4→1.09×, b≥8→**1.00× tie** — the gap is purely one hub
  serializing many independent coordinations. b=inf == "the lead already holds the optimal
  joint plan" -> tie, exactly the user's caveat. This is the N-plates/simultaneity instinct,
  validated and fair. The honest claim is now SCALING ("centralized coordination doesn't
  scale with the number of independent coordinations"), not a fixed multiplier.
- NEXT: fold the pairwise-lock mechanic into the grid world for the LLM test; then the
  topology-aware agent test (does an agent that learns "I'm a hub-worker vs a mesh-peer"
  strategize accordingly?).
- Artifacts: `scratchpad/comms_maze.py` (argmax, rejected — keep as the cautionary case),
  `scratchpad/pair_maze.py` (the fair pairwise sim).

### 2026-06-25 (cont.) — abstract LLM topology test: the dynamic is REAL and vivid
`scratchpad/abstract_llm.py` — text-only LLM agents (haiku) on the pairwise-sync task, no
grid/web. Fair rule: every agent sends ONE targeted message/turn; topology restricts only
WHO you may message (mesh: anyone; tree: workers↔lead only, lead relays one worker/turn).
4 conditions × topology {mesh,tree} × mode {told,discover}. P=3 (3 locks), n=1:

| condition      | solved | turns | msgs | note |
|----------------|--------|-------|------|------|
| mesh / told    | yes    | 9     | 11   | peers coordinate directly, in parallel |
| mesh / discover| yes    | 8     | 9    | discovery ~free: try partner, it works |
| tree / told    | yes    | 16(cap)| **73** | lead serializes ONE pair at a time |
| tree / discover| **NO (2/3)** | 16 | 28 | 26 sends DROPPED — agents fight their topology |

Qualitative gold (the point of the whole demo):
- **tree/told**: the lead orchestrates pair (1,2)→lock@t3, (3,4)→lock@t8 (re-proposed 5×),
  (5,6)→never cleanly within cap. One coordinator, one pair at a time = O(K) + 73 msgs of
  round-trips. The lead is both a serial bottleneck AND a single point of confusion.
- **mesh/discover**: peers immediately try their partner, it works, confirm, BREAK together
  — parallel, ~free recognition.
- **tree/discover (the showpiece)**: agents START by messaging their partner directly (all
  DROPPED in a tree). SOME infer the topology and adapt — `a3→lead: "haven't reached agent 4
  ... can you relay to agent 4?"` — others NEVER do (a5↔a6 keep messaging each other every
  turn, both dropped, lock never opens → run FAILS). The recognition tax is real and here
  fatal. This is exactly the user's two framings: (told) the human picked the structure;
  (discover) the agent recognizes its structure and — the hook — could ask to be re-wired,
  "sagent offers both topologies and the right one is optimizable."
- Caveats: n=1, noisy; tree/told only barely solved at the cap; a P-sweep + seeds to confirm
  the SCALING for real agents is running (`abstract_sweep.log`). Build-order decision (user):
  abstract-first to de-risk → CONFIRMED, greenlight the grid build. Topology-awareness: run
  BOTH told + discover arms (user: both are good stories).
- Artifacts: `scratchpad/abstract_llm.py`, `abstract_p3b.log` (transcripts), `abstract_sweep.log`.

### 2026-06-25 (cont.) — spatial demo built; mechanic finalized; entry aligned
Folded the validated pairwise mechanic into the grid (`world.py`, `lock_lockstep.py`, `run.py`,
`web/index.html`). Mechanic journey (all empirically gated):
- sustained co-presence → too weak (camping: 27t/5msg vs 30t/11msg; comms NOT load-bearing).
- instantaneous press + charges, AUTONOMOUS serve_forever → too fiddly (logical clock jumps on
  every agent action; "press within a window" unreliable even with 56 mesh msgs → both fail).
- **LOCKSTEP** (every agent acts once/tick, presses resolve together) → clean. mesh 8t/34msg,
  tree 9t/56msg (told, P=3). Spatial scaling (P2-5): tree always more messages (the relay tax),
  but with generous budget both solve — the *failure* story lives in the abstract test; the
  spatial page is the visceral view. `run.py` aligned to `uv run python -m examples.agent_maze.run`.
  Web: two-zone (hero side-by-side mazes + terse synced convo + independent scrubbers / sync-default;
  explainer comm-graph + scaling). `lock_run.py` = the rejected autonomous driver (kept for history).

#### ⛔ GATING REQUIREMENT (user, 2026-06-25) — must showcase RECURSIVE SPAWN
The demo currently pre-places all 2P workers — it does NOT show sagent's first-class
budgeted/recursive spawn. MUST change to:
1. **Start with a SINGLE agent.** It understands it cannot solve alone and must SPAWN a team —
   but does NOT know how many; it discovers the count by exploring.
2. **Empty/dead-end paths**: corridors it explores that have NO lock — so exploration is real and
   the agent must map the maze (and not waste spawns) before knowing the team size.
3. The spawn itself is the topology contrast: mesh = ANY agent spawns (recursive, parallel team
   growth, peers are any-to-any); tree = only the lead spawns (serial growth, children are workers
   that talk only to the lead). This is the genuinely-hard-elsewhere capability.
Nit: agent movement must be a SMOOTH cell-to-cell transition (currently reads as jumps).
- Artifacts (this session): `world.py` (lock mechanic + asymmetric `make_lock_level` + press),
  `lock_lockstep.py` (lockstep engine), `run.py` (entry), `web/index.html` (replay page),
  `web/data.js` (captured P=3 told trace).

#### ✅ GATING REQUIREMENT MET + polished (2026-06-25)
- **Spawn-from-one works** (`make_spawn_level` + `run_spawn_arm`): one seed, each turn ONE of
  MOVE/SPAWN/PRESS/SEND (SEND costs a turn). SPAWN takes an agent-CHOSEN tile, validated
  (`can_spawn`: visible + passable + empty); invalid → feedback + wasted turn (no auto-edge).
  mesh = any agent spawns + broadcast; tree = only seed spawns + relay-through-seed. Dead-end
  decoy corridors. The fixes that unlocked it: spawn-first prompt ("you are ONE agent, a lock
  needs TWO"); helper prompt drives broadcast-to-pair-by-letter; id normalization (0→a0) +
  broadcast support. Result (told, 2 locks): **mesh solves ~2/3, spawning 4-6 (recursive);
  tree fails (serial seed-only spawn + relay) — visible in the spawn-count gap.**
- **Maze:** varied corridor lengths (`lengths` cycles per slot), paired plates on OPPOSITE
  sides in DIFFERENT rows (always out of sight → comms genuinely required). Budget 60 (44 was
  too tight for the harder out-of-sight maze — mesh fell to 0-1/2).
- **Smooth movement:** the web already interpolates positions; the "jumps" were the integer
  scrubber → made it fractional (`step=0.02`), slowed playback (TPS 1.25).
- **Polish:** CI gates all green (ruff check + format, codespell, ty error-on-warning — fixed
  via `Lock`/`SpawnMeta`/`PlateInfo` TypedDicts + casts). Removed dead `lock_run.py` (rejected
  autonomous driver) and stale `simulation_optimal_baseline.py` (retracted over-provisioning
  sim, wrong mechanic). `run.py` entry aligned (`python -m examples.agent_maze.run`), serve
  output matches demo 1, port **8001**. README rewritten for the spawn demo. Web: a description
  box under the title (puzzle / goal / mesh-vs-tree / what sagent sells).
- **Honest caveat:** mesh is ~2/3 reliable at budget 60 on **haiku** — diagnosed from the
  transcript as a comprehension miss (both partners pile on ONE plate instead of the two
  different same-letter plates, then thrash on press timing). Fixed the prompt (the "two
  DIFFERENT plates" rule + "press once on an agreed turn"), but haiku still tops out at 1/2.

#### ✅ FINAL — bumped to sonnet, 4-condition cherry-pick capture (2026-06-25)
Empirical A/B (fixed prompt, budget 60): **haiku mesh 1/2, 1/2; sonnet mesh 2/2, 2/2** (23-42t).
Sonnet tree: 2/3 solve but SLOWER (45-46t, more msgs) — a valid contrast ("tree slower also
demonstrates the point"). So **default model → `claude-sonnet-4-6`** (haiku via `LLM_MAZE_MODEL`).
- **Capture restructured** to all FOUR conditions (mesh/tree × told/discover) in one `data.js`
  (`{meta, modes:{told,discover}:{mesh,tree}}`), each cell **cherry-picked best-mesh / worst-tree**
  over k=2 runs. The page has a told/discover **toggle** (view either without `--live`).
  History capped to the last 12 turns sent to the model (cost; transcripts keep the full tape).
- **Shipped sonnet capture:** told/mesh 25t · told/tree 43t (1.7× slower) · discover/mesh 35t (0
  dropped) · **discover/tree FAILS 1/2, 27 dropped** (the recognition-tax showpiece — workers
  never infer they must route through the seed). Cost ~$12; validated the whole pipeline on a
  cheap haiku dry-run (18 structure checks) before spending sonnet. Model shown on the page badge.
- Web nits: hi-DPI canvases (crisp), description box under the title (puzzle/goal/mesh-vs-tree/
  what-sagent-sells), serve output matches demo-1, port 8001.

---

## 12. Environment & how to run

- Branch `demo/agent-mesh-maze` (off demo-1 HEAD; rebase onto sagent main before the PR).
- Keys: `~/.config/sagent/anthropic_api_key` (file-based). haiku default (~$0.07/arm).
- Replay (no key):  `uv run python -m examples.agent_maze.run`
- Re-capture both arms:  `uv run python -m examples.agent_maze.run --live`
- Tests:  `uv run pytest examples/agent_maze/ -q` (15, all green; ruff + ty + codespell clean)

- Branch: `demo/agent-mesh-maze` (off demo-1 HEAD; rebase onto sagent main before the PR so
  demo 2 is independent).
- Keys: `~/.config/sagent/anthropic_api_key` (file-based, never exported to a CLI subscription —
  demo-1 discipline). haiku default.
- Run (planned): `uv run python -m examples.agent_maze.run` (replay) / `--live` (recapture).

## 13. AUTONOMOUS REFACTOR — from lockstep to real sagent Agents (2026-06-29)

PR #230 (the lockstep demo) was approved, but the maintainer's review comment #2 asked the
load-bearing question: *drive agents via `Agent` + tools instead of a hand-rolled `ModelRequest`
lockstep.* That's the right call — the lockstep version reads as "a sim with an LLM in the loop,"
not "sagent agents." This section logs the rebuild (branch `demo/agent-maze-autonomous`).

**The pivot (user, 2026-06-29):** drop the global turn entirely. Agents are autonomous sagent
`Agent`s acting through tools; the World is a *reactive feedback service*. The only thing that ever
needed synchronization was the simultaneous press — so make *that* the one designed mechanic and
let everything else (look/move/message/spawn) be free and async.

**ADVERSARIAL DESIGN REVIEW (4 lenses × 2 "are-you-sure" verifies, run before building).** It
caught real holes and — importantly — the verification *refuted* two attractive-but-wrong fixes:
- ✗ "press once and walk off = zero comms" — refuted: co-presence at latch is required.
- ✗ "only let an agent name a partner who messaged it" — refuted: that makes the TREE arm
  *unsolvable* (tree workers can't DM each other). Instead the comms-requirement comes from
  out-of-sight partner *discovery* + the timing window.
- ✓ KEYSTONE: a wall-clock TTL is unsatisfiable (can't be both short enough to force a handshake
  and long enough to absorb LLM latency — the exact thing that sank the first autonomous attempt).
  Fix: a **logical interaction clock**. Latency-independent, reproducible.
- ✓ Headline must NOT be "total interactions" (diluted by topology-invariant moves; undefined on
  the failure arm). Use **coordination cost per lock opened** + a ✓solved/✗STUCK badge.
- ✓ Recording gaps (scene header, per-cell moves, every-press-attempt, parent-keyed spawn) and the
  **shutdown barrier** (persistent peers are exempt from parent shutdown → leak/cost after solve).

**The mechanic that shipped:** `press(partner=<label>)` arms a plate naming a partner, live
`PRESS_WINDOW` *logical* interactions; a lock opens when both plates are armed, each naming the
other, two distinct agents on the two different same-letter plates, within overlapping windows.

**Build (each step runnable + tested):** engine.py (reactive engine + event log) → tools.py
(WorldTool/CommsTool/SpawnTool) → arena.py (concurrent agent tasks + shutdown barrier) →
event-stream web/index.html → capture.py (4 arms + metrics + data.js). The 2-agent rendezvous
gate: **6/6 live solve-rate**. SpawnTool is a thin demo tool (not sagent's `AgentSpawn`): the
tick-free bodies run as concurrent tasks that must keep exploring, which `AgentSpawn`'s bundled
run-loop (run-to-completion / serve_forever) can't model.

**Scale call (K) — the review was right.** First tried to ship **K=2** (clean, 5-agent maze). But
the k=2 cherry-pick exposed the exact fragility the design review warned about: the captured
`discover/tree` *solved 3/3 and CHEAPER than mesh* (13.5 vs 22.5 cost/lock) — at 2 locks the hub's
serial relay can still keep up, so "tree fails" was partly luck, and that arm read as anti-thesis.
Bumped to **K=3** (3 locks / 6 plates, agent cap 8 to avoid the over-spawn the K=3 probe hit at cap
10). At K=3 the hub reliably chokes: the shipped capture is **mesh ✓ 3/3 in BOTH told and discover**
vs **tree ✗ STUCK 0/3 in BOTH** — a robust, binary contrast. Nice detail: `discover/tree` spawned
**8 agents and opened 0 locks** (the hub can't coordinate them) next to mesh's **4 agents solving
3/3**. Lesson re-confirmed: at small K the contrast is variance-dominated; K≥3 is the floor.

**Validated:** mesh ✓3/3 vs tree ✗STUCK 0/3 (told + discover); recursion shown (mesh lineage
branches, e.g. a3<-a1; tree flat a*<-a0); shutdown clean (no "Task was destroyed" leaks);
event-stream replay renders the contrast + cost-per-lock headline + genealogy. ruff/ty/codespell +
21 tests green.

### Failure path — dead-ends a cold session should NOT re-walk

The autonomous rebuild was *not* a straight line. Each of these was tried, observed to fail, and
fixed; the fix is in the code, but the reason lives here (and in the per-step commit messages
`3d20733`…`66ef3b0`). If you find yourself reaching for one of the ✗ options, this is why not.

1. **✗ Wall-clock press TTL (~20s).** The intuitive design. The adversarial design review proved it
   *unsatisfiable*: it cannot be simultaneously short enough to force a real-time "press now"
   handshake and long enough to absorb the 5–20s LLM latency tail — the identical failure that
   killed the very first (pre-lockstep) `serve_forever` attempt. **→ logical interaction clock**
   (`PRESS_WINDOW` counts decisions, not seconds): latency-independent and reproducible.
2. **✗ Binding partner-naming to "must have messaged me."** Looked like the way to force comms.
   The 2× verify refuted it: in the **tree** arm workers can't DM each other, so it makes that arm
   *unsolvable by construction*. **→ comms is forced instead by out-of-sight partner *discovery***
   (you can't know which agent is on the matching plate without asking) + the logical window.
   (Also refuted: "press once and walk off opens it" — co-presence at latch is required.)
3. **✗ First multi-agent run: 158 messages, stuck at 1/2.** Agents chattered every round and ran
   out of the per-agent round budget before covering both locks. **→ terser prompts** (a couple of
   coordinating messages, then ACT) + the seed **assigns pairs explicitly** + round cap 20→28.
   Result: 2/2 at 57 msgs (`a44b3b9`).
4. **✗ Shipping K=2.** A k=2 cherry-pick at 2 locks produced a `discover/tree` that *solved 3/3 and
   cheaper than mesh* — actively anti-thesis, because at 2 locks the hub's serial relay keeps up.
   This is exactly the variance-fragility the review warned about. **→ K=3** (the floor where the
   hub reliably chokes; see the scale-call paragraph above).
5. **✗ K=3 at max_agents=10.** Mesh over-spawned to the cap (team 10), ~487 messages, and a storm of
   API ReadErrors from 10 concurrent agents. **→ cap max_agents=8** (6 plates need ~6–7 workers;
   the cap stops the over-spawn without starving the team).
6. **✗ Cancelling drive tasks bare.** The step-2 driver cancelled agent tasks without quiescing
   their internal awaits → "Task was destroyed but it is pending" leaks, and (per the review) any
   persistent peer would keep billing *after* the win. **→ shutdown barrier** in `arena.py`:
   `agent.shutdown()` on each, then cancel + `gen.aclose()`, before serializing the log.

**If reviving K=2 or a wall-clock window, re-read this section first.** Both were real, reasonable
attempts that the data/review killed; the current K=3 + logical-clock design is the response.
