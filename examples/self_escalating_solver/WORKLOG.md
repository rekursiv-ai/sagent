# WORKLOG — self-escalating-solver demo (sagent agent-directed model mutation)

> **Read this first.** This is the pickup doc. It records the goal, the full
> trial-and-error so far (what worked, what didn't, and *why*), the current
> design pivot, and exactly where to start. Last updated **2026-06-24**.
>
> **Status:** the original *code-tracing* task is **abandoned** (see § Journey —
> it cannot demonstrate the feature). The task domain is now a **self-contained
> numpy/scipy stochastic gotcha** (NOT Bayesian-catelog — too specific; NOT SMC —
> too much reader domain-knowledge). **The problem is LOCKED and the saturation
> gate PASSED** (cheap fails 75%, strong 100%) — see § 0. Harness plumbing is
> reusable; next is wiring code-execution + the full self-mutate loop.
>
> Repo: `/home/jp/rekursiv/sagent`, branch `feat/AnthropicCLI` (the demo dir is
> currently **untracked** — see § Branch & commit). Demo dir:
> `examples/self_escalating_solver/` (will likely be **renamed** to
> `examples/bayesian_self_escalation/`).

---

## ⭐ CURRENT STATE — CROSS-PROVIDER self-mutation (2026-06-24)

**The resolution to "the cheap model is too capable."** The winning design is
**cross-vendor self-mutation**:
- **cheap = `gemini-2.5-flash-lite` (Google)** — genuinely weak; really fails the plain
  Jacobian bug. (Anthropic's haiku is too capable — it passes even the harder CI /
  standard-error challenge **4/4**, so an *in-vendor* "cheap fails" story is contrived.)
- **strong = `claude-opus-4-8` (Anthropic)** — reliable fixer.
- **self-mutate:** the Google agent calls `AgentSelf({"model_id":"claude-opus-4-8"})` and
  **re-homes itself across vendors mid-run, keeping memory.** Validated live (Opus replied:
  *"I switched from the previous Gemini model"*). Loudest possible showcase of model-agnostic
  mutation — no other stack lets an agent change *companies* on the fly.

**Cross-provider swap — the gotcha:** the Agent MUST be built with
`model_spec=ModelSpec(provider="Google", auth="api", model_id=...)`. AgentSelf swaps *from* that
spec; with just `model_id="claude-opus-4-8"`, `infer_provider` maps `claude-*` → Anthropic, the
provider allow-list defaults to ALL providers, and `build_provider("Anthropic","api")` reads
`ANTHROPIC_API_KEY`. So set `ANTHROPIC_API_KEY` in the **run process** env (never the CLI's) for
the swap to build the Anthropic provider. No `model_spec` → "Agent has no model spec; cannot swap."

**Run:** `uv run python -m examples.self_escalating_solver.run --live --provider cross --trials 4`.
`run.py` `CONFIGS` = {google, anthropic, cross}; `build(provider, model_id) -> (Model, ModelSpec)`.

**Free testing via subscription:** `AnthropicCLI.from_credentials()` drives `claude -p` (no API
key, uses subscription) — used for the empirical search (`scratchpad/empirical_search.py`).
Caveat: token-cost reporting is inflated + a noisy unclosed-subprocess teardown (call `close()`).

**Harness is DONE:** oracle grading (`check(samples)` → PASS/FAIL inside `run_python`, no
ground-truth leak), shared prompts (low/high get identical `SYS_BASE`; self = `SYS_BASE` +
upgrade block — `system_for(allow_upgrade=...)`), timeline capture (think/run/swap), canonical
histograms. Grader = KS vs Gamma(2,2), D<0.05. The harder CI/standard-error grader is built in
`scratchpad/empirical_search.py` (`ORACLE_CI`, batch-means) but **not needed for cross** (flash-lite
fails the simple Jacobian). **Next:** capture the cross hero run + build the 3-panel webpage with
the **Google→Anthropic swap as the money shot**.

---

## Failure paths — everything we tried, and why each failed or worked

Condensed map (full detail in the sections below). The throughline: **modern LLMs are
too capable for most "the cheap model just can't do it" framings** — the gap had to
come from a genuinely weak model (Gemini Flash-Lite) *and* a genuinely uninformative
grader.

1. **Code-output tracing puzzles** — ABANDONED. Flash-lite is over-confident → 0
   self-escalations. No weaker Gemini available (1.5/2.0 → 404 or not in the allow-list).
   Verify-then-escalate → it self-corrects on the cheap tier. Mandatory-escalate → the
   strong model **anchors on the cheap model's wrong trace** and does *worse* than solving
   fresh (memory retention backfired). Feedback loop → self-corrects again. **Root cause:**
   code-tracing is an *effort gap*, not a *capability ceiling* — LLMs are saturated on
   code, so a nudge + a retry lets them recover. No wall to hit.

2. **Bayesian numpy MH-Jacobian gotcha** — the chosen task. Dropping the
   multiplicative-proposal Jacobian secretly samples Exponential (mean 2) instead of Gamma
   (mean 4): a *whole-different-shape* bias, maximally visual, objectively gradable.
   One-shot saturation: Flash-Lite ships the biased version ~75% of the time; gemini-3.1-pro 0%.

3. **In-vendor "cheap fails" doesn't hold** — in an agent loop with grader feedback +
   retries, the cheap model self-corrects. Anthropic's **haiku passes even the harder
   confidence-interval / Monte-Carlo standard-error challenge 4/4** (tested live and free
   via `claude -p`; see `scratchpad/empirical_search.py`). A same-vendor "cheap can't"
   story is contrived.

4. **CROSS-VENDOR self-mutation** — the resolution. cheap = Gemini Flash-Lite (genuinely
   weak), strong = Claude (reliable). The agent re-homes itself Google → Anthropic mid-run.
   **Gotcha:** the Agent needs a `ModelSpec` or the swap aborts with *"Agent has no model
   spec; cannot swap."* `claude-*` auto-infers Anthropic; set `ANTHROPIC_API_KEY` in the run
   process only (never the CLI's).

5. **Grader hints tutor the weak model** — a grader returning "distribution distance 0.36"
   lets Flash-Lite reason its way to the fix. A **bare `PASS`/`FAIL`** grader (no diagnostic)
   → it can't self-correct → it escalates. This one change took the self-upgrade rate 1/4 →
   3/4 and restored the low=FAIL contrast.

6. **Memory retention costs tokens — and the per-task cost story is a KNOWN LIMITATION
   (honest).** The strong model re-reads the cheap model's failed transcript, so a self-mutate
   that flails before swapping costs *more* than the strong model solving fresh. Escalating on
   the FIRST `FAIL` (minimal transcript) trims that — but **empirically the swap money-path
   still lands ~$0.05–0.07, ABOVE always-Opus (~$0.044)**: Sonnet's verbose re-derivation
   roughly matches Opus's concise solve in cost (measured across 3 live `cross` runs —
   money-paths $0.0525 / $0.0635 / $0.0700 / $0.0767, Opus $0.0436). The occasional `< Opus`
   capture (e.g. $0.038) is a **lucky low-tail draw, not typical** — we cherry-pick the shipped
   run and disclose it. The *real* cost win is **adaptive routing, not this single task**:
   trials that DON'T need to swap solve on the cheap model for **$0.002–0.006 (~10× cheaper
   than Opus)**, so cheap-first-escalate-when-needed beats always-Opus *across a workload*. The
   headline value is **autonomy + cross-vendor + memory-preserving self-upgrade** — the agent
   deciding for itself that it's stuck and re-homing to a stronger vendor — not beating Opus on
   one hard task. Do not claim "cheaper than Opus on this task" in the README/UI.

7. **Low-tier variance** — Flash-Lite occasionally self-corrects given enough budget,
   breaking the contrast. Tightened its budget (0.12) + retry-on-503 so the cheap-alone arm
   reliably FAILs while still showing several real attempts.

**Built but NOT needed:** a confidence-interval / batch-means standard-error grader
(`scratchpad/empirical_search.py`, `ORACLE_CI`) for if a future cheap model outgrows the
plain Jacobian. Haiku passed it 4/4; Flash-Lite fails the plain Jacobian, so it's unused.

**Final knobs that make the shipped run clean:** cheap budget 0.12 + re-roll until a *graded*
FAIL (low FAILs with visible attempts; re-roll past a 503/ungraded run and past the occasional
lucky pass); bare-FAIL grader (forces escalation); escalate-on-first-FAIL (keeps the handoff
short — but not below Opus, see #6); high-tier = Opus / self-mutate → Sonnet (distinct models
so the panels differ — cost is *comparable*, NOT a self-mutate win); hero = the **cheapest**
money-path so the swap is always a real cross-vendor switch with cheap-model reasoning before
it; think-text captured in **full** (clip 100000) and collapsed behind `[…]` in the UI, with a
sentence-boundary tidy as a safety net so a block never ends mid-sentence. (Earlier captures
clipped reasoning to 700 chars, so `[…]` revealed a still-truncated block — fixed by
re-capturing with the raised clip.) **The shipped `data.js` is cherry-picked** (low FAIL +
clear swap + full reasoning); a live re-run won't always land that cleanly — by design.

8. **CI gates — examples must be fully clean (no blanket ignore).** Four gates run on `.`:
   `ruff check`, `ruff format --check`, `codespell`, and `ty check` with
   `error-on-warning = true` (every type *warning* is fatal). Gotchas: (a) codespell flags
   ordinary identifiers as typos — a list var and a JS slice-count param both tripped it (and a
   camelCase rename of the param still tripped codespell, which is case-insensitive); renamed to
   `self_runs` / `nShow`. (b) `ty` needs explicit type args (`dict[str, Any]`,
   `list[Any]`), a non-`None` init for the retry-loop `low`, and an explicit `list[Tool]`
   annotation so the conditional tools list doesn't widen to an un-assignable union. (c)
   print / sandboxed-subprocess / marker-token / lazy-import are covered by a scoped
   `examples/self_escalating_solver/**` per-file-ignore mirroring the `*.ipynb` precedent.

### Future improvement — make COST a first-class driver of the switch (option)

Today the agent switches on **capability** (it can't pass the grader), and cost is only an
*observed* number on the panels — which is why the cost story is weak (see #6: the swap costs
≈ Opus). A stronger, more honest framing makes **cost/time the actual trigger**:

- Give the agent an explicit **cost budget AND wall-clock/time budget** up front, surfaced in
  its context, and have it **self-upgrade (or self-*downgrade*) when it projects it will cross
  a threshold** — e.g. "I've spent $X / N seconds and I'm not converging; escalating is the
  rational move," or conversely "this is easy, drop to a cheaper model." sagent already exposes
  `max_budget_usd`; the missing piece is feeding *remaining* budget + elapsed time back to the
  model and prompting it to reason about them.
- This turns the demo from "weak model can't, so it escalates" into "an autonomous agent
  **managing a cost/latency budget**, routing itself across vendors/tiers to stay within it" —
  where the switch decision is genuinely *economic*, not just capability-driven, and where
  showing cheaper-on-average is honest because routing is the whole point.
- Bonus: it exercises self-*downgrade* too (cheap-when-easy), making the adaptive-routing cost
  win from #6 a *demonstrated behaviour* rather than an aggregate argument.

Deferred — current demo ships the capability-triggered switch; this is the next iteration.

---

## 0. LOCKED PROBLEM + saturation result — 2026-06-24  ✅ GATE PASSED

**Problem:** a hand-rolled **Metropolis-Hastings sampler in pure numpy/scipy** with a
**missing Hastings/Jacobian correction**.
- **Target:** unnormalized density ∝ `x * exp(-x/2)` for x>0 (= Gamma(shape 2, scale 2);
  mean 4, mode 2). Stated *unnormalized* so the model implements MH instead of calling
  `scipy.stats.gamma.rvs`.
- **Proposal:** **multiplicative** random walk `x' = x*exp(eps)`, `eps ~ Normal(0,sigma)`
  — asymmetric, so correct MH needs the proposal-density ratio `q(x|x')/q(x'|x) = x'/x`.
- **The gotcha:** the naive sampler uses the symmetric-Metropolis ratio `p(x')/p(x)` and
  **drops the `x'/x` factor**. It then targets `p(x)/x ∝ exp(-x/2)` — a pure
  **Exponential(2)**, not the Gamma. The bias is a *whole different shape*
  (monotone-decreasing vs a bump at 2) → maximally visual. mean 2 vs 4, var 4 vs 8.
- **Fix:** multiply the accept ratio by `x'/x` (in log-space: `log_alpha = 2*eps - 0.5*(x'-x)`).
  One term.

**Why this beats every earlier candidate:** simple/fast (~10-line sampler), numpy+scipy
only, stochastic (random proposals), reader needs **zero** domain knowledge (just sees
wrong-shape vs right-shape), and it *is* the user's original "biased sampler → self-mutate
→ looks good" vision.

**Saturation gate — the thing that killed every earlier task — PASSED.** 4 generations
each, every snippet run in an isolated subprocess and KS-tested vs Gamma(2,2):
- **cheap (`gemini-2.5-flash-lite`): 3/4 BIASED** (Exponential, mean≈2), 1/4 correct.
- **strong (`gemini-3.1-pro-preview`): 4/4 CORRECT** (Gamma, mean≈4).
- The cheap model **self-incriminates in its own comments**: *"the proposal is symmetric
  in log-space, so the proposal ratio = 1"* → ships the bug. The strong model does the
  change-of-variables correctly (the Jacobian appears as the `2*eps` term). The fix is
  literally one factor (`× x'/x`). Excellent timeline narrative.

**⚠️ Signal-design catch (use the right trigger):** with ~35k autocorrelated MCMC samples
the **KS p-value is too sensitive** — even *correct* strong runs scored p=6e-8…0.19, so a
p-value threshold would false-trigger. Use the **KS D-statistic** (biased ≈0.36 vs correct
≈0.01 — clean separation) **or a moment check** (sample mean ≈2 vs target 4). That is the
agent's objective "I'm stuck" signal.

**Repro:** `scratchpad/mh_gotcha.py` (gotcha + before/after plot), `scratchpad/saturation.py`
(cheap-vs-strong test). Models via sagent Google provider; key at
`~/.config/sagent/google_api_key` (read inline, never print).

**Next (de-risk #2):** wire a **code-execution tool** into the agent and test the **full
self-mutate loop end-to-end**: cheap writes the biased sampler → runs it → reads mean≈2 /
D≈0.36 (objective fail) → calls `AgentSelf` → strong continues with the sampler+diagnostic
→ adds `x'/x` → mean≈4 / D≈0.01. Only then build the 3-panel timeline webpage.

### De-risk #2 status — 2026-06-24: INFRA ✅ / MONEY PATH ⚠️ (partial)
Built a sandboxed `run_python` tool (subprocess, numpy+scipy; `scratchpad/derisk2.py`) and
ran the 3 conditions live on the clean branch.
- **Infra works natively** (no backport): multi-turn `run_python`, real cost, and
  **`AgentSelf` escalation fires** (a flash-lite run swapped itself to pro mid-loop).
- **The clean money path (biased → escalate → FIXED) is NOT yet reliable:**
  1. **flash-lite self-corrects given the diagnostic.** Told "mean should be 4", it
     sometimes writes biased (mean≈2) then fixes the Jacobian *itself* on a retry
     (→ mean≈4), no escalation — same ceiling-softening as code-tracing.
  2. **When it escalated, the fix didn't land** — the pro continuation re-ran the biased
     code, stayed at mean≈2, reported BIASED (shallow, ~$0.002). The agent loop seems to
     end after the swap before pro actually diagnoses + fixes + reruns.
  3. **gemini-2.5-flash didn't engage the tool** (runs=0) in one attempt — separate flakiness.
- **To fix before the webpage:** (a) make the post-escalation strong continuation reliably
  fix+rerun — after `AgentSelf`, the system should re-instruct "you are now the STRONG model;
  find the bug in the existing sampler and re-run until mean≈4; do NOT report BIASED"; (b)
  decide cheap-tier/prompt so escalation is the *expected* path (flash-lite self-correcting
  undercuts the story — e.g. detect-and-escalate on the FIRST biased result instead of
  allowing a cheap fix attempt, or use a slightly different framing). Repro: `scratchpad/derisk2.py`.

### Decisions locked — 2026-06-24 (post variant-test, user-confirmed)
Variant test (`scratchpad/variant_test.py`, workflow `self-upgrade-prompt-variants`, 4
framings × 4 trials): **B's explicit "you can upgrade your own model / level up" framing won
on fix-quality** — its escalations were 2/2 genuine fixes, while A/C/D escalated more but the
upgraded model **re-ran the biased code and gave up** (~1/3 fixed). The dominant failure is
**post-swap give-up: the strong model anchors on the cheap model's *defeated* context** —
memory retention carried the defeat, not just the code. B's empowerment framing counters it
(framing the agent's *identity* beat instructing the *task*, which D's directive did not).
Second finding: **flash-lite fabricates `RESULT: SUCCESS` without running code** (skips the
tool, hallucinates printed moments).

- **Reliability target: ~75%, NOT 100%.** Be honest in the UI: "logged success run; ~75% live
  under Gemini + these prompts; varies by model/prompt."
- **End-user command DEFAULTS to `replay`** of a captured success path (deterministic, no API
  key). A `--live` flag re-runs for real (`google`|`anthropic`) and may or may not land.
- **Capture a success path:** run live until one clean money-path run (biased → self-upgrade →
  fixed, harness-graded mean≈4); serialize its full timeline (per step: model, reasoning, code,
  diagnostics, the swap, cost) + biased/fixed histograms. That + the low-tier (biased) and
  high-tier (correct) traces = the replay artifact (`web/data.js`).
- **Prompt is MODEL-AGNOSTIC** (locked text in the worklog turn): generic "upgrade yourself /
  level up / after upgrading you ARE the stronger model, take ownership" + post-swap ownership;
  only `{strong_model}` injected. Framing = variant **B** + ownership directive. Works on
  Gemini or Anthropic unchanged.
- **Harness-computed grading, NOT the agent's `RESULT:` self-report** — extract the agent's
  FINAL sampler code, run it in the harness, compute the true mean → objective per-panel
  pass/fail. Sidesteps flash-lite fabrication; keeps the demo honest.

---

## 1. Goal

Showcase the one thing sagent does that a normal agent stack cannot: **an agent
swaps its *own* model mid-task** (`AgentSelf` → `Agent.swap_model`), *keeping all
of its working memory*. It starts on a cheap/fast model and escalates *itself* to
a strong model the moment the work turns out to be genuinely hard — then continues
(not restarts).

**The irreducible advantage** (this is the whole point; everything else is a
distraction): a pre-task *router* (cheap model judges difficulty, spawns a fresh
expert) ties self-mutation **whenever difficulty is visible upfront AND the work
is atomic**. Self-mutation *only* wins when **both**:
1. **Difficulty is discovered, not visible** — the agent can't know the hard part
   is coming until it is already deep in the work (a cold classifier can't route
   what it hasn't seen yet); and
2. **There is *correct, reusable* accumulated state to continue from** — so the
   strong model resumes from good partial work instead of re-deriving (and instead
   of anchoring on a *wrong* final answer, which is what hurt us — see § Journey).

### End-user experience (the deliverable to build toward)

A user with API keys already configured runs **one shell command**. It:

1. **Prompts for a provider** — `google` or `anthropic` (CLI/API; they have the
   key set up already).
2. **Runs the experiment** — the Bayesian problem set × 3 strategies.
3. **Spins up a local webpage** with **3 panels**, side by side:
   - **Low-tier model** (cheap, pinned) — often *fails* / diverges / gives a wrong
     diagnosis.
   - **High-tier model** (strong, pinned) — solves, but pays full cost on every
     step.
   - **sagent self-mutation** — starts cheap, hits the wall, **swaps its own model
     mid-run**, finishes.
4. **Timeline animation** — each panel replays its run step by step (reasoning →
   code it ran → the diagnostic output / failure), showing *how* it solves or
   fails the problem, in real cost.
5. **Explicitly highlights the mutation moment** — the exact step where the sagent
   panel calls `AgentSelf` to upgrade itself is called out visually (this is the
   money shot).

Everything a run produces must be **capturable end-to-end as a single
doc/report** (the agent is now allowed to *run code*, not just reason — but the
transcript of reasoning + code + output must serialize to one artifact per panel).

---

## 2. Why Bayesian / stochastic problems (the task pivot)

Decided 2026-06-24 with the user. Coding/code-tracing tasks are **too easy** for
modern models — LLMs are saturated on code examples, so a cheap model either
solves it or self-corrects with a nudge; it never hits a wall it can't retry out
of (see § Journey for the data). Bayesian inference is different and fits the
goal's requirements *intrinsically*:

- **Real capability ceiling, not an effort gap.** "This sampler diverges —
  reparameterize the geometry" is reasoning LLMs are *not* saturated on. A cheap
  model genuinely cannot reason its way out of a funnel; a strong one can. This is
  the cliff code-tracing never had.
- **The failure signal is *objective and intrinsic*.** Divergences, R̂ > 1.01,
  tiny ESS, NaN gradients, BFMI warnings — the agent **runs the sampler and watches
  it fail**. That is the "explicit feedback so the agent knows it's stuck" we need,
  built into the domain instead of bolted on. No reliance on the model's (broken,
  over-confident) self-assessment.
- **Memory retention can't backfire here.** On escalation the strong model inherits
  the *model spec + failed sampler config + divergence output* — all unambiguously
  useful. There is no "confident-but-wrong final trace" to anchor on, because the
  failure is a *diagnostic*, not a wrong answer. The router (fresh expert) must
  re-derive the whole model; the self-mutator continues.
- **Stochastic random failure raises difficulty further** — the user's point: an
  algorithm with random failure modes is much harder than a deterministic trace.
- **The demo doubles as a prototype** for the user's real **Bayesian consultant**
  product (same escalation trigger: sampler won't converge → escalate).

---

## 3. Problem source — `rekursiv/bayesian-catelog`  ⭐

**Do NOT invent problems.** Use the user's own curated corpus at
`/home/jp/rekursiv/bayesian-catelog` (note the spelling: *catelog*). It is an
expert-curated (Junpeng Lao) knowledge base distilled from PyMC/Stan/Pyro forums +
Betancourt case studies + Dan Simpson's blog, **explicitly built around the
failure path**: "broken code → diagnosis → wrong fix → iteration." That is exactly
the cheap-fails-then-escalates structure this demo needs.

Concretely useful entry points (verified 2026-06-24):
- **`eval_cases/*.json`** — 54 cases, each `{ "uid": "stan:11203", "gold": "<expert
  answer>" }`. A source-thread id + the curated correct answer. Use `gold` as the
  ground-truth to grade an agent's diagnosis. (Need to join `uid` → the original
  question text; the raw threads are under `raw/` — `raw/stan/`, `raw/pymc/`, etc.)
- **`catalog/interview/probe_problem.txt`** — a rich, self-contained interview-style
  problem (mixture-of-two-Gammas that won't separate — a label-switching /
  non-identifiability failure, with the Stan code + symptom + what was already
  tried). This *format* (full problem statement, symptom, failing code) is the
  ideal demo problem shape.
- `experiments/spike_slab_vs_horseshoe.py` + `experiments/RESULTS.md` — a real
  method-comparison experiment (template for "run code, read diagnostics").
- `README.md`, `SCHEMA.md`, `MANIFEST.md`, `CLAIMS_INDEX.md` — corpus structure.

**Next-session task:** survey `eval_cases/` + `catalog/interview/` and pick (or
adapt) **3–6 problems** with a *reliable* cheap-fails / strong-fixes structure and
an *objective* failure signal. Classic candidates that fit perfectly:
- **Neal's funnel** — centered parameterization diverges; non-centered fixes it.
  Cheap model writes centered, sees divergences; strong model reparameterizes.
- **Label-switching / non-identifiable mixtures** (the interview probe above).
- **Wide/poor priors or bad init** → bad mixing / R̂ won't drop.
- **Sampler choice** (RWMH vs NUTS, step size) → low ESS.

---

## 4. Current design (to build)

**Conditions** (map to the 3 panels; router kept in *data* for the honest
comparison but not a headline panel):
- `low-tier`  / `static-cheap` — one agent pinned to the cheap model.
- `high-tier` / `static-think` — one agent pinned to the strong model.
- `self-mutate` — starts cheap, escalates **itself** via `AgentSelf` when its
  sampler run produces bad diagnostics, continues on the strong model with the
  model spec + diagnostics in memory.
- *(optional)* `router` — cold classify → fresh strong agent re-derives from
  scratch (the counter-argument baseline; expected to tie on accuracy but cost more
  / lose memory).

**Escalation trigger (the fix for everything that failed before):** the agent
**runs the model**, and an *objective* diagnostic (divergence count, R̂, ESS,
NaNs) is the signal. Wire it so the agent sees the diagnostics and decides to
escalate — OR, if the cheap model still won't pull the trigger reliably (the
meta-1 risk below), feed the diagnostics back as an explicit "your sampler
diverged" turn so the stuck state is unambiguous.

**Code execution:** the agent needs to run JAX/BlackJAX (or PyMC/numpyro) and read
diagnostics. Decide the execution path: a sagent code-exec tool vs a sandboxed
Python tool (cf. `agent-team/.../sandboxed_tools.py`). Must be safe and the
transcript must serialize to the single-doc report.

**Models** (current `solver.TIERS`):
- google: `gemini-2.5-flash-lite` (cheap) ↔ `gemini-3.1-pro-preview` (think)
- anthropic: `claude-haiku-4-5` (cheap) ↔ `claude-opus-4-8` (think)

---

## 5. Journey — what we tried, what worked, what didn't

### Attempt 1 — code-output tracing puzzles  ❌ ABANDONED
30 deterministic Python "predict the printed output" puzzles (12 easy / 12 medium
/ 6 hard), answers validated by running them once at build time; agent reasons only.
Four conditions (static-cheap, static-think, router, self-mutate). Ran live on
Gemini. **It does not work**, for reasons that turned out to be fundamental:

| What we tried | Result |
|---|---|
| Original self-mutate prompt ("escalate when unsure") | **0 swaps** ever. flash-lite is over-confident — never feels a wall. self-mutate scored **50% on the 6 hard** — *worse* than static-cheap's 83% (the escalate framing made it under-reason). |
| **Weaker cheap model** (widen the gap from the model side) | **Dead end.** `gemini-1.5-flash` & `gemini-2.0-flash` → **404 on the streaming path**; `gemini-2.0-flash-lite` is **not in sagent's `Google` KNOWN_MODELS allow-list**. `gemini-2.5-flash-lite` is the usable floor. |
| **Verify-then-escalate** prompt (re-derive a 2nd way, escalate on mismatch) | Accuracy ↑ (77.8% → **88.9%** on a 9-mix) but via **self-correction on the cheap model** — **still 0 swaps**. Doesn't exercise the feature. |
| **Mandatory escalate** on gotcha classes | Swaps finally fire, but **inconsistently (2/5)**. And the killer: on `hard-04` it escalated to pro and **still got it wrong** — the pro continuation **anchored on flash-lite's wrong trace** — while a *fresh* pro (static-think) got `hard-04` right. **Memory retention backfired.** |
| **Feedback loop** (submit → "INCORRECT (attempt N)" → retry → escalate when stuck) | The feedback *works* (med-05: wrong → self-fixed on retry), but flash-lite **self-corrects on the cheap model** and never gets stuck enough to escalate. **Still 0 swaps.** |

**Root-cause insight (the load-bearing lesson):** code-tracing has an **effort
gap, not a capability ceiling**. flash-lite *can* trace these; it just needs to be
careful, and given a retry it recovers *itself*. Escalation only becomes
*necessary* at a real ceiling — a problem the cheap model **cannot** do no matter
how many retries, but the strong model can. LLMs are saturated on code, so that
ceiling isn't there. Hence the pivot (§ 2).

Also observed: **memory retention is a double-edged sword** — carrying a *wrong
final answer* across the swap poisons the strong model. The Bayesian design avoids
this because what's carried is a *model spec + objective diagnostics*, not a wrong
answer.

### What works and is REUSABLE (don't rebuild)
- **sagent self-mutation is agent-invokable + observable.** `AgentSelf()` tool in
  `tools=[...]`; the model calls it with `{"model_id": "..."}`; `extract_swaps()`
  in `solver.py` reliably recovers every swap + its reason from `agent.history`.
- **Real cost tracking works** even on stochastic/CLI auth: `agent.total_cost_usd`.
  flash-lite ≈ $0.000001/short-call; `gemini-3.1-pro-preview` ≈ 20× — a sharp
  frontier.
- **Multi-turn memory across turns** — feeding follow-up `UserMessage`s to the same
  `Agent` continues the conversation with full history (used for the feedback loop;
  this is how "objective diagnostics fed back" will work).
- **The run/grade/serialize pipeline** (`run.py` → `results.jsonl` / `summary.json`
  / `web/data.js`; `web/index.html` reads `window.DEMO`) — the *plumbing* is sound;
  it needs new panels (3-up) + a per-step timeline + the swap highlight.
- **Offline mock** (`solver._Offline(MockModelCaps)`) for no-key plumbing checks.
- `run.py --ids a,b,c` and `--conditions` flags for targeted calibration runs.

### Empirical numbers (for reference; all Gemini, live)
- static-cheap (flash-lite) on 6 hard code-trace: **5/6 (83%)**, $0.0067.
- static-think (3.1-pro) on 6 hard: **6/6 (100%)**, $0.0196.
- self-mutate (orig prompt) on 6 hard: **3/6 (50%)**, **0 swaps**, $0.0032.
- flash-lite is **stochastic** on hard problems (hard-04 failed in one run, passed
  in another) — reinforces that the gap is an effort gap, not a ceiling.

---

## 6. Open questions / decisions for the next session

1. **"google or anthropic CLI"** — clarify wording: provider *API* selection (what's
   built) vs the actual Claude/Gemini *CLI subscription auth*. Current harness uses
   **API mode** (`Google.from_key` / `Anthropic.from_key`). Confirm with user.
2. **Meta-1 risk — does the cheap model *reliably* pull the trigger?** Even with an
   objective divergence signal, will flash-lite/haiku call `AgentSelf`? De-risk
   first (below). Fallback: feed the diagnostic back as an explicit turn so "stuck"
   is unambiguous, making the swap near-deterministic while still agent-issued.
3. **Reproducibility** — stochastic runs need fixed seeds so the demo is stable, yet
   the "random failure" must reliably reproduce (engineer seeds so cheap reliably
   diverges and strong reliably fixes).
4. **Code execution path + sandbox** — which tool, what safety, how the transcript
   serializes to one report per panel.
5. **Problem selection + ground truth** — pick 3–6 from `bayesian-catelog`; define
   grading (compare to `gold`; or an objective metric like "final R̂ < 1.01 & 0
   divergences").
6. **Rename** the example dir `self_escalating_solver` → `bayesian_self_escalation`
   (and update `__init__` / module paths / README).

### ⚠️ De-risk BEFORE building the full harness
Author/adapt **one** Bayesian problem (e.g. Neal's funnel) and measure, live:
- Does the **cheap** model (write + run a sampler) **reliably fail** (diverge) where
  the **strong** model **reliably fixes** it? → confirms the capability ceiling.
- Given the divergence diagnostic, does the cheap agent **actually escalate**? →
  confirms meta-1.
Only build the 3-panel/timeline webpage once both hold.

---

## 7. Environment & how to run

- **Repo:** `/home/jp/rekursiv/sagent` (fork; origin `junpenglao/sagent`, upstream
  `rekursiv-ai/sagent`). Run everything via `uv run` inside this repo.
- **Branch:** **`demo/bayesian-self-escalation`**, branched from `upstream/main`
  (2026-06-24; the old `feat/AnthropicCLI` was stale — 235 files behind, lacked the
  thought-sig fix). Demo files are untracked here; **#211 thought-sig fix is native**
  → multi-turn Gemini tool-use works without any backport. **API drift to watch on this
  branch:** `json_freeze`/`JSON` moved to `sagent.lib.custom_json` (was `sagent.lib.json`);
  the model stream API is now `stream(request, publish=...)` (the offline `_Offline`
  mock in `solver.py` still uses the old `on_text`/`on_thinking` signature → MUST update
  before any offline run).
- **API key (Gemini):** `/home/jp/.config/sagent/google_api_key` (readable by user
  `jp`). **Read it inline, NEVER print it:**
  `GOOGLE_API_KEY="$(tr -d '[:space:]' < /home/jp/.config/sagent/google_api_key)" uv run …`
  The key is **live** — remind the user to **rotate it** and there's no longer a
  `/tmp/sagent_gkey` bridge. Do not echo keys anywhere.
- **Run a calibration:** `uv run python -m examples.self_escalating_solver.run
  --provider google --ids <ids> --conditions static-cheap,self-mutate`
- **Lint (CI runs both over `examples/`):** `uv run ruff check --no-fix .` and
  `uv run ruff format --check .`. Examples have per-file-ignores (ARG001, BLE001,
  E402, INP001, PLC0415, SLF001, T201…) but **not** Q000/ISC001/S311/PERF401 — keep
  strings triple-quoted, mark RNG `# noqa: S311`, etc.
- **Gemini transient `503`/`tool_use no blocks`** — handled by sagent's retry; not
  our bug.

### Key sagent API quick-ref
- `from sagent.providers import Google, Anthropic`; `Google.from_key(k)` /
  `Google.from_env()` (reads `GOOGLE_API_KEY`); `provider.model(id)`.
- `from sagent.tools import AgentSelf`; agent calls it with `{"model_id": "..."}`.
- `from sagent.agent import Agent`; `Agent(model=…, system=str, tools=[…],
  name=str, max_budget_usd=float)`; `async for ev in agent.run(UserMessage(text=…)): pass`;
  read result from `agent.history` (last `AssistantMessage.text`).
- Offline: `from sagent.testing import MockModelCaps`; `ModelResponse(message=…,
  tokens=TokenCount(input_tokens=…, output_tokens=…), stop_reason="model_finished")`
  — field is **`tokens=`**, not `usage=`.

---

## 8. Branch & commit

- The demo dir `examples/self_escalating_solver/` is **untracked** on
  `feat/AnthropicCLI`. Before committing, **create a dedicated branch** for the demo
  (preferably from a clean base, not mixed with the AnthropicCLI work):
  `git switch -c demo/bayesian-self-escalation`.
- Final PR drops `results.jsonl` / `summary.json` (run artifacts); keep a synthetic
  sample `web/data.js` so the page renders without a key.
- **Mutating git verbs require approval; no push to `main`; no force-push.**

---

## 9. Current files (code-tracing era — to be reworked for the pivot)
- `problems.py` — 30 code-tracing puzzles. **Will be replaced** by Bayesian problems.
- `solver.py` — 4 conditions, grading, `extract_swaps`, offline mock, `TIERS`.
  *Reusable scaffolding*; the per-condition `solve_*` bodies get reworked to
  write+run models and read diagnostics. (Self-mutate prompt is currently the
  mandatory-escalate variant.)
- `run.py` — CLI driver (`--provider/--ids/--conditions/--limit`) → jsonl/json/data.js.
- `make_sample.py` — synthetic sample data for the webpage (numbers are fictional).
- `web/index.html` — current 2-panel-ish frontier viz. **Rework to 3 panels +
  per-step timeline + swap highlight.**
- `README.md` — code-tracing era; rewrite for the Bayesian pivot.
