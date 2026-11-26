---
name: debug-prompt
description: "ALWAYS invoke this skill when isolating prompt/config-caused behavior through controlled A/B experiments over spawned agents, including repro validation, candidate fixes, and holdback. Do not tweak the prompt and eyeball results -- invoke first."
---

# Debug Prompt

Isolate a suspected prompt/config defect by measuring fresh agents'
behavior under controlled prompt variants, then prove a one-line fix
moves the rate.

## Core rule

**Reproduce the problem before you test any fix.** A fresh agent, in a
cold session, must exhibit the behavior at a high measured rate. Until
that happens you have no test, and every number a "fix" produces is
noise. This is the whole method; the phases below are just how to do it
carefully.

If the behavior happened -- in this session, a transcript, a user
report -- then it EXISTS. A cold session that fails to show it means
your repro task is wrong, not that the bug is absent. Rebuild the task
until it bites.

**One occurrence proves a repro exists, and bounds how hard it is to
find.** The argument is three steps and it is airtight:

1. The behavior occurred under some concrete input -- a session, a
   transcript, a report.
2. Therefore an input that elicits it EXISTS. In the worst case that
   input is the entire originating session, replayed turn by turn. That
   upper bound is always available and always bites; a *minimal* repro
   is merely a shrink of it.
3. Only once an input actually bites do you have a test, and only then
   can any fix be measured.

So a null control never licenses "the behavior is not in the config."
It licenses exactly one move: climb toward the known-good upper bound.
Replay the real turns verbatim -- same files, same corrections, same
order -- and once THAT bites, shrink it. Shrinking a working repro is
tractable; guessing a minimal one from scratch is not, and repeated
nulls are evidence you are guessing.

A "the behavior isn't in the prompt" conclusion drawn from a null
control is the single most expensive error in this method: it retires a
real defect and reverts a possibly-working fix on no evidence.

Then, and only then: a candidate fix must move that rate on
**held-back** trials the fix never touched. Your own in-session
behavior is an anecdote, not data -- it is contaminated by the very
context you are debugging.

Jargon, used throughout: **DV** (dependent variable) is what you
measure -- one number per trial, usually a yes/no ("did the agent do the
thing?"). **IV** (independent variable) is the one thing you change --
the prompt or config text under suspicion. **Control** is the unchanged
config; **treatment** is the config with a candidate fix. An **arm** is
one batch of trials under one condition.

## When to use / not

Use when the user says an agent "keeps doing X," resists a correction
repeatedly, or you suspect a system-prompt line, skill body, STYLE rule,
or sagent tool description is steering behavior wrongly.

## Protocol

Five phases. Do not skip phase 2 -- an unvalidated repro fixes nothing.

### 1. Design

- **One dependent variable (DV):** a binary or categorical parsed from a
  strict `VERDICT:` line (e.g. `VERDICT_DELEGATOR: YES|NO`). Define it so
  verbatim exclusions survive.
- **Who scores the DV -- decide deliberately.** A producing agent may
  **self-report** ONLY when the behavior is objective and the agent is
  not blind to it (e.g. "did you create a new `_helper`" -- structural,
  checkable). When the behavior is **subjective or self-blind** (verbosity,
  tone, over-hedging, "is this convincing") the producer will mis-score
  its own output -- it shares the exact blindness you are measuring. Then
  the DV MUST be scored by a **separate judge agent** that sees only the
  output, never the task framing.
- **Anchor a subjective DV with concrete examples.** Give the judge a
  matched pair -- same input, one exemplar of the behavior and one of the
  target -- so scoring matches against a sample, not the judge's guess.
  Anchor the **positive target**, not just the anti-pattern: without a
  model of "good," the judge (and any primed producer) drifts to a
  degenerate extreme.
- **Do not show the producer the DV or its anchors.** That primes it to
  perform the target and destroys the base rate (see Hermeticity).
- **One independent variable (IV):** the prompt/config text under
  suspicion. Everything else held constant.
- **Hold model + system prompt constant.** Spawn with NO `model_id` and
  NO `system` override so children inherit yours. Changing either turns
  an A/B into a confound (a cheaper worker tier is a different model, not
  a different prompt).
- **Task must be in the regime that triggers the behavior.** A behavior
  gated by task type will read 0% on the wrong tasks (verbosity does not
  fire on "is `sorted()` stable"; it fires on investigation-under-
  correction). Pick the task shape where the user actually saw it.
- **Start from the real turns; synthesize only to shrink.** The
  originating session is the one input KNOWN to elicit the behavior, so
  it is where a repro search starts, not a thing to avoid. Replay its
  turns verbatim into a fresh agent. Contamination is a real hazard --
  the answer may now sit in the worktree -- but the fix is to relocate
  the same shape (a generated copy of the tree under a tmp root, the same
  file contents at a different path), NOT to invent a different task.
  A synthetic case is a *shrink* of a repro that already bites; built
  first, it is a guess, and a guess that reads 0% teaches nothing.
- **Unprimed control:** the bare task with none of your framing. This
  cell is the base rate and the most important one.
- **Order causally-layered cells; a null upstream cell kills the ones
  below it.** When cells sit on a causal chain (a bare/base cell, then
  cells that add successive layers), run the most upstream cell first.
  If the behavior does not appear there, every cell downstream of it is
  moot -- the cause lies upstream, so stop rather than run them. Only
  branch downstream once an upstream cell shows the behavior can occur.
  Spend trials on the discriminating cell, not on cells a prior result
  has already settled.

### 2. Validate the repro

- Spawn the **unprimed control at n>=8** on pristine inputs.
- The behavior must reproduce at a **stable, high rate** (ideally
  >=80%). A coin-flip repro cannot measure a fix.
- **A control that does not reproduce is a BROKEN TEST, not a finding.**
  This is the single most common way a run is wasted, and it is
  seductive because a clean control reads like good news. It is not: a
  control at 0% measures nothing, and any treatment compared against it
  is noise. Do not report "the edits didn't help" off a null control --
  you have no test yet.
- **When the behavior DID happen -- in this session, in a transcript,
  in a user report -- the bug is KNOWN TO EXIST. A null control is then
  proof your task is wrong, never proof the bug is absent.** Do not
  stop. Rebuild the task until it bites, then measure fixes against it.
- **Recreate the CONDITIONS, not the topic.** A repro task must
  reproduce whatever made the behavior likely, which is usually
  something other than subject matter:
  - **Difficulty/headroom.** If the control succeeds every time, there
    is nothing for a fix to improve. Ceiling effects masquerade as "no
    effect." The task must be hard enough to fail.
  - **What the agent must BUILD.** Handing over a working harness,
    scorer, or scaffold removes the failure if the failure was in
    building one. To measure instrument bugs, the producer must write
    its own instrument.
  - **Pressure and provocation.** Behaviors that fire under challenge,
    correction, or repeated pushback need a task that challenges. A
    single neutral question will not elicit them.
  - **Breadth.** Wide, many-file, comparison-heavy work elicits search
    and shortcut behaviors that a 3-file task never will.
  - **An attractive wrong answer.** If the failure is committing early
    to a plausible-but-wrong cause, the task must contain one.
- **Escalate toward the full replay; do not sample task designs.** Each
  null costs an arm and teaches nothing on its own. Order the search so
  every step moves toward the known-good upper bound:
  1. Verbatim replay of the originating turns, full tool set, real
     config channels (an in-tree `AGENTS.md` walked from the agent's
     cwd, the `Skill` tool present so its catalog is in the prompt) --
     anything the live agent had, the producer gets.
  2. Once THAT bites, delete one condition at a time and re-measure.
     The first deletion that kills the rate names the cause.
  3. The surviving minimum is the case you persist in phase 6.
  Three different synthetic tasks each at 0/8 is not three findings; it
  is one finding -- you have not yet reached the real conditions.
- Only after a **verbatim replay** of the originating turns still reads
  0% may you consider (a) the config does not drive it or (b) your DV
  cannot see it -- wrong scorer (self-report on a self-blind behavior),
  wrong task regime, or an unanchored subjective rubric. Re-examine the
  DV against phase 1 before either conclusion. Anything short of the
  full replay leaves the upper bound untested, so the null is about your
  task, not the config. Stop; do not edit shared config.
- **Read the raw producer output before concluding anything.** A rate is
  a summary; the text says whether the DV saw what you meant. Dump 2-3
  full answers at the FIRST null, not after the third -- it is the
  cheapest diagnostic available and it distinguishes "behavior absent"
  from "judge blind" from "task off-regime" in one look.

### 3. Propose + test fixes (small n)

- **Trials run through the `prompt_debug` harness's `run_case`. Do not
  write your own runner.** A new DV means a new mode: a class with
  `parse(meta, sections, *, where)` and `async trial(case, run) -> bool`,
  in `prompt_debug/modes/` or any module a case names as
  `mode = "package.module:Class"`. The driver, retry, and concurrency
  logic are already correct in the harness, and a twin reintroduces their
  bugs. An arm that varies code, not a case file, is an experiment: a
  module with `run(arm, n=, model=)` under `prompt_debug/experiments/`.
- A fix is any config-text change that could move the DV -- not only a
  prohibiting **counter-line**. Candidates include a **positive
  exemplar** (show the target, e.g. a terse-but-sufficient sample), a
  reweighted/removed existing line, or a reordering. Draft 1-3; a
  principle usually beats a pattern-name.
- Run each as a new IV level at n=3 on **fresh pristine inputs**, scored
  by the same judge as the control.
- Keep any that visibly move the DV from the control rate.

### 4. Run (full n)

- Take the winning variant(s) to n>=10 on fresh inputs.
- Report a table: condition -> rate (control vs each treatment).

### 5. Holdback (prove it)

- Apply the fix to the actual config (STYLE/skill/tool).
- **Ship the EXACT winning IV string, byte-for-byte.** The phase-3/4
  winner is a specific string; the config you land must be that same
  string, not a paraphrase and not "the same idea via a different
  mechanism." Shipping `format_listing`'s output when the winner was a
  hand-written imperative reminder tests nothing -- they are different
  IVs. If you cannot land the exact string (e.g. it must be computed at
  runtime), that computed output IS a new IV: re-run phase 3 on it.
- **Run the REAL entry path, not a test harness.** The holdback must
  drive the actual production path (e.g. `Agent.run`), not a unit-test
  scaffold that bypasses tool wrappers, contextvars, or dispatch
  branches. A unit test can pass while the production path never invokes
  your hook (wrapped tools don't forward new methods; a different
  dispatch branch commits the first message). Only the real path proves
  the fix.
- **DV is the behavior, never the artifact.** "The reminder is present
  in history" / "the line is in the assembled prompt" is NOT the DV.
  Measure whether the agent *does the thing*. A fix can be mechanically
  live (injected, present, logged) and still move the behavior 0%.
- Spawn a **new** batch (n>=8) whose inputs and agents the fix
  development never touched, under the fixed config.
- The DV must land at the treatment rate, not the control rate.

Then restate the one-line finding and the exact edit made.

### 6. Persist as a regression guard

- Write the proven case to `prompt_debug/cases/<name>.md`: TOML
  frontmatter (`name`, `n`, `max_flag_rate`, `mode`, and the mode's own
  keys) plus `## producer_task`, an optional `## setup_turn`, any
  `## file: <path>` seeded into the trial's folder (`{DIR}` in a turn
  names it), and the mode's sections (`## judge_rubric` for `text`). Set
  the tolerance with slack above the proven rate (e.g. proven 0/8 ->
  `max_flag_rate = 0.25`).
- The `prompt_debug` CLI's `run` re-runs every case against the REAL
  system prompt the sagent CLI assembles (`default_system_prompt` plus the
  AGENTS.md and environment it tells) and exits with how many failed, so
  a later edit that reintroduces the behavior fails; `list` shows the
  cases. The integration test `prompt_debug/regression_test.py` runs the
  same.
## Running a live run

- **Run FOREGROUND with a hard `timeout`, print the result inline.** Do
  NOT `nohup ... &` then poll a logfile: backgrounded runs get orphaned,
  a later `pkill` spawns confusion, `-15`s lose results, and you cannot
  tell which process wrote what. One `timeout N uv run python script.py`
  that ends by `print`ing `RESULT: k/n` is reliable; a background+poll
  loop is where hours vanish.
- **NEVER `sleep` to wait for a job. Gate on the event: `wait <pid>`.**
  This applies to the one case where backgrounding IS right -- running
  several arms concurrently. Capture each pid and `wait` on them:

  ```bash
  uv run python run.py --variant control > a.log 2>&1 &  P1=$!
  uv run python run.py --variant treat   > b.log 2>&1 &  P2=$!
  wait $P1 $P2        # returns when they finish, not when you guessed
  ```

  A `sleep N` is a guess about a duration you do not control: too short
  and you read a truncated log as a result, too long and you idle. The
  general rule is broader than jobs -- gate on the completion event,
  never on an estimate of it. Reach for `wait`, a blocking read, or a
  status poll that terminates; never a fixed delay.
- Each trial is a full multi-turn agent (~13s), so an 8-trial arm is
  1-2 min. Size the `timeout` accordingly and let it finish; don't kill
  and retry mid-run.
- For a long visible run, pass `run_case(..., progress=cb)` -- `cb(done,
  total, verdict)` fires per completed trial:

  ```python
  def cb(d, t, v):
      print(f"{d}/{t} verdict={v}", flush=True)
  ```

- Build the model exactly as the sagent CLI does, from a catalog kept open
  for the run: `async with ModelCatalog() as catalog: model = catalog[provider]["default"]`.
  Keep `max_concurrency<=2`; higher flakes the transport into
  `ReadError` exhaustion.

## Known findings (don't re-derive)

- **System prompt is ignored; the conversation stream is obeyed.** A
  directive/catalog in the system prompt (or a tool's `prompt()` -> the
  system field) is treated as ambient: skills placed there fire ~0%. The
  same text injected into the first USER turn (`<system-reminder>`)
  fires far higher. Placement is the dominant lever.
- **Full catalog caps ~50%, even in-conversation, even imperative.** A
  generic "here are all N skills, invoke when applicable" listing plateaus
  around 50% regardless of wording or placement. Naming the SPECIFIC
  matched skill imperatively ("skill X applies; you MUST invoke it before
  acting") reaches ~100%. So reaching the ceiling needs **match-gating**
  (inject only the matched skill), not just relocation -- and matching
  requires a matcher (keyword for structural triggers, embedding/model
  for semantic ones).
- **"Motivation" framing beats bullet rules; naming the action beats
  generic appeals.** "Don't enumerate; commit to one cause" (naming the
  action) works; "honor your system prompt" (generic) is 0%.
- **When code made the behavior, the control is that code path.** Agents
  reportedly pruned the STYLE.md they were told to follow. Asked to prune,
  a fresh model kept it 8/8 ("kept for future edits"): no prompt drove it.
  The old pruner's cost rule, which asked no model, dropped it 8/8; the
  fix (the pruner notes, the model queues) dropped it 0/10. Read each
  trial's stated reason before trusting a DV: the first DV counted
  AGENTS.md drops, which lose nothing because AGENTS.md is also in the
  system prompt. Runnable example and living record: the CLI's
  `experiment prune_note <arm> <n>` (`prompt_debug/experiments/prune_note.py`,
  the `prune` mode, and the `prune-keeps-rules` case that guards it).
## Hermeticity (mandatory, or the run is garbage)

- **Put all scratch OUTSIDE the repo.** Write inputs, judge rubrics,
  and outputs under a fresh `mktemp -d` (e.g. `/tmp/…`), never in the
  worktree -- experiment files are not deliverables and must not litter
  or be committed. Delete the dir when done.
- **One fresh input file per agent, used once.** Never `cp` from a file
  another agent may have already mutated -- that leaks a prior agent's
  output as this agent's input. Write each from a canonical template.
- **`system`/`model_id` overrides are per-spawn and copied** (verified:
  `agent_spawn.py` `_resolve_system` reads, never writes parent). So a
  per-child override cannot leak to siblings or you -- but a **shared
  worktree file can**. Files are the only non-hermetic surface.
- **Discard contaminated cells.** A child that reports "already done, no
  edit needed" saw a pre-mutated file; drop it, regenerate, re-run.
- Transient `RetriesExhaustedError` = drop that trial, not the cell.

## Stop conditions

- Repro rate <~50% in phase 2 -> the TASK is wrong; rebuild it and
  re-measure, escalating toward a verbatim replay of the originating
  turns. Only conclude "not a config issue" when the behavior was
  never observed in the wild. If it was observed, it exists, and a null
  control is a broken test.
- **Never conclude "the behavior isn't in the config" while a
  known-eliciting input remains untried.** One observation proves an
  eliciting input exists; the originating session is an upper bound on
  it. Until that bound has been replayed and still reads 0%, the null is
  about your task. This is the conclusion that retires real defects, so
  it needs the strongest evidence, not the weakest.
- Fix does not move the DV on holdback -> **STOP patching the payload.**
  A failed holdback means the IV was wrong or the shipped string differs
  from the tested winner (phase 5). Do NOT keep tweaking wording in a
  loop -- return to phase 1 and name a new IV hypothesis. Iterating
  payloads past a holdback failure is the resistance pattern.
- Ask before editing shared config (STYLE/skill/tool) if the holdback
  has not yet run.
- Never leave shared config (skill.py, prompt.md, a tool) edited-and-
  holdback-failing between turns. Either the fix passes holdback, or you
  revert it -- a broken half-applied fix in the worktree is worse than
  none.

## Common mistakes

| Mistake | Fix |
|---|---|
| Varying model tier "to save cost" | Hold model constant; confound otherwise |
| Reusing the original task file in place | Relocate the same content to a tmp root; do NOT switch to a different task |
| Inventing a synthetic task before any repro bites | Replay the real turns first; synthesize only to shrink a working repro |
| `cp` inputs between trials | One fresh template-written file per agent |
| Skipping the unprimed control | No base rate = no measurable effect |
| Self-reporting a self-blind DV (verbosity, tone) | External judge sees output only; anchor with examples |
| Reading 0% and concluding "no bug" | Rule out a broken DV/wrong task regime first |
| Several synthetic tasks each 0/n -- "the behavior isn't in the config" | One finding, not several: you never reached the real conditions. Replay the originating turns |
| Summarizing rates without reading any producer text | Dump 2-3 full answers at the FIRST null |
| Control reproduces 0/n; comparing treatments anyway | Null control = no test; every treatment number is noise |
| Control succeeds 6/6 -- "no headroom, edits don't help" | Ceiling effect; task too easy. Make it hard enough to fail |
| Handing the producer a working harness/scorer | Removes the failure if the failure was BUILDING one |
| Repro task shares the topic but not the conditions | Recreate difficulty, breadth, pressure, and what must be built |
| Anchoring only the anti-pattern | Anchor the positive target too, or the judge drifts |
| Declaring fixed after phase 3 | Holdback (phase 5) is the only proof |
| Shipping a paraphrase of the winning IV | Land the exact tested string; a computed variant is a new IV |
| DV = "artifact is present" | DV = the agent DID the behavior; present-but-ignored is 0% |
| Holdback via a unit-test harness | Run the real entry path (`Agent.run`); harnesses skip wrappers |
| Patching payloads after a failed holdback | Stop; return to phase 1 with a new IV |
| Citing your own session as data | Fresh spawn only; your context is the bug |
| `sleep N` to wait for a background arm | `wait $PID`; gate on the event, never a guessed duration |
| Writing a new trial runner instead of using the harness | Write a mode or an experiment; the harness owns retries and concurrency |
| `nohup ... &` inside ONE tool call | The call blocks on the shell's children anyway -- backgrounding hides failures without saving time |
