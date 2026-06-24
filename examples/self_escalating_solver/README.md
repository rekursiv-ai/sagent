# Self-escalating solver — cross-vendor agent-directed model mutation

A runnable [sagent](https://github.com/rekursiv-ai/sagent) example of the one thing
sagent does that other agent stacks don't: **an agent upgrades its own model
mid-task — across vendors — keeping all of its working memory.**

One agent starts on a cheap Google model (`gemini-2.5-flash-lite`). It writes a
Metropolis-Hastings sampler, runs it, and a black-box grader replies `FAIL`. It
can't crack the bug, so it calls `AgentSelf` to promote *itself* to an Anthropic
model (`claude-sonnet-4-6`) — Google → Anthropic, mid-run — re-derives the fix, and
passes. Three arms make the contrast:

| arm | model(s) | outcome |
|---|---|---|
| **low-tier** | Gemini Flash-Lite, pinned | flails and `FAIL`s — can't crack it alone |
| **high-tier** | Claude **Opus**, pinned | solves it, but you pay the priciest model on every task |
| **self-mutate** | Gemini Flash-Lite → upgrades itself to Claude **Sonnet** | solves it, **cheaper than always-Opus** |

## Run it

```bash
# 1. Replay the captured run in a local webpage (no API key needed):
uv run python -m examples.self_escalating_solver.run
#    -> serves http://localhost:8000 and prints an `ssh -L` line for remote viewing

# 2. Re-run it live (prompts for provider: cross | google | anthropic):
uv run python -m examples.self_escalating_solver.run --live
#    cross = Gemini -> Claude (the cross-vendor demo).
#    skip the prompt with:  --live --provider cross --trials 4
```

The webpage replays each arm step by step: the cheap model's repeated `FAIL`s, the
glowing **cross-vendor self-upgrade** banner, then the strong model's fix → `PASS`.
Reasoning collapses to a one-line preview — click `[…]` to expand.

## The problem (and why it's a fair test)

Sample from the target whose unnormalized density is `f(x) = x·e^(-x/2)` for `x > 0`
(a Gamma, true mean 4) using a **multiplicative** proposal `x' = x·exp(ε)`. The
textbook symmetric-Metropolis rule `min(1, f(x')/f(x))` is **wrong** here — the
proposal is asymmetric, so a correct sampler needs the Metropolis-**Hastings**
correction (the `x'/x` Jacobian). Drop it and you silently sample an Exponential
(mean ≈ 2), not the Gamma. It's a classic trap a non-expert model falls into.

Grading is a **black-box `check(samples)`** injected into the sandboxed `run_python`
tool: it returns only `PASS`/`FAIL` — no hint, no ground truth. The agent is graded
on the grader's verdict, never its own self-report (a weak model will fabricate
success otherwise).

## How the cross-vendor swap works

`AgentSelf` swaps from the agent's `model_spec`. Given just
`model_id="claude-sonnet-4-6"`, sagent's `infer_provider` maps `claude-*` →
Anthropic, the provider allow-list permits it, and the Anthropic provider is built
from `ANTHROPIC_API_KEY`. The driver sets that env var **in its own process only**
(never the CLI's), reading the key from a file. The Agent must be constructed with a
`ModelSpec` — without one, the swap fails with *"Agent has no model spec; cannot
swap."*

## Keys

API keys live in **files**, not your shell environment, so they never leak into a
Claude CLI subscription:
- `~/.config/sagent/google_api_key`
- `~/.config/sagent/anthropic_api_key`

## Honest notes

- Rates vary by model and prompt; Gemini is stochastic and 503s under load. The
  shipped `web/data.js` is a clean run (low `FAIL`, high `PASS`, 3/4 self-upgrade). A
  live re-run won't always land that cleanly — re-run a couple of times.
- This is a **toy** chosen so the mechanism is visible — not a benchmark. The
  cheap-vs-strong gap is real, but it's a single hand-picked trap.

## Files

- `solver.py` — the three conditions, the sandboxed `run_python` tool, the black-box
  grader, and timeline capture.
- `run.py` — the driver: replay server by default, `--live` to re-run, the
  `{google, anthropic, cross}` provider configs, and `build()` (Model + ModelSpec).
- `web/index.html` + `web/data.js` — the self-contained, replayable 3-panel report.
- `WORKLOG.md` — the full lab notebook: every dead end we hit and what finally
  worked, so this can be picked up and improved without re-deriving the path.
