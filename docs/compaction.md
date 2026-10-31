# Compaction and large outputs

Sagent keeps long-running sessions within model context limits by combining full conversation compaction, tool-result pruning, and bounded tool results.

## Full conversation compaction

The default CLI enables automatic compaction. Disable it with:

```bash
sagent --no-compact
```

In Python, pass a compactor factory; the agent builds it with itself:

```python
from functools import partial

from sagent.agent.agent import Agent
from sagent.agent.compactor import SummaryCompactor

async with Agent(model=model, tools=tools, compactor=SummaryCompactor) as agent:
    await agent.session.compactor.run("keep the API design")  # Or compact now.

Agent(model=model, compactor=partial(SummaryCompactor, keep_recent=6))
```

`SummaryCompactor` compacts before a model call once the next request, past
its system prompt, reaches the largest body `B*` whose summary call still
fits the window:

```text
S(B) = B · 0.05·(10⁶ / clip(B, 62.5k, 10⁶))^¼     # room held for the summary
B*   = max{B : B + s + p ≤ R,  B + s + p + S(B) ≤ C}
```

`s` is the system prompt, `p` the summary prompt, `R` the input limit, and
`C` the window input and output share. `S(B)` assumes a summary needs at most
10% of its body, 8.4% at 128k, and 5% from 1M on; `summary_tokens` takes the
cap, the floor, the power, and the context where the floor starts. A model id with a size, such as `opus+200k`,
caps the window, so the trigger falls with it.

The next request's size is the last reply's billed count, plus an estimate of
what came after it. With no bill yet, as in a new or resumed session, it is an
estimate of the context and the tools.

The summary replaces the context; the session's journal keeps every window.

## What is preserved

Compaction preserves the useful state needed to continue:

- task status and active goals;
- user instructions and constraints;
- relevant files and decisions;
- recent messages when configured (`keep_recent`), and the user's unanswered turn;
- the skills the window loaded, their bodies read again while they fit;
- the background jobs still running.

A model call that overflows the window starts a compaction, then calls again. A summary call that overflows first halves the cap on long tool results, then drops the oldest fifth of the rounds, and asks again. A round starts at a user message, so a dropped round never orphans a tool result.

## Manual compaction

Agents with the `Session` tool can request compaction, with what it must keep:

```text
Session(action="compact", prompt="focus on API decisions")
Session(action="compact", style="session-audit")
```

## Pruning

Pass `Pruner` to drop old tool results instead of summarizing them:

```python
from sagent.agent.pruner import Pruner

async with Agent(model=model, tools=tools, pruner=Pruner) as agent:
    await agent.session.pruner.run()  # Or prune now, whatever it costs.
```

Before each model call, the pruner replaces a result with a stub once keeping it has cost more than fetching it again, and only when dropping the dead results pays for breaking the prompt cache. A drop that keeps the context under the compaction trigger is taken instead of compacting.

`agent.model.settings.pruning`, a `PruningPolicy`, says which results may go:

- `keep_recent`: the newest results, never pruned (3).
- `keep_tools`: tools whose results are never pruned (`Skill`).
- `expiry`: how much rent an unused result may cost before it expires.

A model that prunes on the server, with `manage_context_server_side`, is left to do it there.

## Tool-result size

Each tool result is cut to fit `ToolContext.max_result_chars`, 50,000 characters by default; a tool may hold its own to less. When the agent has a journal file, `Bash` writes a cut command's whole output beside it and names the file in the result.

## Context window

A model's window is `model.capability.limits`; the size chosen is `model.settings.context_size`. A model name with a size tag, such as `opus+200k`, picks a smaller window, and the compaction trigger falls with it:

```python
model = catalog["anthropic"]["opus+200k"]
```

## Costs

Sagent records token counts and model costs for every response. `Agent.spend_usd` returns cumulative session cost, spawned children's included.

Use a hard run budget from the CLI:

```bash
sagent --max-budget-usd 1.00
```

Or in Python:

```python
agent = Agent(model=model, max_spend_usd=1.00)
```

Once cumulative cost reaches the cap, the turn halts and `send()` raises `BudgetExhaustedError` instead of making another paid call.
