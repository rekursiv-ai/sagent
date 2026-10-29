# Compaction and large outputs

Sagent keeps long-running sessions within model context limits by combining full conversation compaction, tool-result microcompaction, and large-output persistence.

## Full conversation compaction

The default CLI enables automatic compaction. Disable it with:

```bash
sagent --no-compact
```

In Python, pass a compactor factory; the agent builds it with itself:

```python
from sagent.agent.agent import Agent
from sagent.agent.compactor import SummaryCompactor

agent = Agent(model=model, tools=tools, compactor=SummaryCompactor)
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

Prompt-too-long recovery drops older grouped rounds while preserving tool-use/tool-result pairing.

## Manual compaction

Agents with the `Session` tool can request compaction, with what it must keep:

```text
Session(action="compact", prompt="focus on API decisions")
Session(action="compact", style="session-audit")
```

## Microcompaction

Microcompaction clears old tool results that no longer need exact text in context. It is controlled by each tool's `supports_microcompaction` flag.

Good candidates:

- large shell output;
- grep/list/search output that can be reproduced;
- web and paper search results;
- file write/edit confirmations.

Poor candidates:

- child-agent final reports;
- messages sent to external services;
- skill bodies needed verbatim.

Sagent waits for old tool results to leave the prompt-cache hot path, preserves recent clearable results, and invalidates read-cache entries when old `Read` results are cleared.

## Tool-result persistence

Large tool outputs can exceed context budgets even before full compaction. Sagent handles this before the next model request.

Defaults:

| Limit | Value |
| --- | --- |
| Generic text result truncation | 400,000 characters |
| Per-tool persistence threshold | 50,000 characters |
| Persisted preview size | 2,000 characters |
| Aggregate per-message tool-output budget | 200,000 characters |

Persisted results are replaced in context with a preview and path:

```text
<persisted-output>
Output too large (...). Full output saved to: ...

Preview (...):
...
</persisted-output>
```

Storage location:

- `session_dir/tool-results` when a session is active;
- `/tmp/sagent_results` for no-session runs.

Persisted files are written with owner-only permissions.

`Read` is exempt because it already enforces line/page limits.

## Context budgets

`ContextBudget.from_model(model)` derives request, response, buffer, reattachment, persistence, and message budgets from model limits.

You can pass an explicit `ContextBudget` to `Agent(...)` or adjust limits at runtime:

```python
agent.max_request_tokens = 200_000
agent.max_response_tokens = 8_192
```

Agents can also call:

```text
AgentSelf(max_request_tokens=200000)
```

Explicit limits are validated against the model's advertised maximums.

## Costs

Sagent records token counts and model costs for every response. `Agent.total_cost_usd` returns cumulative session cost. `Agent.cache_tokens` returns cache creation/read totals.

Use a hard run budget from the CLI:

```bash
sagent --max-budget-usd 1.00
```

Or in Python:

```python
agent = Agent(model=model, max_budget_usd=1.00)
```

Once cumulative cost reaches the cap, Sagent raises instead of making another paid call.
