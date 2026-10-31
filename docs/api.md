# Python API

Sagent is a Python runtime for typed agent loops. The CLI is a thin surface over the same `Agent`, `Model`, `Provider`, and `Tool` contracts.

## Minimal agent

```python
import asyncio

from sagent.agent.agent import Agent
from sagent.providers.catalog import ModelCatalog
from sagent.tools.glob_tool import Glob
from sagent.tools.grep import Grep
from sagent.tools.read import Read


async def main() -> None:
    async with ModelCatalog() as catalog:
        async with Agent(
            model=catalog["google"]["gemini-pro-3.1"],
            system_prompt="You summarize files concisely.",
            tools=[Read(), Glob(), Grep()],
        ) as agent:
            print(await agent.send("Summarize README.md"))


asyncio.run(main())
```

`async with Agent(...)` starts the runtime; `send()` returns the reply's text once the agent is idle.

## Agent constructor

```python
agent = Agent(
    model=model,
    tools=(),
    system_prompt=None,
    journal=None,
    compactor=NoShrinker,
    pruner=NoShrinker,
    tool_context=None,
    max_rounds=None,
    max_spend_usd=None,
    persistent_retry=False,
    agents_md=None,
    allowed=None,
    environment=True,
    account=None,
)
```

Every argument is keyword-only.

| Argument | Meaning |
| --- | --- |
| `model` | Required `Model`; assign `agent.model` later to switch. |
| `tools` | Tools the model may call; names must differ. |
| `system_prompt` | What the model is told it is; `None` uses the default prompt. |
| `journal` | Where the session is written; `None` keeps nothing. Open a `JsonlJournal` yourself and pass it. |
| `compactor` | Factory for what summarizes the session, usually `SummaryCompactor`. |
| `pruner` | Factory for what drops old tool results, usually `Pruner`. |
| `tool_context` | Working directory, result size limit, terminal width, and fetch session; `ToolContext()` when omitted. |
| `max_rounds` | Tool-call rounds one turn may run; `None` for no cap. |
| `max_spend_usd` | USD the agent may spend; `None` for no cap. |
| `persistent_retry` | Wait out a long rate limit instead of raising. |
| `agents_md` | Where `AGENTS.md` files are looked for; defaults when omitted. |
| `allowed` | Tool names the model may call; `None` for all. The rest are still offered, so the prompt cache holds. |
| `environment` | Tell the model its directory, git state, platform, and shell. |
| `account` | Named login for the model's provider; `None` for its default. |

## Running agents

```python
reply = await agent.send("Do the task")

async for words in agent.stream("Now explain it"):
    print(words, end="")
```

- `send(text, attachments=())` returns the last text the model wrote in the turn.
- `stream(text, attachments=())` yields the reply's text as it is written.
- Both raise `BudgetExhaustedError` past `max_spend_usd` and `RoundLimitError` past `max_rounds`.
- `attachments` takes `Attachment(mime_descriptor=..., data=...)` images and documents.

Every runtime event -- thinking, tool calls, tool results, compactions -- goes to `agent.runtime`. See [Streaming](streaming.md).

## Observable state

| Attribute | Meaning |
| --- | --- |
| `model` | Current model; assign to switch at the next model call. |
| `system_prompt` | Assign to change what the model is told it is. |
| `tools` | Tools offered, by name. |
| `allowed` | Tools the model may call; assign to change. |
| `session` | The conversation: iterate it, `append()` records, `clear()`, and reach its `journal`, `compactor`, and `pruner`. |
| `runtime` | What runs the turns: `observers`, `async for event in agent.runtime`, `halt()`, `jobs`. |
| `spend_usd` | USD the model calls have cost, a resumed session's included. |
| `max_rounds`, `max_spend_usd` | The caps; assign to change. |
| `running` | Whether the runtime serves turns. |
| `root` | Where the session started. |

## Models and providers

A `ModelCatalog` holds every provider, sharing one HTTP client and process pool. Looking a model up does no I/O.

```python
async with ModelCatalog() as catalog:
    model = catalog["google"]["gemini-pro-3.1"]
    utility = catalog["google"]["utility"]
```

A provider can also be used alone:

```python
from sagent.providers.google.api import Google

async with Google() as google:
    model = google["gemini-pro-3.1"]
```

A `Model` exposes:

- `capability`: every setting it offers, its limits, and its prices.
- `settings`: the settings chosen; assign to change them.
- `count(request)`: tokens a `Request` would send.
- `stream(request)`: yields `TextChunk`s, then the finished `Response`.

```python
from dataclasses import replace

model.settings = replace(model.settings, thinking_effort="high")
```

`Request` holds system prompt blocks, session records, and tools. `Response` holds the reply's records, token counts, spend, and stop reason.

## Session records

A session is a sequence of frozen dataclasses from `sagent.session.types`: `UserMessage`, `AssistantMessage`, `Thinking`, `ToolCall`, tool results such as `ShellCommandResult` and `UncategorizedToolResult`, `SystemMessage`, and more. `JsonlJournal` writes one record per JSON line.

## Tools

A tool has a `name`, `description`, JSON `schema`, and `display`, and implements:

```python
def summary(self, arguments: Mapping[str, PlainTree]) -> str: ...
def serialize_key(
    self, context: ToolContext, arguments: Mapping[str, PlainTree]
) -> str | None: ...
async def call(
    self, context: ToolContext, arguments: Mapping[str, PlainTree]
) -> AnyToolResult: ...
```

See [Tools](tools.md) for built-ins and custom-tool examples.

## Costs

Each `Response` carries its `TokenCount` and `TokenCost`. `agent.spend_usd` sums them across the session, children's spend included. With `max_spend_usd` set, the turn halts and raises `BudgetExhaustedError` once spend reaches the cap.
