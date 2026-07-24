# Python API

Sagent is a Python runtime for typed agent loops. The CLI and Slack service are thin surfaces over the same `Agent`, `Model`, `Provider`, `Tool`, and `Message` contracts.

## Minimal agent

```python
import asyncio

from sagent import tools
from sagent.agent import Agent
from sagent.lib.custom_json import json_freeze
from sagent.providers import Google


async def main() -> None:
    agent = Agent(
        model=Google.from_env().model("gemini-3.1-pro-preview"),
        system="You summarize files concisely.",
        tools=[tools.Read(), tools.Glob(), tools.Grep()],
    )
    result = await agent.run(json_freeze({"prompt": "Summarize README.md"}))
    print(result.content)


asyncio.run(main())
```

`Agent.run()` accepts a JSON directive with a `prompt` field and returns a `Message`.

## Agent constructor

```python
agent = Agent(
    model=model,
    model_spec=None,
    system="",
    tools=[],
    compactor=None,
    name="Agent",
    description="An AI agent.",
    max_tool_call_rounds=None,
    max_attempts=5,
    thinking="adaptive",
    effort=None,
    session_dir=None,
    budget=None,
    max_budget_usd=None,
    persistent_retry=False,
    track_changed_files=True,
)
```

| Argument | Meaning |
| --- | --- |
| `model` | Required `Model` backend. |
| `model_spec` | Recipe for rebuilding/switching the model, used by agent tools. |
| `system` | String or dict of named system prompt sections. Dict values may be strings or zero-arg callables. |
| `tools` | List of tool objects. Duplicate `tool_id` values are rejected. |
| `compactor` | Conversation compactor, usually `SummaryCompactor`. |
| `name` | Live-agent registry and UI name. |
| `description` | Human-facing description when an agent is used as a tool. |
| `max_tool_call_rounds` | Per-run cap on model requests that include tool calls. |
| `max_attempts` | Retry cap for transient model failures. Must be at least 1. |
| `thinking` | Provider-specific thinking mode. `None` disables explicit thinking config. |
| `effort` | Provider-specific effort level. Rejected if the model does not support effort. |
| `session_dir` | Directory for persisted session state and tool-result storage. |
| `budget` | Explicit `ContextBudget`; validated against model limits. |
| `max_budget_usd` | Cost cap for this agent's session. |
| `persistent_retry` | Keep retrying persistent runs after recoverable failures. |
| `track_changed_files` | Track file changes for safety reminders. |

## Running agents

`run(directive, events=None)` runs one prompt to completion. `events` is an optional `asyncio.Queue[Message | None]` for streamed model text, thinking, tool labels, tool results, and completion metadata. `None` is a request-boundary sentinel.

```python
events: asyncio.Queue[Message | None] = asyncio.Queue()
result = await agent.run(json_freeze({"prompt": "Do the task"}), events=events)
```

`run_forever(events=None)` drains `agent.inbox`, joins queued strings into prompts, calls `run()`, survives cancellation, and exits when it receives Sagent's quit sentinel. The REPL and Slack service use this shape for long-lived agents.

The inbox is the coordination point for REPL input, background tool completion, and peer-agent messages. Prefer putting text into the inbox over mutating history directly.

## Observable state

Important public properties:

| Property | Meaning |
| --- | --- |
| `messages` | Mutable conversation history. |
| `tools` | Tool set available to this agent. |
| `model` | Current model backend. |
| `model_spec` | Provider/auth/model/account recipe when known. |
| `budget` | Active `ContextBudget`. |
| `max_request_tokens` | Request token limit; setter validates model max. |
| `max_response_tokens` | Response token reserve; setter validates model max. |
| `total_cost_usd` | Cumulative subtree model cost for this session (includes descendant agents; live during active runs). |
| `total_tokens` | Cumulative subtree `TokenCount` for this session (live during active runs). |
| `total_active_elapsed_seconds` | Cumulative wall-clock seconds spent in `run()` across the session (live-ticks while active). |
| `cache_tokens` | `(cache_creation_tokens, cache_read_tokens)` from `total_tokens`. |
| `cache_ttl` | Prompt-cache TTL for outgoing requests (`"5m"` default or `"1h"`); setter validates. |
| `session_id` | Stable short session ID. |
| `status` | Current status string. |
| `active` | True while `run()` is executing. |
| `last_elapsed` | Wall-clock seconds for the most recent run. |
| `last_model_request_tokens` | Last request token count. |
| `last_model_response_tokens` | Last response token count. |

`swap_model(model, spec=...)` replaces the active model and its model recipe. `AgentSelf(model_id="...")` uses the same mechanism.

## Messages

Messages are frozen dataclasses with common metadata: `parent_id`, `id`, and `timestamp`.

| Type | Content |
| --- | --- |
| `TextMessage` | `str` content with a text descriptor. |
| `BytesMessage` | `bytes` content for binary data. |
| `JsonMessage` | JSON content. |
| `MultipartMessage` | Tuple of nested messages. |

`Message` is the union of these four types. Serialization stores `descriptor`, `content`, `_id`, `_parent_id`, and `_timestamp`; bytes are base64 encoded and multipart messages serialize recursively.

Common descriptors:

- `text/plain`: ordinary text.
- `text/x-error`: expected tool or runtime error.
- `text/x-thinking`: model thinking stream.
- `text/x-tool-label`: human-readable tool status.
- `multipart/x-tool-result`: tool result message.
- `application/x-done`: run completion metadata.

## Models and providers

A `Provider` constructs `Model` objects:

```python
provider = Google.from_env()
model = provider.model("gemini-3.1-pro-preview")
utility = provider.utility_model()
```

A `Model` exposes:

- `buffer(request) -> ModelResponse`
- `stream(request, on_text=None) -> ModelResponse`
- `is_context_overflow(error) -> bool`

`stream()` returns a complete `ModelResponse` while calling `on_text(chunk)` for streamed text chunks.

`ModelRequest` contains messages, tools, system prompt, token limits, thinking/effort settings, and request metadata. `ModelResponse` contains content messages, stop reason, token counts, cache counts, response IDs, and request/response/total cost.

## Tools

Tools receive a tool-call `Message`, parse its JSON directive, and return a `Message`.

Batch tools implement:

```python
async def run(self, msg: Message) -> Message: ...
```

Streaming tools implement:

```python
streaming = True


def run(self, msg: Message) -> AsyncGenerator[Message, None]: ...
```

For streaming tools, all yielded messages before the last are forwarded as events. The final yield is the tool result.

See [Tools](tools.md) for built-ins and custom-tool examples.

## Cost and context budgets

`Pricing` is per-million-token USD and tracks request, response, cache write, and cache read prices. `TokenCount` tracks input, output, cache creation, and cache read tokens.

`ContextBudget.from_model(model)` derives conservative request/response/headroom defaults from the model limits. Explicit budgets are validated when the agent is constructed and when token-limit properties are changed.

If `max_budget_usd` is set, Sagent records every model response and raises once cumulative session cost reaches the cap.
