# sagent🪄

<p align="center">
  <img alt="sagent logo" src="assets/logo-custom.webp" width="180">
</p>

<p align="center">
  Typed Python library and CLI for multi-provider, multi-agent LLM applications.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.12-blue.svg">
  <a href="https://pypi.org/project/sagent/"><img alt="PyPI" src="https://img.shields.io/pypi/v/sagent.svg"></a>
  <a href="https://github.com/rekursiv-ai/sagent/actions"><img alt="CI" src="https://img.shields.io/badge/ci-GitHub%20Actions-blue.svg"></a>
  <a href="https://github.com/rekursiv-ai/sagent/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue.svg"></a>
</p>

<p align="center">
  <a href="https://github.com/rekursiv-ai/sagent/blob/main/docs/tutorial.md">Tutorial</a>
  ·
  <a href="https://github.com/rekursiv-ai/sagent/blob/main/docs/concepts.md">Concepts</a>
  ·
  <a href="https://github.com/rekursiv-ai/sagent/blob/main/docs/providers.md">Providers</a>
  ·
  <a href="https://github.com/rekursiv-ai/sagent/blob/main/docs/tools.md">Tools</a>
  ·
  <a href="https://github.com/rekursiv-ai/sagent/blob/main/docs/cli.md">CLI</a>
  ·
  <a href="https://github.com/rekursiv-ai/sagent/blob/main/docs/sessions.md">Sessions</a>
  ·
  <a href="https://github.com/rekursiv-ai/sagent/blob/main/docs/security.md">Security</a>
  ·
  <a href="https://github.com/rekursiv-ai/sagent/blob/main/docs/architecture.md">Architecture</a>
  ·
  <a href="https://github.com/rekursiv-ai/sagent/blob/main/docs/api.md">API</a>
  ·
  <a href="https://github.com/rekursiv-ai/sagent/blob/main/docs/streaming.md">Streaming</a>
  ·
  <a href="https://github.com/rekursiv-ai/sagent/blob/main/docs/compaction.md">Compaction</a>
  ·
  <a href="https://github.com/rekursiv-ai/sagent/blob/main/docs/slack.md">Slack</a>
  ·
  <a href="https://github.com/rekursiv-ai/sagent/tree/main/examples">Examples</a>
</p>

```python
from sagent import tools
from sagent.agent import Agent
from sagent.lib.json import json_freeze
from sagent.providers import Google

agent = Agent(
    model=Google.from_env().model("gemini-3.1-pro-preview"),
    system="You are a scientist.",
    tools=[tools.Read(), tools.Glob(), tools.Grep()],
)
result = await agent.run(json_freeze({"prompt": "analyze the CSV in ./data/"}))
print(result.content)
```

## Why sagent exists

Most serious coding agents are CLIs, editor extensions, hosted
assistants, or non-Python runtimes. Sagent gives you the agent runtime
as typed Python objects you can import, compose, test, and embed. The
CLI is one entry point over the library, not the center of the design.

Use Sagent when you want:

- a Python API first and a CLI second;
- provider swapping without changing the agent loop;
- custom tools as normal Python objects;
- session persistence and compaction;
- child agents and peer messaging for review, delegation, and map-reduce work.

Three pieces make Sagent distinctive:

- **Hot-swappable providers.** The same agent, tools, session, and
  compactor can run against Anthropic, OpenAI, Google, Moonshot,
  DashScope, MiniMax, or an OpenAI-compatible endpoint.
- **Multi-agent primitives.** `AgentSelf`, `AgentSpawn`, and
  `AgentSend` let agents inspect themselves, spawn isolated children,
  and send messages to named peers.
- **Typed runtime objects.** `Agent`, `Tool`, `Message`, `Model`, and
  `Provider` are Python protocols and dataclasses that can be used
  directly from application code.

## What sagent does

- Runs agents against Anthropic, OpenAI, Google, Moonshot, DashScope, MiniMax, and OpenAI-compatible endpoints.
- Exposes tools for local files, shell commands, web access, paper search, and agent coordination.
- Keeps the same `Agent` behind CLI, Slack, parent agents, and application code.
- Represents provider responses, tool calls, tool results, and user messages as typed `Message` objects.
- Lets agents call `AgentSelf`, `AgentSpawn`, and `AgentSend` as ordinary tools.

## Install

```bash
pip install sagent
```

Sagent requires Python 3.12.

## Quickstart: CLI

```bash
export GOOGLE_API_KEY=...
sagent --provider Google --model gemini-3.1-pro-preview
```

For non-interactive use, pipe a prompt on stdin:

```bash
printf 'Say hi in one sentence.' | \
  sagent --provider Google --model gemini-3.1-pro-preview \
  --output-format json
```

Use `--continue` to resume the most recent session for this working directory, `--session PATH` for an explicit session directory, or `--no-session-persistence` when prompts and auto-memory should not be written to disk. Use `--max-budget-usd N` to cap API spend for the current run.

## Quickstart: Python

```python
import asyncio

from sagent import tools
from sagent.agent import Agent
from sagent.lib.json import json_freeze
from sagent.providers import Anthropic


async def main() -> None:
    agent = Agent(
        model=Anthropic.from_env().model("claude-sonnet-4-6"),
        system="You are a concise coding assistant.",
        tools=[tools.Read(), tools.Grep(), tools.Glob()],
    )
    result = await agent.run(json_freeze({"prompt": "Summarize README.md"}))
    print(result.content)


asyncio.run(main())
```

`Agent.run()` accepts a JSON directive with a `prompt` key and returns a `Message`.

## Provider setup

Sagent ships API-key providers for Anthropic, OpenAI, Google, Moonshot, DashScope, MiniMax, and generic OpenAI-compatible endpoints. Set the key for the provider you plan to use:

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export GOOGLE_API_KEY=...
export MOONSHOT_API_KEY=...
export DASHSCOPE_API_KEY=...
export MINIMAX_API_KEY=...
```

| Provider | Environment variable | Example model |
| --- | --- | --- |
| `Anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-4-6` |
| `OpenAI` | `OPENAI_API_KEY` | `gpt-5.5` |
| `Google` | `GOOGLE_API_KEY` | `gemini-3.1-pro-preview` |
| `Moonshot` | `MOONSHOT_API_KEY` | `kimi-k2.6` |
| `DashScope` | `DASHSCOPE_API_KEY` | `qwen3.6-plus` |
| `MiniMax` | `MINIMAX_API_KEY` | `MiniMax-M2.7` |

See [Providers](docs/providers.md) for more detail.

## Examples

The [`examples/`](examples/) directory contains small, runnable examples:

- `offline_custom_tool.py`: run an agent/tool/model loop without API keys.
- `decorator_tool.py`: wrap a function as a tool.
- `custom_tool.py`: implement the full `Tool` protocol.
- `multi_agent_reviewer.py`: spawn an isolated reviewer child.
- `openai_compatible_provider.py`: connect an OpenAI-compatible endpoint.

Start with the [tutorial](docs/tutorial.md), then use the examples as copyable patterns.

## Concepts

Sagent has five core contracts: `Message`, `Tool`, `Model`, `Provider`, and `Agent`.

- `Message` is the typed payload that crosses providers, tools, sessions, compaction, and UI surfaces.
- `Tool` receives a JSON directive in a message and returns a message.
- `Model` is the backend request/response interface.
- `Provider` owns authentication and constructs models.
- `Agent` owns the loop, model, tools, inbox, session, and compactor.

`TextMessage` is intentionally central: it is the common communication interface across the runtime.

See [Concepts](docs/concepts.md) and [Architecture](docs/architecture.md).

### Inbox zero

Most agent frameworks are turn-based: user sends a message, agent
processes it, agent responds, repeat. Sagent instead uses a
drain-run-check loop:

```text
while True:
    drain inbox into user messages
    call model
    if tool calls exist: dispatch tools and loop
    if inbox is empty and model is done: go idle
```

The agent goes idle only when the inbox is empty and the model has
nothing left to do. It wakes when anything lands in the inbox.

Every surface - REPL, Slack, CLI, parent agent, or application code -
puts messages in the same inbox. User input, background task results,
and agent-to-agent messages use the same mechanism instead of separate
plumbing. User messages go to the front; background and peer messages
append at the back.

Context-affecting slash commands follow the same rule. `/clear` is
queued and interpreted at the agent's single inbox drain point. Surface
local commands that do not mutate model context, such as `/model`, may
be handled by the REPL before entering the inbox.

### Message: typed payloads plus graph edges

Sagent messages use MIME-style descriptors for heterogeneous payloads,
plus ids and parent ids for graph structure. The public `Message` union
contains `TextMessage`, `BytesMessage`, `JsonMessage`, and
`MultipartMessage`.

`MultipartMessage` content is recursive: compound messages hold nested
messages. An assistant turn containing text, thinking, and tool calls
uses the same message graph as a single text chunk. Descriptors such as
`text/plain`, `multipart/x-tool-call`, and `application/x-done` tell
callers how to interpret the payload.

### Tool: one input message, one output message

Tools are normal Python objects with a small protocol:

```python
class Tool(Protocol):
    name: str
    tool_id: str
    description: str
    directive_schema: JSON
    supports_microcompaction: bool

    def summary(self, msg: Message) -> str: ...
    def prompt(self) -> str | None: ...
    async def run(self, msg: Message) -> Message: ...
```

Input is a `Message` with a JSON directive. Output is a `Message`.
Expected tool failures return `descriptor="text/x-error"` rather than
raising through the agent loop.

`Agent` follows the same interface pattern as a tool. `AgentSpawn` is
a tool that builds a child `Agent`, runs it, and returns the child's
final output as a tool response. That is what makes recursive agent
composition work without a separate orchestration layer.

### AgentSelf, AgentSpawn, AgentSend

- `AgentSelf` lets an agent inspect or mutate its own state: update
  status, compact context, clear context, change model, inspect
  diagnostics, and adjust token limits.
- `AgentSpawn` creates child agents with explicit tool/depth limits for
  isolated reviews, subtasks, and map-reduce work.
- `AgentSend` delivers a message to another live named agent's inbox.
  This makes multi-agent coordination peer-to-peer rather than only
  parent-to-child.

## Security and privacy

Sagent is an agent runtime, not a sandbox. Enabled tools run with the current
process permissions: `Bash` executes local commands, file tools read and write
accessible paths, and provider/network tools send data to their configured
services. Sessions are plaintext local state and may contain prompts, model
responses, tool results, file snippets, and paths.

Use narrow tool sets, pass `--no-session-persistence` for one-off sensitive
prompts so sessions and auto-memory are disabled, and run Sagent inside your own
OS/container sandbox when a task needs hard isolation. See
[Security](docs/security.md).

## Current scope

Sagent does not currently include:

- MCP integration;
- LSP integration;
- native sandboxing;
- a desktop UI;
- a tree-sitter repo map;
- a hosted service;
- browser automation.

## Adjacent projects

This comparison focuses on the runtime shape rather than every feature
of each project.

| | [Sagent](https://github.com/rekursiv-ai/sagent) | [aider](https://github.com/Aider-AI/aider) | [LangChain](https://github.com/langchain-ai/langchain) | [OpenClaw](https://github.com/openclaw/openclaw) | [Cline](https://github.com/cline/cline) | [Claude Code](https://docs.anthropic.com/en/docs/claude-code) | [Codex CLI](https://github.com/openai/codex) | [Gemini CLI](https://github.com/google-gemini/gemini-cli) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Python library | ✅ | 🟡 | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Multi-provider | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| Context compaction | ✅ | 🟡 | 🟡 | ❌ | 🟡 | ✅ | ❌ | ✅ |
| User-initiated backend swap | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| Agent-initiated backend swap | ✅ | ❌ | 🟡 | ❌ | ❌ | ❌ | ❌ | ❌ |
| Agent self-mutation | ✅ | ❌ | ❌ | ❌ | ❌ | 🟡 | ❌ | ❌ |
| Context hot-swap | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | ❌ | ❌ | ❌ |
| Recursive agent spawn | ✅ | ❌ | ✅ | 🟡 | ❌ | ✅ | ❌ | ❌ |
| Multi-agent | ✅ | ❌ | ✅ | ✅ | ❌ | 🟡 | ❌ | ❌ |
| GitHub stars | -- | 44.4k | 135.8k | 368.6k | 61.4k | -- | 80.1k | 103.2k |

✅ = yes, 🟡 = partial, ❌ = no. Corrections welcome --
[open a PR](https://github.com/rekursiv-ai/sagent/pulls).

### How each project works

**[aider](https://github.com/Aider-AI/aider)** --
Git-native pair programmer. The LLM emits markdown-formatted edits (14
edit formats) and aider parses them -- there is no structured tool calling.
All providers route through litellm as a single string-addressed transport.
`/model` switches the backend mid-session by raising `SwitchCoder`, which
reconstructs the entire `Coder` object; conversation history carries over
but the swap is destructive. A tree-sitter repo map with PageRank ranking
provides structural code awareness that Sagent lacks. No multi-agent
capabilities beyond a synchronous Architect-to-Editor handoff. Importable
via `Coder.create()` but the scripting API is explicitly unsupported and
may change without notice.

**[LangChain/LangGraph](https://github.com/langchain-ai/langchain)** --
Broad Python application framework for LLM pipelines. Multi-provider,
multi-agent (via LangGraph state machines), and fully programmatic. Context
compaction, backend swapping, and agent self-mutation are all possible but
application-defined rather than built-in -- the framework provides building
blocks, not an opinionated agent loop. Sagent is a smaller, more
opinionated runtime with typed protocols, a concrete inbox loop, and
built-in session persistence.

**[OpenClaw](https://github.com/openclaw/openclaw)** --
Multi-platform personal assistant (desktop, mobile, web) with multi-provider
and multi-agent support. Agents coordinate across channels but the system
is oriented toward end-user assistant workflows rather than developer
tooling. TypeScript-based, not available as a Python library.

**[Cline](https://github.com/cline/cline)** --
VS Code extension with multi-provider support. Users can switch models in
the settings panel mid-conversation, but the extension is not importable as
a library. Single-agent with no spawn or coordination primitives. Context
management is truncation-based rather than structured compaction.

**[Claude Code](https://docs.anthropic.com/en/docs/claude-code)** (Anthropic) --
Closed-source vendor CLI with strong tool-use capabilities and structured
context compaction. Agents can spawn recursive sub-agents and compact their
own context, but cannot switch providers (Anthropic-only) or dynamically
adjust token limits. Not available as a Python library; the SDK is
JavaScript. No user-initiated backend swap since there is only one backend.

**[Codex CLI](https://github.com/openai/codex)** (OpenAI) --
Rust-based CLI locked to OpenAI models. Single-agent, single-provider, no
compaction, no programmatic API. Clean local-execution model with
sandboxing, but no extensibility surface for custom tools, provider
swapping, or multi-agent coordination.

**[Gemini CLI](https://github.com/google-gemini/gemini-cli)** (Google) --
TypeScript CLI locked to Google models. Has context compaction via
summarization. Single-agent, single-provider, no programmatic API, no
custom tool protocol. Designed as a terminal interface for Gemini, not as a
composable runtime.

## Architecture map

| Module | Role |
| --- | --- |
| `bin/cli.py` | Terminal entry point |
| `bin/slack.py` | Slack Socket Mode entry point |
| `agent/` | Turn loop, retry, dispatch, sessions |
| `compactor.py` | Structured compaction and prompt-too-long retry |
| `custom_types.py` | Message, Tool, Model, Provider protocols |
| `providers/` | Anthropic, OpenAI, Google, Moonshot, DashScope, MiniMax, OpenAI-compatible |
| `tools/` | Built-in tools for files, shell, web, search, and agent coordination |
| `repl/` | prompt_toolkit REPL and diff rendering |
| `sessions.py` | Per-cwd session storage |
| `prompt.py` | System prompt assembly |

## Name

**sagent** (noun, neologism) /ˈseɪ.dʒənt/

From *sage* + *agent*.

An AI assistant that confidently performs a task you didn't ask for while ignoring the one you did.

> *"I asked the sagent to fix one failing test -- it deleted the test and reported all green."*

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for local validation and public contribution flow.

## License

Apache License 2.0
