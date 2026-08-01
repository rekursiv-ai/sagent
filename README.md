# sagent🪄

[![PyPI version](https://img.shields.io/pypi/v/sagent.svg)](https://pypi.org/project/sagent/)
[![CI](https://github.com/rekursiv-ai/sagent/actions/workflows/package-validation.yml/badge.svg?branch=main)](https://github.com/rekursiv-ai/sagent/actions/workflows/package-validation.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Discord](https://img.shields.io/discord/1530237005311639592?logo=discord&logoColor=white&label=Discord&color=5865F2)](https://discord.gg/2GZFPPvCqn)

<p align="center">
  <img alt="sagent logo" src="https://raw.githubusercontent.com/rekursiv-ai/sagent/main/assets/logo-custom.webp" width="180">
</p>

<p align="center">
  A coding-agent CLI and strongly-typed Python library -- self-mutating, hot-swapping, multi-provider, with async tool calls and bidirectional recursive spawn.
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
  <a href="https://github.com/rekursiv-ai/sagent/blob/main/docs/selfhosted.md">Self-hosted</a>
  ·
  <a href="https://github.com/rekursiv-ai/sagent/blob/main/docs/showcase.md">Showcase</a>
  ·
  <a href="https://github.com/rekursiv-ai/sagent/tree/main/examples">Examples</a>
</p>

## Quick Start

```bash
# Mac:
#   # Required for quick install.
#   brew install uv
#   # Optional for improved performance.
#   brew install ripgrep fd

# Ubuntu/Debian:
#   # Required for quick install.
#   sudo apt-get install -y curl
#   curl -LsSf https://astral.sh/uv/install.sh | sh
#   # Optional for improved performance.
#   sudo apt-get install -y ripgrep fd-find

uv tool install sagent

sagent
```

## Better CLI

Things Claude Code, Codex CLI, and Gemini CLI don't do:

- **Async REPL.** Chat with agents *about* jobs while those jobs run. No ctrl+b, no manual juggling.
- **Hot self-mutation.** Switch provider, model, or thinking effort mid-session in plain English. No restart.
- **One CLI, every provider.** Anthropic, OpenAI, Google, Moonshot, DashScope, MiniMax, OpenAI-compatible endpoints, self-hosted HuggingFace models, and a managed `llama.cpp` server, all behind one binary.
- **Unified cost tracking.** One USD total across every provider in a session; sub-agent costs roll up to the root. `--max-budget-usd N` caps the whole tree.
- **Self-directing agent fleets.** Agents retune their own runtime -- provider, model, thinking, context -- mid-task. A coordinator can do it to its workers over `AgentSend`: *"switch to o1, crank thinking, recompact and drop the file reads."*
- **Recursive agent messaging.** Any spawned agent can spawn and `AgentSend` to any peer, so coordination is a tree, not a star. Claude Code's experimental Agent Teams is flat (one lead, no nesting); Codex and Gemini have no peer messaging.
- **Interruptible, detachable tasks.** Tell a stuck task to stop, or detach one and let it keep running.
- **Richer built-in tools.** `PaperSearch`/`PaperFetch` walk citation graphs and fetch PDFs, multi-backend `WebSearch`, `WebFetch` with markdown extraction, atomic read/write tracking on file tools.
- **Unix-aligned and pipeable.** `stdin`, `stdout`, exit codes, and `--output-format json` are first-class. Pipe through `jq`, drop into `ipython` (same `prompt_toolkit` underneath).

## Uniquely also an API

- **One runtime, every surface.** The same `Agent` class powers the CLI, your application code, and recursive sub-agents.
- **Typed Python objects.** `Agent`, `Tool`, `Model`, `Provider`, and `Message` are protocols and dataclasses you import, compose, and unit-test.
- **Peer-to-peer agent messaging.** Any spawned agent can `AgentSend` to any other named peer -- not just its parent. Like user input, peer messages preempt the receiving agent's tool calls, so no agent blocks waiting on a stuck child.

Use it as a library:

```python
from sagent import tools
from sagent.agent import Agent
from sagent.lib.custom_json import json_freeze
from sagent.providers import Google

agent = Agent(
    model=Google.from_env().model("gemini-3.1-pro-preview"),
    system="You are a scientist.",
    tools=[tools.Read(), tools.Glob(), tools.Grep()],
)
result = await agent.run(json_freeze({"prompt": "analyze the CSV in ./data/"}))
print(result.content)
```

## Install

Sagent requires Python 3.12 or newer. `ripgrep` and `fd-find` are
optional -- sagent has Python fallbacks when absent -- but recommended
for faster `Grep` / `Glob`. PDF rendering uses the bundled `pypdfium2`
wheel and needs no system install. The [Quick Start](#quick-start)
above installs the `sagent` CLI.

Add sagent to your own project as a library:

```bash
uv add sagent
```

Or run from a source checkout:

```bash
git clone --depth 1 https://github.com/rekursiv-ai/sagent.git
cd sagent
uv run sagent --help
```

## Run

Bare `sagent` uses Anthropic and reads `ANTHROPIC_API_KEY`:

```bash
export ANTHROPIC_API_KEY=...
sagent
```

Pick a different provider by setting its key (see
[Provider setup](#provider-setup)) and passing `--provider`:

```bash
export OPENAI_API_KEY=...
sagent --provider OpenAI
```

`--provider` defaults to the first name in `--allow-providers`, so
`SAGENT_ALLOW_PROVIDERS` alone picks the default backend and also caps
which providers spawned sub-agents may use:

```bash
SAGENT_ALLOW_PROVIDERS=OpenAI sagent   # OpenAI is now the default provider
```

Pipe a prompt on stdin for non-interactive use:

```bash
printf 'Say hi in one sentence.' | \
  sagent --provider OpenAI --output-format json
```

Use `--continue` to resume the most recent session for this working directory, `--session PATH` for an explicit session directory, or `--ephemeral` when prompts and auto-memory should not be written to disk. Use `--max-budget-usd N` to cap API spend for the current run.

See [CLI](https://github.com/rekursiv-ai/sagent/blob/main/docs/cli.md) and [Sessions](https://github.com/rekursiv-ai/sagent/blob/main/docs/sessions.md) for the full flag set.

## Quickstart: Python

```python
import asyncio

from sagent import tools
from sagent.agent import Agent
from sagent.lib.custom_json import json_freeze
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

See [API](https://github.com/rekursiv-ai/sagent/blob/main/docs/api.md), [Tutorial](https://github.com/rekursiv-ai/sagent/blob/main/docs/tutorial.md), and [Concepts](https://github.com/rekursiv-ai/sagent/blob/main/docs/concepts.md) for more detail.

## Provider setup

Sagent ships API-key providers for Anthropic, OpenAI, OpenAISubscription, Google, Moonshot, DashScope, MiniMax, and generic OpenAI-compatible endpoints, a subscription-backed `AnthropicCLI` that rides your installed `claude` login, plus a managed local `LlamaCpp` provider. Set the key (or run the login) for the provider you plan to use:

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export GOOGLE_API_KEY=...
export MOONSHOT_API_KEY=...
export DASHSCOPE_API_KEY=...
export MINIMAX_API_KEY=...
```

and

```bash
export SAGENT_ALLOW_PROVIDERS=...
```

to set the default value of the `--provider` flag.

| Provider | Environment variable | Example model |
| --- | --- | --- |
| `Anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-4-6` |
| `AnthropicCLI` | none (`claude auth login --claudeai`) | `claude-sonnet-4-6` |
| `OpenAI` | `OPENAI_API_KEY` | `gpt-5.6-sol` |
| `Google` | `GOOGLE_API_KEY` | `gemini-3.1-pro-preview` |
| `Moonshot` | `MOONSHOT_API_KEY` | `kimi-k2.6` |
| `DashScope` | `DASHSCOPE_API_KEY` | `qwen3.6-plus` |
| `MiniMax` | `MINIMAX_API_KEY` | `MiniMax-M2.7` |
| `SelfHosted` | none | `Qwen/Qwen3.6-27B` |
| `LlamaCpp` | none (uses `LLAMA_CPP_MODEL` + `LLAMA_CPP_SERVER`) | `qwen3.6-27b-12gb` |

See [Providers](https://github.com/rekursiv-ai/sagent/blob/main/docs/providers.md) for the provider matrix, inference rules, and OpenAI-compatible provider setup.

## Self-hosted models

Install the local runtime extra from a checkout:

```bash
uv sync --extra selfhosted
```

Or add it to your project from PyPI:

```bash
uv add "sagent[selfhosted]"
```

Then pass a HuggingFace repo ID or local snapshot path:

```bash
sagent --provider SelfHosted --model Qwen/Qwen3.6-27B+bfloat16+cuda
sagent --provider SelfHosted --model Qwen/Qwen3.6-27B+cuda+bfloat16
```

For a small smoke test:

```bash
sagent --provider SelfHosted --model Qwen/Qwen3-0.6B+float16+cuda \
  --effort none --max-response-tokens 32 --max-tool-call-rounds 1
```

SelfHosted options use `+` suffixes after the model name. Device, dtype, and `compile` can appear in any order, but each category can appear once.

The `LlamaCpp` provider is a second local option: it manages a
`llama-server` subprocess and talks to it over its OpenAI-compatible
endpoint. Point `LLAMA_CPP_SERVER` at a built `llama-server` binary and
`LLAMA_CPP_MODEL` at a `.gguf` file, then run
`sagent --provider LlamaCpp --model qwen3.6-27b-12gb`.

See [Self-hosted Models](https://github.com/rekursiv-ai/sagent/blob/main/docs/selfhosted.md) for options, local snapshot paths, and runtime requirements.

## Examples

The [`examples/`](https://github.com/rekursiv-ai/sagent/tree/main/examples/) directory contains small, runnable examples:

- `offline_custom_tool.py`: run an agent/tool/model loop without API keys.
- `decorator_tool.py`: wrap a function as a tool.
- `custom_tool.py`: implement the full `Tool` protocol.
- `multi_agent_reviewer.py`: spawn an isolated reviewer child.
- `openai_compatible_provider.py`: connect an OpenAI-compatible endpoint.

Start with the [tutorial](https://github.com/rekursiv-ai/sagent/blob/main/docs/tutorial.md), then use the examples as copyable patterns. See [Examples](https://github.com/rekursiv-ai/sagent/tree/main/examples/) and [Tools](https://github.com/rekursiv-ai/sagent/blob/main/docs/tools.md).

## Security and privacy

Sagent is an agent runtime, not a sandbox. Enabled tools run with the current
process permissions: `Bash` executes local commands, file tools read and write
accessible paths, and provider/network tools send data to their configured
services. Sessions are plaintext local state and may contain prompts, model
responses, tool results, file snippets, and paths.

Use narrow tool sets, pass `--ephemeral` for one-off sensitive
prompts so sessions and auto-memory are disabled, and run Sagent inside your own
OS/container sandbox when a task needs hard isolation. See
[Security](https://github.com/rekursiv-ai/sagent/blob/main/docs/security.md).

## Comparison

<details>
<summary>How Sagent compares to aider, LangChain, Claude Code, Codex CLI, Gemini CLI, and other adjacent projects</summary>

Not yet in Sagent: MCP, LSP, native sandboxing, desktop UI, tree-sitter repo map, hosted service, browser automation.

This comparison focuses on the runtime shape rather than every feature
of each project.

| | [Sagent](https://github.com/rekursiv-ai/sagent) | [aider](https://github.com/Aider-AI/aider) | [LangChain](https://github.com/langchain-ai/langchain) | [OpenClaw](https://github.com/openclaw/openclaw) | [Cline](https://github.com/cline/cline) | [Claude Code](https://docs.anthropic.com/en/docs/claude-code) | [Codex CLI](https://github.com/openai/codex) | [Gemini CLI](https://github.com/google-gemini/gemini-cli) | [Flue](https://github.com/withastro/flue) | [Pi](https://pi.dev/) | [Attractor](https://github.com/strongdm/attractor) | [npcsh](https://github.com/npc-worldwide/npcsh) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Python library                 | ✅ | 🟡    | ✅     | ❌      | ❌    | ❌          | ❌        | ❌         | ❌    | ❌    | ❌    | ✅  |
| Multi-provider                 | ✅ | ✅    | ✅     | ✅      | ✅    | ❌          | ❌        | ❌         | ✅    | ✅    | ✅    | ✅  |
| Context compaction             | ✅ | 🟡    | 🟡     | ❌      | 🟡    | ✅          | ✅        | ✅         | ❌    | ✅    | ✅    | ❌  |
| User-initiated backend swap    | ✅ | ✅    | ❌     | ✅      | ✅    | ❌          | ❌        | ❌         | ❌    | ✅    | ❌    | ✅  |
| Agent-initiated backend swap   | ✅ | ❌    | 🟡     | ❌      | ❌    | ❌          | ❌        | ❌         | ❌    | ❌    | 🟡    | ❌  |
| Agent self-mutation            | ✅ | ❌    | ❌     | ❌      | ❌    | ❌          | ❌        | ❌         | ❌    | 🟡    | ❌    | 🟡  |
| Context hot-swap               | ✅ | 🟡    | 🟡     | 🟡      | 🟡    | ❌          | ❌        | ❌         | 🟡    | ✅    | ✅    | ❌  |
| Recursive agent spawn          | ✅ | ❌    | ✅     | 🟡      | ❌    | 🟡          | 🟡        | ❌         | ✅    | 🟡    | ✅    | ✅  |
| Multi-agent (fully detached)   | ✅ | ❌    | ✅     | ✅      | ❌    | 🟡          | 🟡        | ❌         | ✅    | 🟡    | ✅    | 🟡  |
| GitHub stars (May 2026)        | -- | 44.4k | 135.8k | 368.6k | 61.4k | --         | 80.1k     | 103.2k     | 2.5k  | 48.6k | 1.1k  | 388 |

✅ = yes, 🟡 = partial, ❌ = no. Corrections welcome --
[open a PR](https://github.com/rekursiv-ai/sagent/pulls).

### How each project works

- **[aider](https://github.com/Aider-AI/aider)** -- git-native pair programmer; markdown-diff edits (no structured tool calls), litellm transport, destructive mid-session `/model` swap, tree-sitter repo map, no multi-agent.
- **[LangChain/LangGraph](https://github.com/langchain-ai/langchain)** -- broad LLM-app framework; everything is possible but application-defined, not an opinionated agent loop.
- **[OpenClaw](https://github.com/openclaw/openclaw)** -- TypeScript multi-platform personal assistant; multi-agent but end-user-oriented, no Python library.
- **[Cline](https://github.com/cline/cline)** -- VS Code extension; multi-provider, single-agent, truncation-based context, not importable.
- **[Claude Code](https://docs.anthropic.com/en/docs/claude-code)** (Anthropic) -- Anthropic-only vendor CLI; recursive sub-agents and compaction, but no provider swap and no Python library (JS SDK).
- **[Codex CLI](https://github.com/openai/codex)** (OpenAI) -- OpenAI-only Rust CLI; sandboxed local execution, single-agent, no compaction, no API.
- **[Gemini CLI](https://github.com/google-gemini/gemini-cli)** (Google) -- Google-only TypeScript CLI; summarization compaction, single-agent, no API, no custom tools.
- **[Flue](https://github.com/withastro/flue)** (Astro) -- headless TypeScript harness; pluggable sandboxes, recursive `session.task()` delegation, model chosen per call (no agent-initiated swap), no UI/compaction.
- **[Pi](https://pi.dev/)** ([earendil-works/pi](https://github.com/earendil-works/pi)) -- minimal TypeScript harness; branchable session tree, `/reload` soft self-mutation, sub-agents opt-in only.
- **[npcsh](https://github.com/npc-worldwide/npcsh)** -- Python agentic shell; filesystem-defined NPC personas, many built-in modes, hub-and-spoke sub-agents, rate-limit-fallback "compaction".
- **[Attractor](https://github.com/strongdm/attractor)** (StrongDM) -- a spec, not an implementation; DOT-graph pipeline where nodes are AI tasks and the graph is the workflow.

</details>

## Name

**sagent** (noun, neologism) /ˈseɪ.dʒənt/

From *sage* + *agent*.

An AI assistant that confidently performs a task you didn't ask for while ignoring the one you did.

> *"I asked the sagent to fix one failing test -- it deleted the test and reported all green."*

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for local validation and public contribution flow.

## License

Apache License 2.0
