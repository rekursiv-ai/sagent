# sagent🪄

<p align="center">
  <img alt="sagent logo" src="https://raw.githubusercontent.com/rekursiv-ai/sagent/main/assets/logo-custom.webp" width="180">
</p>

<p align="center">
  The self-mutating multi-provider coding-agent CLI and typed Python library.
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
  <a href="https://github.com/rekursiv-ai/sagent/blob/main/docs/selfhosted.md">Self-hosted</a>
  ·
  <a href="https://github.com/rekursiv-ai/sagent/tree/main/examples">Examples</a>
</p>

## Better CLI

Things Claude Code, Codex CLI, and Gemini CLI don't do:

- **One CLI, every provider.** Anthropic, OpenAI, Google, Moonshot,
  DashScope, MiniMax, OpenAI-compatible endpoints, self-hosted
  HuggingFace models, and a managed `llama.cpp` server — all behind
  one binary.
- **Unified cost tracking.** One USD total across every provider in
  a session; sub-agent costs roll up to the root automatically.
  `--max-budget-usd N` caps the whole tree.
- **Hot self-mutation.** *"Switch to OpenAI then back."* Mid-session,
  no restart.
- **Self-directing agent fleets.** Agents mutate their own runtime —
  provider, model, thinking effort, context — in plain English,
  mid-task. Paired with peer-to-peer `AgentSend`, a coordinator can
  retune its workers on the fly: *"switch to o1, crank thinking,
  recompact and drop the file reads."* It can also tell you how many
  tokens it's holding.
- **Recursive agent-to-agent messaging.** Any spawned agent can spawn
  and `AgentSend` to peers, so coordination is an arbitrary tree of
  agents, not a flat star. Claude Code's experimental Agent Teams
  (`CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`) is the closest comparable
  feature but is flat: one fixed lead, peer teammates, no nested
  teams. Codex and Gemini CLIs have no peer messaging at all.
- **Interruptible and detachable tasks.** *"The task is stuck."*
  *"Detach `foo` and let it keep running."*
- **Richer built-in tools.** `PaperSearch` and `PaperFetch` walk
  citation graphs and fetch PDFs, multi-backend `WebSearch`,
  `WebFetch` with markdown extraction, atomic read/write tracking
  on file tools.
- **Unix-aligned and pipeable.** `stdin`, `stdout`, exit codes, and
  `--output-format json` are first-class — not a non-interactive
  escape hatch. Pipes like `jq`, REPLs like `ipython` (same
  `prompt_toolkit` underneath).

## Uniquely also an API

- **One runtime, every surface.** The same `Agent` class powers the
  CLI, your application code, and recursive sub-agents.
- **Typed Python objects.** `Agent`, `Tool`, `Model`, `Provider`,
  and `Message` are protocols and dataclasses you import, compose,
  and unit-test.
- **Peer-to-peer agent messaging.** Any spawned agent can `AgentSend`
  to any other named peer — not just its parent. Like user input,
  peer messages preempt the receiving agent's tool calls, so no agent
  blocks waiting on a stuck child.

Use it as a library:

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

## Install

Sagent requires Python 3.12 or newer. `ripgrep` and `fd-find` are
optional — sagent has Python fallbacks when absent — but recommended
for faster `Grep` / `Glob`. PDF rendering uses the bundled `pypdfium2`
wheel and needs no system install.

### CLI

Installs the `sagent` binary into an isolated environment so it lands
on your PATH without touching the system Python.

#### Ubuntu / Debian

```bash
sudo apt-get install -y ripgrep fd-find pipx
pipx install sagent
```

#### macOS

```bash
brew install ripgrep fd pipx
pipx install sagent
```

### Library

For embedding sagent in your own Python project.

#### Ubuntu / Debian

```bash
sudo apt-get install -y ripgrep fd-find python3-venv
python3 -m venv .venv && source .venv/bin/activate
pip install sagent
```

#### macOS

```bash
brew install ripgrep fd
python3 -m venv .venv && source .venv/bin/activate
pip install sagent
```

### From source

#### Ubuntu / Debian

```bash
sudo apt-get install -y ripgrep fd-find git
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone --depth 1 https://github.com/rekursiv-ai/sagent.git
cd sagent
sagent/bin/cli.py --help
```

#### macOS

```bash
brew install ripgrep fd uv git
git clone --depth 1 https://github.com/rekursiv-ai/sagent.git
cd sagent
sagent/bin/cli.py --help
```

## Quickstart: CLI

Use Claude backend,

```
sagent/bin/cli.py --allow-providers AnthropicCLI --provider AnthropicCLI --auth credentials
```

```bash
export GOOGLE_API_KEY=...
sagent/bin/cli.py --provider Google --model gemini-3.1-pro-preview
```

For non-interactive use, pipe a prompt on stdin:

```bash
printf 'Say hi in one sentence.' | \
  sagent/bin/cli.py --provider Google --model gemini-3.1-pro-preview \
  --output-format json
```

Use `--continue` to resume the most recent session for this working directory, `--session PATH` for an explicit session directory, or `--ephemeral` when prompts and auto-memory should not be written to disk. Use `--max-budget-usd N` to cap API spend for the current run.

See [CLI](docs/cli.md) and [Sessions](docs/sessions.md) for the full flag set.

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

See [API](docs/api.md), [Tutorial](docs/tutorial.md), and [Concepts](docs/concepts.md) for more detail.

## Provider setup

Sagent ships API-key providers for Anthropic, OpenAI, OpenAISubscription, Google, Moonshot, DashScope, MiniMax, and generic OpenAI-compatible endpoints, plus a managed local `LlamaCpp` provider. Set the key for the provider you plan to use:

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
| `SelfHosted` | none | `Qwen/Qwen3.6-27B` |
| `LlamaCpp` | none (uses `LLAMA_CPP_MODEL` + `LLAMA_CPP_SERVER`) | `qwen3.6-27b-12gb` |

See [Providers](docs/providers.md) for the provider matrix, inference rules, and OpenAI-compatible provider setup.

## Self-hosted models

Install the local runtime extra from a checkout:

```bash
uv sync --extra selfhosted
```

Or install it from PyPI:

```bash
pip install "sagent[selfhosted]"
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

See [Self-hosted Models](docs/selfhosted.md) for options, local snapshot paths, and runtime requirements.

## Examples

The [`examples/`](examples/) directory contains small, runnable examples:

- `offline_custom_tool.py`: run an agent/tool/model loop without API keys.
- `decorator_tool.py`: wrap a function as a tool.
- `custom_tool.py`: implement the full `Tool` protocol.
- `multi_agent_reviewer.py`: spawn an isolated reviewer child.
- `openai_compatible_provider.py`: connect an OpenAI-compatible endpoint.

Start with the [tutorial](docs/tutorial.md), then use the examples as copyable patterns. See [Examples](examples/) and [Tools](docs/tools.md).

## Security and privacy

Sagent is an agent runtime, not a sandbox. Enabled tools run with the current
process permissions: `Bash` executes local commands, file tools read and write
accessible paths, and provider/network tools send data to their configured
services. Sessions are plaintext local state and may contain prompts, model
responses, tool results, file snippets, and paths.

Use narrow tool sets, pass `--ephemeral` for one-off sensitive
prompts so sessions and auto-memory are disabled, and run Sagent inside your own
OS/container sandbox when a task needs hard isolation. See
[Security](docs/security.md).

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

**[Flue](https://github.com/withastro/flue)** (Astro) --
TypeScript "agent harness framework," explicitly headless and runtime-agnostic
(Node.js, Cloudflare Workers, GitHub Actions). Agents are TypeScript modules
with triggers (HTTP webhook, CLI). Sandbox is pluggable: a fast in-process
`just-bash` virtual sandbox by default, or full Linux containers via
Daytona/E2B connectors. `session.task()` spawns child agents in the same
sandbox; the same primitive is exposed to the LLM, so agents can recursively
delegate. Multi-provider via model strings (`anthropic/claude-sonnet-4-6`,
`openrouter/...`). No interactive UI, no built-in compaction, no
agent-initiated backend swap -- the developer chooses the model at `init()`
or per call. Skills, AGENTS.md, and per-call MCP tool injection are
first-class.

**[Pi](https://pi.dev/)** ([earendil-works/pi](https://github.com/earendil-works/pi),
formerly `badlogic/pi-mono`) --
TypeScript "minimal terminal coding harness." The design point is the
opposite of Sagent's: ship aggressively few defaults and make every layer
extensible (skills, prompt templates, themes, extensions, packages
distributed via npm or git). `/model` and `Ctrl+L` swap the backend
mid-session; `/tree`, `/fork`, and `/clone` make session history a
branchable tree (genuine context hot-swap); `/compact` runs a structured
summarization prompt that records read/modified files. `/reload` lets the
agent rewrite its own skills, prompts, themes, and extensions and pick up
the change in place -- a soft form of self-mutation. Sub-agents ship only
as an example extension; plan mode, permission gates, sandboxing, and MCP
support are all similarly opt-in. Print/JSON, RPC, and SDK modes make it
embeddable. Star count reflects the whole monorepo
(`pi-coding-agent` + `pi-agent-core` + `pi-ai` + TUI/web-UI libraries),
not the coding agent in isolation.

**[npcsh](https://github.com/npc-worldwide/npcsh)** --
Python "AI-powered, agentic shell" built on the sibling `npcpy` library and
LiteLLM. The design point is the opposite of Sagent's minimalism: agents
are filesystem-defined NPC personas (`.npc` files) grouped in YAML "teams,"
tools are `.jinx` skill files, and dozens of full-screen modes ship in the
box -- `/deep_research`, `/wander`, `/guac` (LLM Python REPL), `/yap`
(voice), `/kg` (knowledge graph), `/convene` (multi-NPC discussion),
`/delegate` (sub-agent with review loop), `/serve` (OpenAI-compatible team
API). `/set model` and `/set provider` swap the backend mid-session;
`/reattach` resumes prior sessions but there is no branchable session
tree. Context "compaction" is a rate-limit fallback that drops middle
messages (`_state.py:4183-4186`), not LLM-summarization. Sub-agents exist
but are orchestrator-hub-and-spoke, not detached peers.

**[Attractor](https://github.com/strongdm/attractor)** (StrongDM) --
*Specification, not an implementation.* A pair of NLSpecs (`attractor-spec.md`,
`coding-agent-loop-spec.md`, `unified-llm-spec.md`) you hand to a coding agent
and ask it to build. Attractor proper is a DOT-graph pipeline runner: nodes
are AI tasks, edges encode routing/conditions, the graph IS the workflow.
The spec mandates structured fidelity modes (`full`/`truncate`/`compact`/
`summary:{low,medium,high}`) for cross-stage context handoff, per-node model
selection via a CSS-like stylesheet, parallel/fan-in handlers, human-gate
nodes, and checkpoint/resume. The companion coding-agent-loop spec defines
provider-aligned toolsets (apply_patch for OpenAI, edit_file for Anthropic,
gemini-cli tools for Gemini) and subagent spawn primitives. Categories above
reflect what an Attractor-conformant implementation must support; the actual
runtime shape depends on whoever builds it.

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
