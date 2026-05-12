# Sagent vs LangChain/LangGraph

LangChain is a ~700-package ecosystem for any LLM application: RAG,
agents, evals, extraction, classification, ETL. Sagent is one package
for coding agents. They overlap on "call an LLM with tools" and diverge
elsewhere.

## What each one is

LangChain: `langchain-core` (Runnable/LCEL), `langchain` (chains,
retrievers, output parsers), `langgraph` (state-machine agents with
durable checkpointing), `langsmith` (hosted traces and evals), plus
hundreds of integration packages.

Sagent: one Apache-licensed Python package shipping five typed
contracts (`Message`, `Tool`, `Model`, `Provider`, `Agent`), one locked
inbox-driven runtime, built-in tools (files, shell, search, web,
papers, Slack, Linear, audio, wiki, skills, background jobs, agent
coordination), and four surfaces on the same `Agent` (CLI, REPL, Slack
Socket Mode, library API). No document loaders, embeddings, vector
stores, retrievers, output parsers, or graph DSL.

## Execution model

### LangGraph: state-machine workflow runner

`StateGraph` of typed `State`, conditional edges, `ToolNode` for
parallel tool calls, `astream` / `astream_events` for streaming.
`interrupt(payload)` persists state via the `Checkpointer` (memory,
SQLite, Postgres) and returns control to the caller; resume by
re-invoking with `Command(resume=value)` and the node replays from the
top. Built for human-in-the-loop approvals that survive process
restart.

### Sagent: drain-driven inbox loop

```text
while True:
    drain inbox into RuntimeEvent batches
    dispatch each event in one match block
    if cohort empty and no model call in flight and no compaction pending:
        call model
```

Every input is a `RuntimeEvent` on a `GatedDeque`: user text, model
chunks, tool results, halt, kill, clear, peer messages, background
completions, compaction completions. Tool calls run as `asyncio.Task`s
in a cohort set; the model gate fires when the cohort drains.

Five in-flight verbs:

- `halt`: cancel the model, gate on user input, leave tools running.
- `kill <id|all>`: cancel tasks, drop from cohort.
- `detach <id|all>`: stub with `[detached]` (satisfies the provider's
  `tool_use`/`tool_result` invariant), let tools finish in the
  background, deliver as `DetachedResult` in the next round.
- `undetach <id|all>`: re-gate on a detached tool.
- `clear`: cancel model, detach all tools, wipe history, gate.

User input arriving mid-cohort uses detach automatically. `Await(types)`
blocks `drain()` until a matching event lands, so the runtime sleeps
without polling.

### Where each model is better

LangGraph wins on durable cross-process pause/resume, declarative
branching/looping topology, and human-in-the-loop gates that survive
restart. Sagent wins on mid-turn user injection, multiple input sources
on one loop, the five in-flight verbs as primitives, and peer-to-peer
agent messaging at runtime. Reinventing one inside the other yields
the other.

## Background work, scheduling, and inbox priority

| Capability | Sagent | LangChain / LangGraph |
| --- | --- | --- |
| Run multiple tool calls from one turn concurrently | `asyncio.Task` per call, cohort `set[str]` gates the next model call | `ToolNode` + `asyncio.gather` |
| Run a single tool call asynchronously, deliver result later | `background: true` / `delay: N` injected into every tool's directive schema by `BackgroundAwareTool` | Application-defined: write a tool that returns a job id and a separate poll tool |
| Model-callable job control (list / cancel / resume long-running work) | `BackgroundTask` tool with `list` / `cancel <id>` / `foreground <id>` operations | Application-defined |
| Registry of in-flight long-running operations | `BackgroundTaskEntry` covering tool calls, persistent subagents, detached cohort members, hidden infra | Application-defined |
| Pause execution mid-run and resume later in another process | None. Sessions persist transcripts, not mid-cohort state | `interrupt()` + `Checkpointer` (memory / SQLite / Postgres backends) |
| Resume semantics after pause | N/A | Replay-based: the node re-runs from the top up to the `interrupt()` call |
| Pluggable persistence backend for run state | Local JSONL session transcripts only | `Checkpointer` interface with shipped backends (memory / SQLite / Postgres) and a community pattern for more |
| Inject user context that preempts in-flight work | `UserMessage` via `push_front`, stubs cohort with `[detached]`, fires model on the new state | `interrupt()` + `Command(resume=...)` — graph-level pause and replay, not per-cohort rebase |
| Inject user context that does *not* preempt in-flight work | `UserQueuedMessage`, buffers, coalesces into next `UserMessage` after cohort drains | Application-defined |
| Route different message sources to different inbox priorities | `push_front` vs `push_back` on `GatedDeque` | Application-defined |
| Wait for an event without polling, in-process | `Await(types)` gates `drain()` until matching event lands; the agent loop stays running and idle | LangGraph stops the graph at `interrupt()`; the orchestrating process can do anything and resume by re-invoking the graph |
| Deliver a message to another agent at a future time | `AgentSend(delay=N)` via `loop.call_later` | Application-defined (external scheduler) |
| Address a live agent by name and deliver into its inbox | `AgentSpawn(persistent=true)` + named registry; `AgentSend` delivers | Application-defined: long-running graph + external trigger (HTTP webhook, Postgres `LISTEN`, MQ subscription) + `Checkpointer` |
| Typed serialisable run state | Per-event dataclasses; JSONL session transcript | Typed `State` dict per graph; serialised by the `Checkpointer` |
| Stream typed intermediate execution events | `RuntimeEvent` observer fan-out (cost, session, REPL, budget) | `astream_events` / `astream_log` over the Runnable / graph |
| Resume the same conversation on a fresh process | `--continue` reloads session transcript and replays state | `Checkpointer` restores graph state from durable storage |

Concurrent tool execution is at parity. Durable cross-process pause is
LangGraph's. Background tools, model-callable job control,
non-preempting user queue, inbox priority, delayed peer delivery, and
persistent live agents are sagent primitives; on LangGraph they are
patterns you assemble.

On streaming events: sagent fans `RuntimeEvent` dataclasses to
in-process observers; LangChain's `astream_events` exposes a richer
chain-of-Runnables view with per-Runnable hooks and metadata. LangChain
streams from anywhere in a composed pipeline; sagent streams from one
loop.

## Tools

LangChain: `@tool` decorator → `BaseTool` with Pydantic args, bound via
`model.bind_tools([...])`, dispatched by `ToolNode`.

Sagent: `Tool` is a `Protocol`:

```python
class Tool(Protocol):
    name: str
    tool_id: str
    description: str
    directive_schema: JSON
    supports_microcompaction: bool

    def summary(self, args: Mapping[str, object]) -> str: ...
    def summary_result(self, result: ToolResult) -> str | None: ...
    def prompt(self) -> str: ...
    async def run(self, args: Mapping[str, object]) -> ToolResult: ...
```

`prompt()` contributes a per-request system-prompt fragment;
`summary()` / `summary_result()` produce UI labels; optional
`CompactRestorable.post_compact_restore` rehydrates tool state after
compaction. `Agent` itself implements the same shape — `AgentSpawn` is
a tool whose `run` builds a child `Agent`. Recursive composition falls
out of the protocol; no separate orchestration layer.

## Providers

LangChain: one `BaseChatModel` per provider in its own package
(`langchain-openai`, `langchain-anthropic`, …), factory via
`init_chat_model("openai:gpt-...")`, swap via
`Runnable.configurable_alternatives`.

Sagent: one `Provider` / `Model` protocol across Anthropic, OpenAI,
Google, Moonshot, DashScope, MiniMax, llama.cpp, and self-hosted
Transformers. Cost, cache-control, extended-thinking, effort,
retry-classification, and overflow-classification all normalise on the
protocol, so the agent loop does not see provider-specific behaviour
across a mid-session swap.

## Multi-agent

LangGraph: subgraphs, supervisor patterns, `Send` fan-out. Topology
first.

Sagent: three runtime primitives.

- `AgentSelf`: inspect or mutate own state (status, compaction, model
  swap, token limits).
- `AgentSpawn`: build a child agent with explicit tool/depth limits.
- `AgentSend`: deliver to another live named agent's inbox.
  Peer-to-peer with delayed delivery.

Agents spawn and send at execution time, like processes over pipes.

## Memory and compaction

LangChain: `ConversationBufferMemory`, `ConversationSummaryMemory`,
`ConversationSummaryBufferMemory`, vector-backed memory. LangGraph adds
`Checkpointer` durability and a memory store API.

Sagent:

1. **Session persistence** — per-cwd JSONL transcripts, replayable.
2. **Full compaction** — `Compactor` protocol; writes
   `pre_compact_<N>.jsonl`, runs post-compact enrich (file reattach,
   status injection, tool restore), retries on prompt-too-long.
3. **Microcompaction** — per-tool result trimming and disk offload via
   `supports_microcompaction`; `post_compact_restore` rehydrates state.

No cross-process `Checkpointer` equivalent. No LangChain equivalent of
microcompaction or `CompactRestorable`.

## RAG and the breadth question

Roughly half of LangChain's surface area is RAG and adjacent: ~200
document loaders (PDF, HTML, Notion, Confluence, GDrive, S3, …), code-
and language-aware text splitters, ~30 embedding providers, ~80 vector
store integrations, and retrievers with re-ranking, MMR, parent-document,
multi-query, and self-query patterns. Output parsers (Pydantic, JSON,
retry-on-fail) sit on the same layer.

Sagent ships none of this. Coding agents typically read files directly
through a `Read` / `Grep` / `Glob` tool surface, which has been a
reasonable default for the workloads sagent targets. If you need RAG
over ten million Confluence pages, sagent expects you to either write
the ingestion + retrieval as a tool or use LangChain's prebuilt stack.

Sagent is not a LangChain replacement. It is a different shape that
overlaps in the agent loop only.

## Observability and evals

LangChain ships `astream_events` / `astream_log` plus LangSmith for
hosted traces, datasets, and evals.

Sagent's runtime publishes `RuntimeEvent` items to in-process observers.
Built-in observers cover cost tracking, session writes, REPL rendering,
budget caps, and tool labels. Streaming text and thinking flow through
`on_text` / `on_thinking` callbacks. There is no hosted observability
surface; the cost tracker and JSONL transcripts are the default
instrumentation.

## Typing and size

LangChain leans on Pydantic; the surface is large and type fidelity
varies across sub-packages. Sagent's runtime contract is dataclasses
plus `runtime_checkable` Protocols, basedpyright-clean as a project
rule. The runtime engine fits in `custom_types.py` plus
`agent/runtime.py`; a contributor can read the whole agent layer in an
afternoon.

LangChain's install graph includes dozens of optional integration
packages. Sagent is one wheel plus one `[selfhosted]` extra for local
Transformers.

## What sagent has that LangChain does not

Each item is a capability the agent unlocks, followed by the runtime
mechanics that deliver it. None are theoretically impossible in
LangChain; they are awkward enough to retrofit that almost no
LangChain-based agent has them.

- **The user can interrupt mid-cohort and the agent keeps the prior
  work running.** Type while three tools are in flight; the new message
  preempts and the tools finish in the background.
  - `[detached]` placeholder results stub unfinished cohort members so
    the provider's `tool_use` / `tool_result` invariant stays valid.
  - The runtime fires the model on the new `UserMessage` immediately.
  - Detached tools keep running; their real results land as
    `DetachedResult` user-context in the next round.
  - LangGraph's `interrupt()` pauses the graph but does not rebase the
    in-flight tool cohort against the provider invariant.

- **The user has five distinct verbs for controlling in-flight work**
  instead of one "stop" button.
  - `halt`: cancel the model call, gate on user input, leave tools
    running.
  - `kill <id|all>`: cancel tasks, drop from cohort.
  - `detach <id|all>`: stub now, finish in the background.
  - `undetach <id|all>`: re-gate on a previously detached tool.
  - `clear`: cancel model, detach all tools, wipe history, gate.

- **The agent rewrites its own runtime config mid-session** — model
  swap, compact, rebudget tokens — via one tool call.
  - `AgentSelf` covers `status`, context verbs (`clear`/`compact`/
    `recompact`), model swap, token-budget rebudgeting, provider
    `model_options`, diagnostics, and catalog.
  - LangChain's `configurable_alternatives` + `with_config()` is
    caller-driven; the agent cannot mutate its own config.

- **Agents can address other live agents by name and deliver into
  their inbox**, with optional delayed delivery.
  - `AgentSend` writes `UserMessage(text=f"[from {sender}]: {content}")`
    into the target's `GatedDeque`.
  - `AgentSpawn(persistent=true)` registers the child in the
    live-agent registry under a name.
  - `delay=N` uses `asyncio.get_running_loop().call_later` for future
    delivery.
  - The tool's `prompt()` contribution dynamically lists currently-live
    peers.
  - LangGraph's `Send` is fan-out within one graph step; no named
    addressability across runs, no delayed delivery.

- **One inbox unifies every input source**, so there is no separate
  control plane per surface.
  - REPL keystrokes, Slack messages, peer `AgentSend`s, background
    task completions, model responses, tool results, and runtime
    commands all land on the same `GatedDeque`.
  - `push_front` vs `push_back` gives priority control.
  - `Await(types)` blocks `drain()` without polling until a matching
    event lands.

- **Tools survive history compaction** without losing their state.
  - `supports_microcompaction` opts a tool into per-result trimming
    and disk offload.
  - `CompactRestorable.post_compact_restore` rehydrates tool-specific
    context after full compaction.
  - LangChain has summary memory but no per-tool restoration hook.

- **Context overflow recovers automatically inside the runtime**, not
  at the application layer.
  - A prompt-too-long error triggers in-flight compaction.
  - The request retries on the compacted history without the
    application seeing the failure.

- **Provider swap mid-session does not leak provider-specific
  behaviour into the agent loop.**
  - One `Model` protocol normalises cache TTL, prompt-cache
    breakpoints, extended-thinking, effort hints, retryable-error
    classification, and overflow classification.
  - The agent loop sees identical semantics across Anthropic, OpenAI,
    Google, Moonshot, DashScope, MiniMax, llama.cpp, self-hosted, and
    OpenAI-compatible endpoints.

- **The same `Agent` runs behind CLI, REPL, Slack, parent agents, and
  the library API.** Surfaces differ only in how messages enter the
  inbox and how events render; the agent logic does not fork.

- **The `Bash` tool understands shell ASTs**, so it parallelises
  read-only commands, tracks `cd`, and routes the model to dedicated
  tools.
  - `bashlex` parses cached per request; `is_read_only` classifies
    side-effects per-command (git, sed, type-checker, flag handling).
  - `unwrap_cd_prefix` normalises `cd X && CMD` so cwd tracks and
    matchers see the real command.
  - Sibling tools (`Edit`, `List`, `Glob`, `Grep`) implement
    `bash_match(trees)` so a `Bash` call to `ls`/`sed`/`find`/`grep`
    nudges the model toward the dedicated tool.
  - LangChain's `ShellTool` wraps `subprocess.run` with none of this.

## Where sagent is weaker

- No durable cross-process pause/resume. Sagent persists transcripts,
  not mid-run state. Use LangGraph's `Checkpointer` + `interrupt()` if
  you need "pause for approval, resume next Monday on another machine."
- No RAG layer. No loaders, splitters, embeddings, vector stores,
  retrievers, or re-rankers. `Read` / `Grep` / `Glob` is the whole
  retrieval surface.
- No graph DSL for non-agent LLM pipelines (classify → route → extract
  → summarise). The only loop is the agent loop.
- No general-purpose output parsers. Tool calls validate against
  `directive_schema` (JSON Schema); structured extraction outside the
  tool-call path is not covered.
- Smaller prompt-template system. YAML recipes with `.format()` /
  `.replace()` substitution and per-tool `Tool.prompt()` contributions.
  No `ChatPromptTemplate`, Jinja2 loops, or few-shot example selectors.
- No hosted observability. No LangSmith. You get in-process
  `RuntimeEvent` observers, a cost tracker, and JSONL transcripts.
- Narrower provider coverage. No native Cohere, Bedrock, Vertex, Azure
  OpenAI, Together, Replicate, Mistral, or Groq. Use them via the
  OpenAI-compatible adapter when they expose one.
- No MCP, no LSP, no native sandbox, no tree-sitter repo map, no
  browser automation. aider's PageRank-ranked repo map has no analogue
  here.
- Locked runtime. `AgentRuntime` is fixed by contract. If the loop's
  opinions don't fit, fork.
- Tiny ecosystem. No community integrations, no books, no courses.
  LangChain has 135k+ GitHub stars; sagent does not.
- Python 3.12+ required. LangChain runs on 3.10+.
- Single-process. Distributing agents across machines is on you.
- No batch / async-many API. `Agent.run` is single-input; LangChain
  Runnables expose `batch` / `abatch`.

## When to reach for which

Pick LangChain if you need any of:

- RAG, retrievers, vector stores, document loaders, output parsers.
- Durable cross-process pause/resume for human-in-the-loop.
- A graph DSL for non-agent LLM pipelines.
- LangSmith traces, evals, or prompt management.
- A provider sagent does not cover, or community integrations.

Pick sagent if you need any of:

- One inbox unifying REPL, Slack, peer agents, and background tasks.
- Mid-turn user injection, halt/kill/detach/clear, or peer-to-peer
  agent messaging as primitives.
- The agent itself swapping models, compacting context, or spawning
  peers via tool calls.
- A small Python package you can read end-to-end before depending on it.

LangChain answers "framework for LLM applications." Sagent answers
"typed runtime for one kind of agent." Pick by the question, not the
brand.
