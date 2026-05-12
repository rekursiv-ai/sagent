# Sagent vs LangChain/LangGraph

LangChain is a ~700-package ecosystem for any LLM application: RAG,
agents, evals, extraction, classification, ETL — the agent is a
workflow the framework executes. Sagent treats agents as live,
addressable actors: each runs an inbox-driven loop, and users, peers,
background tasks, and the agent itself all communicate through that
inbox. The two overlap on "call an LLM with tools" and diverge
everywhere else.

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
restart — agent-as-workflow concerns. Sagent wins on mid-turn user
injection, multiple input sources on one loop, the five in-flight
verbs, and peer-to-peer agent messaging — agent-as-actor concerns.

## Runtime capability matrix

### Concurrent execution and background work

| Capability | Sagent | LangChain / LangGraph |
| --- | --- | --- |
| Multiple tool calls from one turn, concurrent | `asyncio.Task` per call, cohort `set[str]` gates the next model call | `ToolNode` + `asyncio.gather` |
| Single tool call async, result delivered later | `background: true` / `delay: N` on every tool via `BackgroundAwareTool` | Write a tool that returns a job id plus a separate poll tool |
| Model-callable job control | `BackgroundTask` with `list` / `cancel` / `foreground` | Application-defined |
| Registry of in-flight long-running ops | `BackgroundTaskEntry` (tool calls, persistent subagents, detached cohort, hidden infra) | Application-defined |

### Pause, persist, resume

| Capability | Sagent | LangChain / LangGraph |
| --- | --- | --- |
| Cross-process pause mid-run | None — sessions persist transcripts, not mid-cohort state | `interrupt()` + `Checkpointer` |
| Resume semantics | N/A | Node replays from the top up to the `interrupt()` call |
| Pluggable persistence backend | Local JSONL transcripts only | `Checkpointer`: memory / SQLite / Postgres |
| Resume the same conversation on a fresh process | `--continue` reloads transcript and replays | `Checkpointer` restores graph state |
| Typed serialisable run state | Per-event dataclasses; JSONL transcript | Typed `State` dict per graph |

### Mid-turn injection, inbox, peers, events

| Capability | Sagent | LangChain / LangGraph |
| --- | --- | --- |
| Inject user context that *preempts* in-flight work | `UserMessage` via `push_front`, stubs cohort with `[detached]`, fires model | `interrupt()` + `Command(resume=...)` — graph-level, not per-cohort rebase |
| Inject user context that does *not* preempt | `UserQueuedMessage`, coalesces into next `UserMessage` after cohort drains | Application-defined |
| Inbox priority routing | `push_front` vs `push_back` on `GatedDeque` | Application-defined |
| Wait without polling | `Await(types)` blocks `drain()` until match | LangGraph stops at `interrupt()`; the host re-invokes the graph |
| Address a live agent by name | `AgentSpawn(persistent=true)` + registry; `AgentSend` delivers | Long-running graph + external trigger + `Checkpointer` |
| Delayed delivery to another agent | `AgentSend(delay=N)` via `loop.call_later` | External scheduler |
| Stream typed intermediate events | `RuntimeEvent` observer fan-out | `astream_events` over Runnables — richer per-Runnable metadata |

Concurrent tool execution is at parity. Durable cross-process pause is
LangGraph's. The rest of the sagent column is primitives; the LangGraph
column there is patterns you assemble.

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

### Headline differentiators

All five below are facets of the same architectural commitment: agents
are live actors on a shared inbox, addressable by users, peers, and
themselves.

**1. Mid-cohort user injection that respects the provider's tool-call
invariant.** Type while three tools are in flight; the message preempts,
the model fires immediately, and the tools finish in the background.

- Unfinished cohort members get stubbed with `[detached]` placeholders,
  satisfying the `tool_use` / `tool_result` invariant.
- Detached tools keep running; real results arrive as `DetachedResult`
  in the next round.
- LangGraph's `interrupt()` pauses the graph but does not rebase
  in-flight tool cohorts against the provider.

**2. Five verbs for controlling in-flight work**, not one "stop" button.

- `halt` — cancel model, gate, leave tools running.
- `kill` — cancel tasks, drop from cohort.
- `detach` — stub now, finish in the background.
- `undetach` — re-gate on a detached tool.
- `clear` — cancel model, detach all, wipe history, gate.

**3. The agent rewrites its own runtime config via tool calls.** Model
swap, compaction, token rebudgeting — all addressable from inside the
loop.

- `AgentSelf` covers status, context verbs (`clear`/`compact`/
  `recompact`), model/provider swap, token budgets, provider
  `model_options`, diagnostics, catalog.
- LangChain's `configurable_alternatives` is caller-driven; the agent
  cannot mutate its own config.

**4. Named live-agent registry with inbox delivery.** Agents address
each other by name, with delayed delivery.

- `AgentSpawn(persistent=true)` registers under a name.
- `AgentSend` delivers into the target's `GatedDeque` (`delay=N` via
  `loop.call_later` for future delivery).
- `prompt()` contribution lists currently-live peers.
- LangGraph's `Send` is fan-out within one graph step; no addressing
  across runs, no delayed delivery.

**5. One inbox unifies every input source.** REPL, Slack, peer sends,
background completions, model responses, tool results, runtime commands
— all on the same `GatedDeque`, with `push_front`/`push_back` priority
and `Await(types)` blocking without polling. No separate control plane
per surface.

### Other primitives

- **Tools survive compaction.** `supports_microcompaction` opts a tool
  into per-result trimming; `CompactRestorable.post_compact_restore`
  rehydrates state. LangChain has no per-tool restoration hook.
- **Context overflow auto-recovers.** Prompt-too-long triggers
  in-flight compaction and retry; the application never sees it.
- **Provider swap normalises semantics.** One `Model` protocol covers
  cache TTL, prompt-cache breakpoints, thinking, effort, retry, and
  overflow classification across Anthropic, OpenAI, Google, Moonshot,
  DashScope, MiniMax, llama.cpp, self-hosted, OpenAI-compatible.
- **One `Agent` behind every surface.** CLI, REPL, Slack, parent
  agents, library API — surfaces differ only in message ingress and
  event rendering.
- **`Bash` understands shell ASTs.** `bashlex` parses cached per
  request; `is_read_only` parallelises reads and serialises writes;
  `unwrap_cd_prefix` tracks cwd shifts; sibling tools (`Edit`, `List`,
  `Glob`, `Grep`) implement `bash_match(trees)` to nudge the model
  toward dedicated tools. LangChain's `ShellTool` wraps
  `subprocess.run` with none of this.

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
"runtime for agents that talk — to users, peers, and themselves — over
one inbox." Pick by the question, not the brand.
