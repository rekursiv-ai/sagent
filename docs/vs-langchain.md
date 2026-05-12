# Sagent vs LangChain/LangGraph

LangChain is the closest Python-shaped neighbour to sagent on the
adjacent-projects table, so it gets a deep comparison. The short version:
**LangChain is a ~700-package ecosystem covering RAG, agents, evals, and
LLM pipelines of every shape; sagent is a single-package runtime
specialised for coding agents, which deliberately omits most of what
LangChain ships.** They overlap on "call an LLM with tools" and diverge
nearly everywhere else. If your problem is broad LLM-application
plumbing, sagent is the wrong tool — that is not a slight, it is the
design.

## What each one is

**LangChain** is an ecosystem. `langchain-core` ships the `Runnable` /
LCEL interfaces; `langchain` ships chains, retrievers, and output
parsers; `langgraph` ships state-machine agents with durable
checkpointing; `langsmith` provides hosted traces and evals; and a few
hundred integration packages plug in chat models, embedding models,
vector stores, document loaders, and tools. The project is positioned as
a framework for *any* LLM application: RAG, agents, extraction,
classification, ETL.

**Sagent** is a single Apache-licensed Python package focused on coding
agents. It ships:

- Five typed contracts: `Message`, `Tool`, `Model`, `Provider`, `Agent`.
- One locked runtime: `AgentRuntime.run_forever`, an inbox-driven loop.
- Built-in tools for files, shell, search, web, papers, Slack, Linear,
  audio, wiki access, skills, background jobs, and agent coordination.
- A CLI, a `prompt_toolkit` REPL, a Slack Socket Mode surface, and a
  Python library API — all on top of the same `Agent` object.

Sagent deliberately omits document loaders, embedding models, vector
stores, retrievers, re-rankers, output parsers, and a graph DSL.

## Execution model

This is the sharpest architectural difference.

### LangGraph: state-machine workflow runner

LangGraph models an agent as a `StateGraph`. Nodes are functions over a
typed `State`; edges are conditional. `ToolNode` invokes tools in
parallel via `asyncio.gather` when the model emits multiple calls in a
turn. `astream` and `astream_events` expose token-level streaming and
intermediate events.

Durable interrupts are first-class. A node calls `interrupt(payload)`;
the configured `Checkpointer` (memory, SQLite, Postgres) persists state;
control returns to the caller. You resume by re-invoking the graph with
`Command(resume=value)`. The node re-runs from the top up to the
interrupt call. This is designed for human-in-the-loop approval that
must survive process restart.

### Sagent: drain-driven inbox loop

```text
while True:
    drain inbox into RuntimeEvent batches
    dispatch each event in one match block
    if cohort empty and no model call in flight and no compaction pending:
        call model
```

Everything is a `RuntimeEvent` on a `GatedDeque`: user text, model
response chunks, tool results, halt, kill, clear, peer-agent messages,
background task completions, compaction completions. One loop, one pipe,
one match block.

Tool calls run as `asyncio.Task`s tracked in a cohort set. The model
gate fires only when the cohort drains. Five verbs control in-flight
work:

- `halt`: cancel the model call, gate on user input, leave tools running.
- `kill <id|all>`: cancel running task(s), drop them from the cohort.
- `detach <id|all>`: stub unfinished tools with `[detached]` placeholders
  so the provider's `tool_use` / `tool_result` invariant stays valid,
  let them finish in the background, deliver their results as
  `DetachedResult` events into the next round.
- `undetach <id|all>`: re-gate the model on a previously detached tool.
- `clear`: cancel the model, detach all tools, wipe history, gate.

User input that arrives mid-cohort uses the detach machinery
automatically — type while three tools are running, the runtime stubs
them, appends the user message, fires the model, and the detached tools
land as `DetachedResult` user-context in the next round.

`Await(types)` gates the deque: `Halt`, `Clear`, and `ModelResponseError`
push an `Await` so `drain()` blocks until a matching event (or `Quit`)
arrives. That's how the runtime waits for user input without polling.

### Where each model is better

LangGraph wins when:

- You need durable cross-process pause/resume. The `Checkpointer` is
  built for this; sagent persists sessions but does not model mid-run
  pause across processes.
- You want declarative branching/looping topology the framework
  enforces.
- You want human-in-the-loop gates that survive restart.

Sagent wins when:

- You want mid-turn user injection that handles the provider invariant
  for you, without rebuilding the agent graph.
- You want multiple input sources (REPL, Slack, peer agents, background
  tasks) unified on one loop with one control plane.
- You want fine-grained in-flight verbs (`halt` / `kill` / `detach` /
  `undetach` / `clear`) as runtime primitives.
- You want peer-to-peer agent messaging at runtime, not a static graph.

Neither model is strictly "better." LangGraph optimises for durability
and explicit control flow. Sagent optimises for dynamic interactivity
and one-loop simplicity. Reinventing one inside the other would yield
the other.

## Tools

Both frameworks let you wrap a function and let the model call it. The
abstractions diverge on what else the tool layer carries.

**LangChain:** `@tool` decorator → `BaseTool` with Pydantic-typed args.
Bound to a model via `model.bind_tools([...])`. Dispatch is done by
`ToolNode` inside a graph.

**Sagent:** `Tool` is a `Protocol` with metadata the runtime consumes:

```python
class Tool(Protocol):
    name: str
    tool_id: str                 # MIME-style identifier
    description: str
    directive_schema: JSON       # JSON Schema for args
    supports_microcompaction: bool

    def summary(self, args: Mapping[str, object]) -> str: ...
    def summary_result(self, result: ToolResult) -> str | None: ...
    def prompt(self) -> str: ...
    async def run(self, args: Mapping[str, object]) -> ToolResult: ...
```

Plus an optional `CompactRestorable.post_compact_restore` so tools can
rehydrate their state after the conversation is summarised.

Concrete differences:

- `prompt()` lets each tool contribute a per-request system-prompt
  fragment. The agent assembles them.
- `summary()` / `summary_result()` produce pre- and post-execution UI
  labels.
- `supports_microcompaction` flags whether old results are eligible for
  microcompaction.
- `post_compact_restore` reinjects tool-specific context after
  compaction.

`Agent` itself implements the same shape. `AgentSpawn` is a tool whose
`run` builds a child `Agent` and returns its final message as a
`ToolResult`. Recursive agent composition falls out of the tool
protocol; there is no separate orchestration layer.

## Providers

LangChain has one `BaseChatModel` per provider, each in its own package
(`langchain-openai`, `langchain-anthropic`, etc.). `init_chat_model(
"openai:gpt-...")` gives a string-keyed factory.
`Runnable.configurable_alternatives` allows runtime swap.

Sagent has one `Provider` / `Model` protocol that all backends
implement. Anthropic, OpenAI, Google, Moonshot, DashScope, MiniMax,
llama.cpp, and self-hosted (Transformers) all normalise to the same
`ModelRequest` → `ModelResponse` shape. Cost, cache-control,
extended-thinking, effort hints, persistent-retry, context-overflow
classification, and image limits live on the same protocol. Mid-session
swap is a first-class agent verb (`Agent.swap_model`, exposed to the
model itself via the `AgentSelf` tool).

LangChain can swap models. Sagent normalises the runtime semantics
around the swap (cache TTL, prompt-cache breakpoints, overflow recovery,
retry classification) so the agent loop doesn't see provider-specific
behaviour.

## Multi-agent

LangGraph multi-agent: subgraphs, supervisor patterns, `Send` API for
fan-out. Multi-agent is a graph topology you build. Subgraphs are real
sub-runtimes; sagent's table marks LangChain ✅ on recursive spawn and
fully-detached multi-agent because LangGraph genuinely supports it.

Sagent multi-agent: three primitives in the runtime.

- `AgentSelf`: the agent inspects or mutates its own state — diagnostics,
  status, compaction, history clear, model swap, token-limit changes.
  Exposed to the model as a tool.
- `AgentSpawn`: build a child agent with explicit tool/depth limits.
  Children inherit provider, model, and tool knobs unless overridden.
  Recursion is depth-bounded.
- `AgentSend`: deliver a message to another live named agent's inbox.
  Peer-to-peer, not parent-only. Persistent named children stay in the
  live-agent registry and accept future sends.

LangGraph multi-agent is *spec-first*: write the topology, instantiate
it. Sagent multi-agent is *runtime-first*: agents spawn and send at
execution time, like processes talking over pipes.

## Memory and compaction

LangChain ships memory classes: `ConversationBufferMemory`,
`ConversationSummaryMemory`, `ConversationSummaryBufferMemory`, plus
vector-backed memory. LangGraph adds a `Checkpointer` for state
durability and a memory store API.

Sagent separates three concerns:

1. **Session persistence.** Per-cwd JSONL transcripts, replayable.
2. **Full compaction.** A `Compactor` protocol with `should_compact` and
   `compact`. Writes `pre_compact_<N>.jsonl` transcripts when a session
   directory is set, runs a post-compact enrich pipeline (file
   reattach, status injection, tool restore), and supports
   prompt-too-long retry that triggers compaction in-flight.
3. **Microcompaction.** Per-tool result trimming and disk offload. Tools
   opt in via `supports_microcompaction`. Tools needing rehydration
   implement `post_compact_restore` to reinject their state into the
   compacted history.

LangChain has nothing equivalent to microcompaction or the
`CompactRestorable` hook. Conversely, sagent has nothing equivalent to
LangGraph's cross-process `Checkpointer`; sessions are durable, but the
runtime is not designed to pause mid-tool-call across processes.

## RAG and the breadth question

Roughly half of LangChain's surface area is RAG and adjacent: ~200
document loaders (PDF, HTML, Notion, Confluence, GDrive, S3, …), code-
and language-aware text splitters, ~30 embedding providers, ~80 vector
store integrations, and retrievers with re-ranking, MMR, parent-document,
multi-query, and self-query patterns. Output parsers (Pydantic, JSON,
retry-on-fail) sit on the same layer.

Sagent ships none of this. Coding agents typically read files directly
through a `Read` / `Grep` / `Glob` tool surface; for the workloads
sagent targets, that's been a reasonable default. If you need RAG over
ten million Confluence pages, sagent expects you to either write the
ingestion + retrieval as a tool or use LangChain's prebuilt stack.

This is the single most useful disambiguator for a reader landing on
sagent expecting "Python agent framework": **sagent is not a LangChain
replacement; it is a different shape that overlaps in the agent loop
only.**

## Observability and evals

LangChain ships `astream_events` / `astream_log` plus LangSmith for
hosted traces, datasets, and evals.

Sagent's runtime publishes `RuntimeEvent` items to in-process observers.
Built-in observers cover cost tracking, session writes, REPL rendering,
budget caps, and tool labels. Streaming text and thinking flow through
`on_text` / `on_thinking` callbacks. There is no hosted observability
surface; cost tracker and JSONL transcripts are the default
instrumentation.

## Typing and size

LangChain leans on Pydantic; the surface is large and type fidelity
varies across sub-packages. Sagent's runtime contract is dataclasses
plus `runtime_checkable` Protocols, basedpyright-clean as a project
rule. The runtime engine fits in `custom_types.py` plus
`agent/runtime.py`; a contributor can read the whole agent layer in an
afternoon.

LangChain's install graph includes dozens of optional integration
packages. Sagent is one wheel with one optional extra (`[selfhosted]`
for local Transformers).

## What sagent has that LangChain does not

A list this short is the honest one. Most things LangChain "doesn't
have" it could ship in a week if it wanted to; the items below are
shaped by sagent's opinionated runtime and are awkward to retrofit into
a building-block framework.

- **Cohort detach for mid-turn user injection.** When the user types
  while a cohort of tool calls is in flight, the runtime stubs
  unfinished calls with `[detached]` placeholders so the provider's
  `tool_use` / `tool_result` invariant stays valid, fires the model on
  the user message, and delivers the detached results into the next
  round as `DetachedResult` user-context. LangGraph can `interrupt()`,
  but it does not transparently rebase mid-cohort against the provider
  invariant — that work falls on you.
- **Five in-flight verbs as primitives.** `halt` / `kill` / `detach` /
  `undetach` / `clear` are runtime events with documented semantics,
  not application-defined patterns.
- **`AgentSelf` exposed as a model-callable tool.** The model can swap
  its own backend, compact its own context, clear its own history, and
  change its own token limits via ordinary tool calls. LangChain agents
  do not have a built-in self-mutation surface the model can drive.
- **`AgentSend` peer-to-peer between live named agents.** LangGraph's
  `Send` is fan-out within a graph; sagent's `AgentSend` delivers into
  another live agent's inbox by name, with no shared graph required.
- **One inbox unifying user / Slack / peer / background / commands.**
  REPL keystrokes, Slack messages, peer-agent sends, and background
  task completions all land on the same `GatedDeque`. The agent does
  not need a separate control plane per source.
- **Microcompaction with `CompactRestorable.post_compact_restore`.**
  Tools opt in to per-result trimming and rehydrate their state after
  full compaction. LangChain has summary memory but no per-tool
  restoration hook.
- **Prompt-too-long recovery as an event.** Context overflow on a
  request triggers in-flight compaction and retry inside the runtime,
  not at the application layer.
- **Hot-swappable providers with normalised semantics.** Cache TTL,
  prompt-cache breakpoints, extended-thinking, effort hints,
  retryable-error classification, and overflow classification all live
  on one `Model` protocol. LangChain models expose provider-specific
  knobs that do not all line up.
- **Same `Agent` behind CLI, REPL, Slack, parent agents, and library
  calls.** Surfaces differ in how messages enter the inbox and how
  events are rendered; the agent logic does not fork per surface.
- **Full `bashlex` AST analysis on the `Bash` tool.** Parsing is cached
  per request. `is_read_only` classifies a parse as side-effect-free
  using per-command rules (git subcommand safety, sed mutation
  detection, type-checker mutation, leading-flag handling), enabling
  the runtime to run multiple read-only bash calls concurrently while
  serialising writes. `unwrap_cd_prefix` normalises `cd X && CMD` into
  `(cwd, CMD)` so the runtime tracks cwd shifts and matchers see the
  real command. `match_pipeline` extracts clean two-command pipelines
  for matcher analysis, and stdout-redirect detection distinguishes
  fd-1 redirects (which change what the model sees) from cosmetic
  stderr redirects. Sibling tools (`Edit`, `List`, `Glob`, `Grep`)
  implement a `bash_match(trees)` hook so when the model reaches for
  `ls`, `sed`, `find`, or `grep` through Bash, the tool result nudges
  it toward the dedicated tool. LangChain's `ShellTool` is a thin
  wrapper around `subprocess.run` with no AST analysis, no read-only
  classification, and no peer-tool routing.

None of these are theoretically impossible in LangChain. The point is
that sagent ships them as primitives, and they are awkward enough to
build on top of LangGraph that almost no LangChain-based agent actually
has them.

## Where sagent is weaker

Read this section if you are evaluating sagent for production work. The
gaps below are real and not all fixable by "writing a tool for it."

- **No durable cross-process pause/resume.** LangGraph's `Checkpointer`
  + `interrupt()` is built for human-in-the-loop approvals that survive
  process restart, queue across days, and resume on a different machine.
  Sagent persists session transcripts but the runtime expects to keep
  running. If your workflow needs "pause for human approval, come back
  on Monday," LangGraph is the right tool, not sagent.
- **No RAG layer.** No document loaders, no text splitters, no embedding
  models, no vector stores, no retrievers, no re-rankers. If your
  problem is retrieval-heavy, sagent gives you `Read` / `Grep` / `Glob`
  and expects you to write the rest. For coding agents over a local
  tree this is often fine; for RAG over a ten-million-page corpus it is
  not.
- **No graph DSL.** Heterogeneous LLM pipelines (classification →
  routing → extraction → summary) are easy to express in LangGraph and
  awkward in sagent, where the only loop is the agent loop.
- **Tool-call validation, but no general-purpose output parsers.**
  Sagent validates tool-call directives against each tool's
  `directive_schema` (JSON Schema), which covers the
  structured-output-from-the-model case for tool use. It does not ship
  the broader LangChain surface — `PydanticOutputParser`,
  `JsonOutputParser`, retry-on-fail wrappers, `OutputFixingParser`,
  structured-extraction chains — for parsing model output into Python
  objects outside the tool-call path.
- **A different shape of prompt-template system.** Sagent ships YAML
  recipes (`assets/sagent.yaml`, `bare.yaml`, `codex.yaml`) that map
  logical roles to markdown files, with `.format()` / `.replace()`
  placeholder substitution and per-tool `Tool.prompt()` contributions
  composed by the agent. It does not ship a runtime `PromptTemplate` class object,
  `ChatPromptTemplate` / `MessagesPlaceholder` multi-turn templating,
  Jinja2 conditionals/loops, or few-shot example selectors. For
  prompt-heavy work that depends on those primitives — semantic
  example selection over a dataset, dynamic chat-template composition,
  template inheritance with conditionals — LangChain's templating is
  more powerful.
- **No hosted observability.** No LangSmith equivalent for traces,
  datasets, evals, prompt management, or shared dashboards. You get
  in-process `RuntimeEvent` observers, a cost tracker, and JSONL
  transcripts.
- **Narrower provider coverage.** Anthropic, OpenAI, Google, Moonshot,
  DashScope, MiniMax, llama.cpp, self-hosted Transformers, and generic
  OpenAI-compatible endpoints. No first-class Cohere, Bedrock, Vertex,
  Azure OpenAI, Together, Replicate, Mistral, or Groq. You can use them
  through the OpenAI-compatible adapter when they expose one.
- **No MCP, no LSP, no native sandbox, no tree-sitter repo map, no
  browser automation.** All stated explicitly in the README. aider in
  particular has a tree-sitter repo map with PageRank ranking that
  sagent does not match for structural code awareness.
- **A locked runtime.** `AgentRuntime` is locked per an internal
  contract. The loop is opinionated by design. If sagent's opinions
  don't match your problem, your option is to fork; you cannot rewire
  the loop while keeping the rest of the package.
- **Tiny ecosystem.** No community-maintained integrations, no
  StackOverflow answers, no books, no courses, no third-party tools.
  LangChain has 135k+ GitHub stars and an enormous community surface;
  sagent has neither.
- **Python 3.12+ only.** LangChain runs on much older Python; sagent
  does not.
- **Single-process design.** Distributing agent work across machines or
  workers is on you. LangGraph plus a Postgres checkpointer at least
  gives you a starting point.
- **No batch / async-many API.** LangChain Runnables expose `batch` and
  `abatch` for parallel application of the same pipeline across many
  inputs. Sagent's `Agent.run` is single-input.

## When to reach for which

Reach for LangChain when:

- You need RAG, retrievers, vector stores, document loaders, or output
  parsers.
- You want LangGraph's durable cross-process `interrupt()` /
  Checkpointer for human-in-the-loop workflows.
- You want a graph DSL for heterogeneous LLM pipelines beyond agents.
- You want LangSmith traces, evals, prompt management.
- You need a provider sagent does not cover, or you need the long tail
  of community integrations.

Reach for sagent when:

- You are building a coding-style agent that runs locally and needs
  one inbox unifying REPL, Slack, peer agents, and background tasks.
- You need mid-turn user injection, in-flight halt / kill / detach /
  clear, or peer-to-peer agent messaging as primitives, not patterns.
- You want the agent itself to swap models, compact context, and spawn
  peers via ordinary tool calls.
- You want a small, opinionated, basedpyright-clean Python package you
  can read end-to-end before depending on it.

The frameworks are not in direct competition. LangChain is the answer
when the question is "framework for LLM applications." Sagent is the
answer when the question is "typed runtime for one kind of agent." If
your problem is "framework for LLM applications" and you reach for
sagent, you will have to build most of LangChain yourself.
