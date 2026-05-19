# Architecture

Sagent has five core contracts: `Message`, `Tool`, `Model`, `Provider`, and `Agent`.

## Drain-run-check loop

Most agent frameworks are request-response: receive one prompt, run, return. Sagent uses an inbox loop:

```text
while True:
    drain inbox into user messages
    call model
    if tool calls exist: dispatch tools and loop
    if inbox is empty and model is done: go idle
```

The agent wakes when new work lands in its inbox. REPL input, background task results, child-agent messages, Slack messages, and direct user prompts all enter through the same path.

This invariant is what lets Sagent support mid-turn injection, delayed messages, persistent child agents, Slack routing, and background task completion without separate control planes.

### Inbox semantics: root vs subagent

The drain-run-check loop is the *only* execution model. There is no separate "root agent" mode. The CLI's root agent, every `AgentSpawn` child (ephemeral or `persistent=true`), every Slack-routed agent, and every `Agent.run(...)` call from a Python embedder invoke the same `Agent.serve_forever()` → `AgentRuntime.run_forever()` loop.

Consequences:

- `AgentSend` to any live label wakes the recipient identically. `inbox.push_back` resolves the recipient's `inbox.drain()` await; the model-call gate fires on the next iteration when the history tail is a `UserMessage` or `ToolResult`.
- "Next response" in the `AgentSend` description means the recipient's next *assistant turn*, produced by the model-call gate — not the next time the user types. A delayed self-send (`AgentSend(to=<self>, delay=N)`) wakes the agent N seconds later with no user interaction required.
- There is no polling. An idle agent is blocked on `await inbox.drain()` consuming zero CPU; the wake is purely event-driven.
- `persistent=true` on `AgentSpawn` does not enable any extra wake mechanism. It only changes who owns the task that runs `serve_forever()` and how the label is registered. An idle root agent receives `AgentSend` messages with the same semantics as a persistent child.

When `AgentSend` messages appear to land in an inbox but the agent never responds, the cause is almost always one of:

1. A stuck in-flight model call (the gate cannot refire while `self.model_call is not None`; see `docs/private/bugs46.md` for the OpenAISubscription idle-watchdog gap). Stacked `[from X]` previews above the `>` prompt are diagnostic: the runtime is rendering `_mid_stream_queue`. `/tasks` will show `fg=1` on the stuck model call. `Ctrl+C` clears the stream and the queued messages drain.
2. A label collision routing the message to the wrong agent (regression pinned at `tools/agent_spawn_test.py::test_root_label_collision`).
3. The response did fire but produced an unremarkable assistant turn that scrolled past unnoticed.

## Messages

`Message` is a MIME-like typed payload. Text, bytes, JSON, tool calls, tool results, provider responses, and multipart assistant turns all use the same graph-shaped structure.

`TextMessage` is intentionally central. It is the common communication interface between providers, tools, sessions, compaction, and UI surfaces. High connectivity around `TextMessage` is an architecture invariant, not accidental coupling.

Messages carry IDs, parent IDs, timestamps, descriptors, and typed content. Serialization preserves that metadata so sessions can be replayed and repaired.

## Tools

A tool receives a JSON directive inside a `Message` and returns a `Message`. It also exposes a JSON Schema, a prompt section, a status summary, and a microcompaction flag.

There are two tool execution shapes:

- Batch tools: one await, one result.
- Streaming tools: async generator, intermediate events, final result.

The dispatch layer validates tool calls, emits UI events, supports backgrounding, enforces large-output budgets, and appends tool results in provider-valid form.

## Agent as tool

`Agent` follows the same interface pattern as a tool. It has `name`, `description`, `directive_schema`, `summary`, `prompt`, and `run`.

That lets agents compose recursively. `AgentSpawn` constructs a child agent, runs it, and returns the child's final message as a tool result. Persistent children remain in the live-agent registry and receive future `AgentSend` messages.

## Agent coordination triad

- `AgentSelf`: self-inspection and self-mutation, including diagnostics, compaction, model swap, history clear, and token limits.
- `AgentSpawn`: child-agent execution with inherited provider/model/tool knobs and optional depth limits.
- `AgentSend`: peer-to-peer messages between live named agents.

Together, these primitives support isolated review, map-reduce work, persistent background agents, and inbox-based coordination.

## Providers and models

Providers own authentication and construct models. Models accept typed requests and return typed responses with token counts, stop reason, IDs, cache usage, and cost.

Provider normalization happens at the edge. Inside the agent loop, Anthropic, OpenAI, Google, OpenAI-compatible endpoints, and other providers all produce the same `ModelResponse` shape.

## Compaction and persistence

Sagent separates three related concerns:

- Session persistence: durable local messages and metadata.
- Full compaction: summarizing old conversation when context is tight.
- Microcompaction/result storage: shrinking or offloading old/large tool outputs.

This keeps long sessions usable without making providers or tools know about storage details.

## Surfaces

The CLI, REPL, Slack service, parent agents, child agents, and Python applications all use the same `Agent` object. Surfaces differ in how they put messages into the inbox and render events; they do not own separate agent logic.
