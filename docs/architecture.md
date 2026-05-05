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
