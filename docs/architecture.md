# Architecture

Sagent has five core contracts: session records, `Tool`, `Model`, `Provider`, and `Agent`.

## Drain-dispatch-gate loop

Most agent frameworks are request-response: receive one prompt, run, return. Sagent's `Runtime.run_forever` is an inbox loop:

```text
while True:
    if nothing is in flight: publish AgentIdle
    drain the inbox
    dispatch each event
    call the model when the context ends in something to answer
      and nothing is in flight
```

The agent wakes when an event lands in its inbox. User input, tool results, background task results, and messages from other agents all enter the inbox; code outside the runtime adds one with `runtime.push(event)`.

This invariant is what lets Sagent support mid-turn input, delayed messages, persistent child agents, and background task completion without separate control planes.

### Inbox semantics: root vs subagent

The loop is the *only* execution model. There is no separate "root agent" mode. The CLI's root agent, every `AgentSpawn` child, and every `Agent.send(...)` from a Python embedder run on the same `Runtime.run_forever()` loop, started by `async with Agent(...)`.

Consequences:

- `AgentSend` to any live label wakes the recipient identically: the message is pushed to its inbox, and the model-call gate fires once the context ends in something to answer.
- "Next response" in the `AgentSend` description means the recipient's next *assistant turn*, not the next time the user types. A delayed self-send (`AgentSend(to=<self>, delay=N)`) wakes the agent N seconds later with no user interaction required.
- There is no polling. An idle agent is blocked on its inbox, consuming zero CPU; the wake is purely event-driven.
- A message the user sends while the model streams is answered next: the reply's calls start detached, and their results arrive later.

`Halt` stops the model call and waits for the user's next message; tool calls keep running. Meanwhile the inbox holds back every event except `Halt`, `Kill`, `Clear`, and `Compact`.

## Session records

The session is a sequence of typed, frozen records from `sagent.session.types`: `UserMessage`, `AssistantMessage`, `Thinking`, `ToolCall`, typed tool results, `SystemMessage`, compactions, and clears. The history is a tape that only grows; what the model sees of it is resolved from the last clear or compaction on.

Providers need user and model turns to alternate, and each call answered before the next turn. The runtime merges adjacent user-side turns, and repairs at the model-call gate what it cannot keep.

## Tools

A tool takes the model's JSON arguments and the calling agent's `ToolContext`, and returns the session record of what it did. It also exposes a JSON Schema, a summary, a terminal display, and a `serialize_key` saying which calls it must not overlap.

Tool calls run in one of two registries:

- `cohort`: the calls the next model call waits on.
- `detached`: calls already answered by a stub; each one's real result is delivered later as a new turn, so nothing the model has read is rewritten.

A call is detached when the model asks for `background` or `delay`, when it runs too long, or when the user interrupts its cohort.

## Agent coordination triad

- `AgentSelf`: self-inspection and self-mutation, including diagnostics, compaction, model and settings changes, and status.
- `AgentSpawn`: child-agent execution with inherited model, settings, and tools, each overridable, and optional depth limits.
- `AgentSend`: peer-to-peer messages between live named agents.

A `Team` holds the live agents of one session, by label, and who spawned whom. Together, these primitives support isolated review, map-reduce work, persistent background agents, and inbox-based coordination.

## Providers and models

Providers own authentication and serve models by name. A model's `stream(request)` yields text chunks, then a `Response` with records, token counts, spend, and stop reason.

Provider normalization happens at the edge. Inside the agent loop, Anthropic, OpenAI, Google, OpenAI-compatible endpoints, and other providers all produce the same `Response` shape.

## Compaction and persistence

Sagent separates three related concerns:

- Session persistence: the journal, one JSON line per record.
- Full compaction: summarizing old conversation when context is tight (`SummaryCompactor`).
- Pruning and result bounds: dropping old tool results (`Pruner`) and cutting large ones to fit.

This keeps long sessions usable without making providers or tools know about storage details.

## Surfaces

The CLI, REPL, parent agents, child agents, and Python applications all use the same `Agent` object. Surfaces differ in how they push events to the runtime and render what it publishes; they do not own separate agent logic.
