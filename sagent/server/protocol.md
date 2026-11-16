# Sagent daemon protocol

This is the canonical description of Sagent's long-lived daemon protocol.
The daemon is a transport host around native Sagent objects. It does not run a
second model loop, tool system, journal, or conversation engine.

## Two different machine-readable interfaces

Sagent has two machine-readable process interfaces with different purposes:

| Interface | Purpose | Lifetime |
| --- | --- | --- |
| `sagent --output-format stream-json` | One-shot CLI streaming | One prompt/run, then exit |
| `sagent.server` daemon | Interactive native session host | Long-lived process |

### `stream-json`

The existing `stream-json` mode is a CLI output format. It reads prompt input,
calls the normal `Agent.send()` path, emits streaming descriptors, emits a
final result, and exits. It is useful for scripts and pipes.

It is intentionally not a session RPC protocol. It does not provide request
IDs, session open/close, concurrent requests, session state, or a halt request.

### Daemon protocol

The daemon is a long-lived newline-delimited JSON protocol. It is needed when a
separate frontend must keep a session open and interact with it over time:

```text
request  -> daemon
response <- daemon
event    <- daemon
```

The daemon protocol is separate so the existing CLI semantics remain simple and
backward-compatible. It reuses Sagent's native runtime and Treekle values; it
does not replace `stream-json` or change what the terminal CLI means.

## Ownership boundaries

| Namespace | Responsibility |
| --- | --- |
| `system.*` | Process handshake and shutdown |
| `session.*` | Native session lifecycle, state, records, input, and halt |
| `catalog.*` | Optional catalog/provider operations |
| `commands.*` | Native command discovery and execution |

The vocabulary is declared in [`protocol.py`](./protocol.py). The dispatcher
implements the methods; it does not define another agent runtime.

## Canonical methods

```text
system.initialize
system.shutdown

session.open
session.list
session.state
session.records
session.team.state
session.agent.state
session.submit
session.halt
session.close

catalog.skills
catalog.modelSettings
catalog.modelSettings.set
catalog.usage

commands.list
commands.execute
```

The `session.*` methods are the daemon core. Catalog/model-settings methods are
optional provider-facing functionality and should not be confused with session
lifecycle.

Older unnamespaced methods such as `initialize`, `openSession`, and
`sendMessage` are not part of this native contract. If compatibility aliases
are ever required, they must be explicitly implemented and documented as a
separate compatibility layer.

## Requests and responses

Requests with an `id` receive one matching response:

```json
{"id":1,"method":"system.initialize","params":{}}
```

```json
{"id":1,"result":{"protocol":"sagent/v2"}}
```

Notifications have no request ID. They are emitted while requests continue to
be accepted:

```json
{"method":"session.event","params":{"...":"..."}}
```

## Native runtime events

The canonical runtime notification is `session.event`:

```json
{
  "method": "session.event",
  "params": {
    "protocol": "sagent/v2",
    "sessionId": "project-session",
    "sequence": 42,
    "eventType": "ModelResponsePartial",
    "category": "assistant-stream",
    "agentLabel": "Agent",
    "value": {
      "py/object": "sagent.runtime.types.ModelResponsePartial",
      "text": "hello"
    }
  }
}
```

The daemon preserves the native runtime event type and payload. `category` is a
stable classification of the native Sagent value, not a frontend presentation
type. It allows a client to recognize every concrete `ToolResult` subclass
without duplicating Sagent's class list.

When native Sagent behavior is not recoverable from plain JSON without
reimplementing Python, the envelope includes a `native` object. At present it
may contain:

- `text`, produced by `ToolResult.text()`;
- `summary`, produced by the selected agent's native tool implementation;
- `callId`, the native call identity with any detached-arrival marker removed;
- `arrival`, when a call or result is a detached-result delivery.

`ModelResponseComplete` additionally contains `records`, with each completed
response record using the same envelope as persisted session records. `value`
always remains the complete, untouched native event.

`ModelCallStarted.value` names the exact `provider` and provider-independent
`model` selected after Sagent's pre-call hook. This is the live source of truth
for model-call provenance; a frontend must not infer it from cached team state.

The daemon does not translate events into frontend concepts such as turns,
cards, message kinds, rendered tool rows, or visual completion state.

Non-runtime daemon notifications use `session.notification` with the same
metadata envelope. They are for daemon-level events such as command output or
usage updates, not replacements for native runtime events.

## Native records

`session.records` exposes the native journal records for hydration and replay.
Records are the source of truth for persisted conversation state. The daemon
does not need to create a second UI-shaped conversation snapshot.

The frontend adapts native records and live native events into its own view
model. Initial hydration and live updates should use the same frontend adapter.

Each record has its exact concrete `recordType`, stable native `category`, raw
`value`, and optional native-computed metadata. Journal ordering is the source
of replay ordering; the daemon does not manufacture record or turn IDs.

`TurnContext` records persist the effective `provider` and `model_id` for
replay. Its existing `model` field remains the restorable Sagent configuration
name and may include setting tags; clients should prefer `model_id` for a
provider/model identity label and fall back to `model` for older journals.

Session responses expose these records directly; they do not carry a second
UI-shaped conversation snapshot.

## Native runtime state

`session.state`, `session.team.state`, and `session.agent.state` project the
runtime's native lifecycle facts. These fields are intentionally distinct:

- `runtime.idle` means a user-side message submitted now can be answered now.
  It is a dispatch property, not a presentation-ready/busy flag.
- `runtime.busy` means foreground model, compaction, or tool work is active.
  Frontends should use this field for a working indicator.
- `runtime.waitingForInput` means the inbox gate is armed and the runtime is
  waiting for input. In this state `idle` may be false while `busy` is false.
- agent and session `running` are compatibility projections of `runtime.busy`.

This distinction is owned by `Runtime`. A frontend must not reconstruct busy
state by negating `idle` or by parsing event text.

## Treekle serialization

The daemon passes native values to Sagent's existing `treekle.json_dumps`
serializer when writing JSONL. Treekle owns encoding of dataclasses, tuples,
mappings, binary values, and other supported values.

The daemon envelope adds only transport and native semantic metadata:

- protocol version;
- event type;
- session ID;
- agent label;
- event sequence;
- stable native category;
- native-computed tool text or summary, when applicable.

The daemon must not maintain a second recursive serializer for Sagent values.

## TUI relationship

The TUI and daemon use the same native Sagent objects but have different
transport needs:

| TUI | Daemon |
| --- | --- |
| Direct `runtime.push(...)` | `session.submit` |
| Direct `runtime.halt()` | `session.halt` |
| Direct `runtime.observers` | `session.event` |
| Direct `agent.session` | `session.records` |
| Direct Team access | `session.team.state` |

The TUI should remain an in-process native frontend. It does not need to run
through the daemon, and the daemon does not contain TUI rendering logic.

## Protocol status

The daemon protocol is versioned as `sagent/v2`. The method names and envelope
shape above are the intended native contract. Reconnect/replay semantics for
missed event sequences are not yet defined; sequence numbers currently provide
ordering and a basis for a future replay mechanism.
