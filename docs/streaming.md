# Streaming and machine-readable output

Sagent has two streaming layers:

1. Model streaming: `Model.stream(request)` yields `TextChunk`s as the reply is written, then the finished `Response`.
2. Runtime events: `agent.runtime` publishes every event of a turn -- text, thinking, tool calls, tool results, compactions, idle.

The CLI exposes runtime events through `--output-format stream-json`.

## Reply text

`Agent.stream()` yields the reply's text as it is written:

```python
async for words in agent.stream("Explain this repo"):
    print(words, end="")
```

## Runtime events

Iterate `agent.runtime` for every event published from then on, or add a callback to `agent.runtime.observers`:

```python
from sagent.runtime import types
from sagent.session.types import Succeeded


def observe(event: types.RuntimeEvent) -> None:
    if isinstance(event, types.ModelResponseThinkingStreamed):
        print(event.text, end="")
    elif isinstance(event, types.ModelResponseCompleted):
        if isinstance(event.outcome, Succeeded):
            print(event.outcome.value.spend.total)


agent.runtime.observers.append(observe)
reply = await agent.send("Explain this repo")
agent.runtime.observers.remove(observe)
```

Common events in `sagent.runtime.types`:

| Event | Meaning |
| --- | --- |
| `ModelResponseStarted` | A model call began. |
| `ModelResponseTextStreamed` | Reply text chunk. |
| `ModelResponseThinkingStreamed` | Thinking text chunk, when the provider returns it. |
| `ModelResponseCompleted` | A model call ended. Its `outcome` is `Succeeded` with the finished `Response` (records, tool calls, tokens, spend), `Cancelled`, or `Failed` with what was raised. It says nothing of the turn: after a call that failed the turn may go on. |
| `ModelResponseSuspended` | A transient failure; the turn waits, then calls again. |
| `ModelIdle` | The model's work on the turn stopped. Its `outcome` is `Succeeded` (it finished), `Cancelled` (a halt or a clear stopped it), or `Failed` with what was raised, past retrying. |
| `UserMessageCompleted` | A person's message was answered, or was not: with its turn's outcome, or `Cancelled` when no turn took it in (a clear dropped it; a halt came before the call for it; the runtime quit). `await runtime.ask(message)` returns the outcome of the message it was given. |
| `ModelRetryCompleted` | A retry called the model again for a stopped turn (`Succeeded`), or found none stopped (`Cancelled`). |
| `SessionCompactStarted`, `SessionCompactCompleted` | Compaction began, and ended: `Succeeded`, `Cancelled`, or `Failed`. |
| `ChildEvent` | A spawned child's event. |
| `ModelHalt` | Someone asked to halt the turn. |
| `AgentIdle` | The agent has no more work and waits for input. |

A tool result arrives as the session record it is, a `ToolMessage`.

## CLI output formats

```bash
printf 'Say hi' | sagent --provider Google --model gemini-pro-3.1
printf 'Say hi' | sagent --provider Google --model gemini-pro-3.1 --output-format json
printf 'Say hi' | sagent --provider Google --model gemini-pro-3.1 --output-format stream-json
```

`text` prints final message content.

`json` prints:

```json
{"content":"final answer"}
```

`stream-json` prints newline-delimited JSON event records. Each record includes at least `descriptor` and `content`. The final record has `descriptor: "result"`.

```json
{"descriptor":"text/plain","content":"Hello"}
{"descriptor":"result","content":"Hello"}
```

| Descriptor | Meaning |
| --- | --- |
| `text/plain` | Reply text chunk. |
| `text/x-thinking` | Thinking text chunk. |
| `text/x-tool-label` | Label of a tool call, with its `call_id`. |
| `application/x-tool-result` | Tool result, with `call_id` and `is_error`. |
| `application/x-error` | The turn failed. |
| `text/x-notice` | Retry wait or login switch. |
| `result` | Final reply. |

A failed run prints `{"error": "..."}` and exits 1.

## CLI input formats

`--input-format text` reads all stdin as one prompt.

`--input-format stream-json` reads newline-delimited JSON objects and collects each object's `prompt` field. Blank lines and objects without `prompt` are ignored. Prompts are joined with blank lines into one agent request.

```bash
cat <<'EOF' | sagent --provider Google --model gemini-pro-3.1 --input-format stream-json --output-format json
{"prompt":"Summarize file A."}
{"prompt":"Then compare it with file B."}
EOF
```

## Caveats

`stream-json` is a machine-readable event format, not a stable RPC protocol. Descriptors are the compatibility surface; exact content strings may change as tool summaries and provider messages evolve.

Headless `stream-json` input is prompt-oriented: it does not replay a full prior conversation. Use sessions when you need conversation continuity.
