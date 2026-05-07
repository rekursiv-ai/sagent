# Streaming and machine-readable output

Sagent has two streaming layers:

1. Model streaming: providers can stream text chunks while still returning a complete `ModelResponse`.
2. Agent events: `Agent.run(..., events=queue)` emits text chunks, thinking, tool labels, tool results, and completion metadata as `Message` objects.

The CLI exposes agent events through `--output-format stream-json`.

## Python event queue

```python
import asyncio

from sagent.lib.json import json_freeze


events: asyncio.Queue = asyncio.Queue()
result = await agent.run(json_freeze({"prompt": "Explain this repo"}), events=events)
```

The queue receives `Message` objects during the run and a `None` sentinel at the request boundary.

Common event descriptors:

| Descriptor | Meaning |
| --- | --- |
| `text/plain` | Model text chunk or ordinary text event. |
| `text/x-thinking` | Provider thinking content when available. |
| `text/x-tool-label` | Human-readable label for a pending tool invocation. |
| `multipart/x-tool-result` | Foreground tool result. |
| `text/x-error` | Expected tool/runtime error. |
| `text/x-interrupted` | Run was cancelled. |
| `text/x-signal-status-changed` | Agent status changed. |
| `application/x-done` | Final token/cost metadata for the run. |

The returned `result` is still the final assistant message.

## CLI output formats

```bash
printf 'Say hi' | sagent --provider Google --model gemini-3.1-pro-preview
printf 'Say hi' | sagent --provider Google --model gemini-3.1-pro-preview --output-format json
printf 'Say hi' | sagent --provider Google --model gemini-3.1-pro-preview --output-format stream-json
```

`text` prints final message content.

`json` prints:

```json
{"content":"final answer"}
```

`stream-json` prints newline-delimited JSON event records. Each record includes at least `descriptor` and `content`. The final record has `descriptor: "result"`.

```json
{"descriptor":"text/plain","content":"Hello"}
{"descriptor":"application/x-done","content":"{...}"}
{"descriptor":"result","content":"Hello"}
```

## CLI input formats

`--input-format text` reads all stdin as one prompt.

`--input-format stream-json` reads newline-delimited JSON objects and collects each object's `prompt` field. Blank lines and objects without `prompt` are ignored. Prompts are joined with blank lines into one agent request.

```bash
cat <<'EOF' | sagent --provider Google --model gemini-3.1-pro-preview --input-format stream-json --output-format json
{"prompt":"Summarize file A."}
{"prompt":"Then compare it with file B."}
EOF
```

## Continuous agents

`Agent.run_forever(events=queue)` is for long-lived surfaces such as REPLs and chat adapters. It drains `agent.inbox`, joins queued strings into a prompt, calls `run()`, and waits for more inbox items. It exits only on Sagent's quit sentinel.

`AgentSend`, background task completion, and host UIs all communicate through the same inbox path.

## Streaming tools

Batch tools return one final result. Streaming tools yield intermediate `Message` objects and then a final result:

```python
class ProgressTool:
    streaming = True

    async def run(self, msg):
        yield TextMessage("starting", "text/plain")
        yield TextMessage("done", "text/plain")
```

The dispatch layer forwards intermediate yields to the event queue. The last yielded message becomes the actual tool result that is appended to conversation history.

## Caveats

`stream-json` is a machine-readable event format, not a stable RPC protocol. Descriptors are the compatibility surface; exact content strings may change as tool summaries and provider messages evolve.

Headless `stream-json` input is prompt-oriented: it does not replay a full prior conversation. Use sessions when you need conversation continuity.
