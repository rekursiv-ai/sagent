# Tools

Tools are ordinary Python objects. They let an agent read files, run commands, search the web, inspect papers, send service requests, coordinate with other agents, and call application-specific code.

Pass only the tools an agent needs. File, shell, and network tools act with the current process permissions.

## Tool protocol

A tool exposes:

- `name`: human-readable tool name.
- `tool_id`: MIME-style identifier such as `application/x-tool-bash`.
- `description`: model-facing description.
- `directive_schema`: JSON Schema for input directives.
- `supports_microcompaction`: whether old results may be shortened.
- `summary(msg)`: short label for UI/status output.
- `prompt()`: optional per-request system prompt section.
- `run(msg)`: async execution.

Batch tools return one `Message`. Streaming tools set `streaming = True` and make `run(msg)` an async generator; yielded messages before the last are intermediate events, and the last yield is the final tool result.

Expected operational errors should return `TextMessage(..., "text/x-error")`. Unexpected bugs should raise.

## Function tools

Use the decorator path for deterministic functions.

```python
from sagent.tools import tool


@tool
def add(a: int, b: int) -> int:
    return a + b
```

Sagent infers the JSON Schema from type hints. Decorated functions get `tool_id = "application/x-tool-<name>"` and default to `supports_microcompaction = False`.

## Class-based tools

Use a class when the tool needs state, custom schema, a prompt section, custom summaries, or direct `Message` control.

```python
from sagent.custom_types import Message, TextMessage
from sagent.lib.custom_json import JSON, json_freeze
from sagent.lib.message import get_directive


class EchoTool:
    name = "Echo"
    tool_id = "application/x-tool-echo"
    description = "Echo text back to the caller."
    supports_microcompaction = True
    directive_schema: JSON = json_freeze(
        {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        }
    )

    def summary(self, msg: Message) -> str:
        return "Echo"

    def prompt(self) -> str | None:
        return None

    async def run(self, msg: Message) -> Message:
        directive = get_directive(msg)
        return TextMessage(str(directive["text"]), "text/plain")
```

## Exported built-ins

`sagent.tools.__all__` is the authoritative public export list.

| Tool | Purpose | Key inputs |
| --- | --- | --- |
| `Read` | Read text, images, PDFs, notebooks. | `file_path`; optional `offset`, `limit`, `last_lines`, `pages`. |
| `Write` | Create or overwrite files. | `file_path`, `content`. Existing files must have been read first. |
| `Edit` | Exact string replacement. | `file_path`, `old_string`, `new_string`; optional `replace_all`. |
| `Glob` | Find files by filename pattern. | `pattern`; optional `path`, `max_results`. |
| `Grep` | Search file contents with ripgrep. | `pattern`; filters and output controls. |
| `List` | List directory entries. | `path`; optional `show_hidden`, `long`, `max_results`. |
| `Bash` | Run shell commands. | `command`; optional `timeout`, `description`. |
| `WebSearch` | Query web search backends. | `query`; optional domain filters and `backend`. |
| `WebFetch` | Fetch and extract readable URL content. | `url`. |
| `PaperSearch` | Search scholarly papers. | `query`; optional source, year, OA, limit filters. |
| `PaperDetails` | Paper metadata, references, citations. | `id`; optional `operation`. |
| `PaperAuthor` | Author search and author papers. | `query` or `id`; optional `operation="papers"`. |
| `PaperFetch` | Download paper PDFs. | `id`. |
| `AgentSelf` | Let an agent inspect/mutate itself. | Flat patch object; all fields optional. |
| `AgentSpawn` | Run child agents. | `prompt`; optional model/tools/depth/session controls. |
| `AgentSend` | Send to a live named agent inbox. | `to`, `content`; optional `delay`. |
| `BackgroundTask` | Manage background tool jobs. | `operation=list|cancel|foreground`; optional `id`. |
| `Slack` | Slack Web API operations. | `operation`. |
| `Linear` | Linear issue/comment operations. | `operation`. |
| `Wiki` | Query an initialized llm-wiki. | `operation`. |
| `Skill` | Load a user-authored skill body. | `skill`; optional `args`. |
| `PlayAudio` | Play a short local WAV notification. | `path`. |

## File tools

`Read` supports line windows for text files, page ranges for PDFs, visual rendering for common image formats, and notebook extraction for `.ipynb`.

`Write` is intentionally stricter than normal file writes: if the target file already exists, the agent must have read it in the current tool state. This prevents accidental clobbering of unseen user work.

`Edit` uses exact string matching as its safety gate. A non-unique `old_string` fails unless `replace_all` is true. Edits and writes are serialized per path so concurrent agents do not write the same file at once.

`Glob`, `Grep`, and `List` are preferred over shelling out for discovery, search, and directory listing.

## Bash

`Bash` runs `/bin/bash -c <command>` with the agent process permissions. The default timeout is 120 seconds and the maximum is 600 seconds.

When `Bash` is constructed with peer tools, it can warn the model when a dedicated tool should be used instead of a shell command.

## Web tools

`WebSearch` uses DuckDuckGo by default.

Use `allowed_domains` and `blocked_domains` to scope results.

`WebFetch` fetches HTTP(S) URLs, extracts main content, caches results for 15 minutes, rewrites known Reddit URLs to fetchable forms, and rejects URLs resolving to loopback, private, link-local, multicast, reserved, or unspecified addresses. Redirect targets are checked too.

## Paper tools

Paper tools use Semantic Scholar, OpenAlex, arXiv, and open-access PDF metadata.

`PaperSearch`:

- `source="s2"` queries Semantic Scholar. This is the default.
- `source="openalex"` queries OpenAlex.
- `source="fused"` merges both, deduplicating by DOI.
- Optional filters: `year_from`, `year_to`, `open_access_only`, `limit`, `abstract_chars`.

`PaperDetails` accepts DOI or arXiv IDs. Omit `operation` for metadata, use `operation="references"` for cited papers, and `operation="citations"` for citing papers. Citation mode also supports `influential_only` and `year_from`.

`PaperAuthor` searches authors by `query`, returns metadata by `id`, or lists publications with `id` plus `operation="papers"`.

`PaperFetch` downloads PDFs from arXiv first, then open-access URLs. PDF bytes are cached under `~/.sagent/papers`.

## Agent coordination tools

`AgentSelf` accepts a single patch object. All fields are optional; omitted fields are unchanged:

- `status`: update visible status text.
- `diagnostics`: set `true` to include token/cost/model/provider state in the result.
- `context`: `"compact"`, `"recompact"`, or `"clear"`.
- `context_prompt`: guidance for compact/recompact, or reason for clear.
- `model_id`, `provider`, `auth`, `account`: switch model/provider.
- `max_request_tokens`, `max_response_tokens`: adjust token limits.
- `model_options`: provider-specific settings (`thinking`, `effort`, `cache_ttl`).
- `catalog`: `"providers"` or `"models"` for read-only discovery.

`AgentSpawn` creates child agents. Children can inherit the parent model/tools or override provider, auth, model ID, account, tools, max tool-call rounds, and max depth. With `persistent=true`, a child stays alive and can receive messages through `AgentSend`.

Persistent subagents have a durable lifecycle. The parent session
records each child's `kind=persistent_subagent` entry with a stable
`run_id` and `session_dir`. Terminal states (`completed`, `failed`,
`cancelled`) are written as new lifecycle records so a future resume
knows which children should restart and which stay archived. Explicit
cancellation via `BackgroundTask cancel persistent:<label>` or
`/kill <label>` writes a `cancelled` record before shutting the child
down; the implicit `Ctrl-D` / `/quit` path leaves running children
resumeable. See [CLI](cli.md) for the slash target grammar.

`AgentSend` sends text to a live named agent inbox, optionally after a delay.

## Background tasks

If `BackgroundTask` is available to an agent, Sagent injects `background` and `delay` fields into other tool schemas.

```json
{"command": "long-running command", "background": true}
{"command": "remind me", "delay": 60}
```

A backgrounded tool immediately returns a placeholder result. When the tool completes, its real result is delivered to the agent inbox. Use:

- `BackgroundTask(operation="list")`
- `BackgroundTask(operation="cancel", id="...")`
- `BackgroundTask(operation="foreground", id="...")`

`delay` implies backgrounding.

## Service tools

`Slack` uses a bot token and supports:

- `send`
- `list_channels`
- `list_messages`
- `read_thread`
- `list_users`
- `create_channel`

`Linear` uses `LINEAR_API_KEY` and supports:

- `list_issues`
- `get_issue`
- `create_issue`
- `update_issue`
- `add_comment`

These tools mutate external services when asked. Give them only to agents that need them.

## Wiki, skills, and audio

`Wiki` operates on an initialized llm-wiki with `SCHEMA.md`. Operations are `locate`, `list`, `read_page`, `read_index`, and `lint`.

`Skill` loads a user-authored skill by name and returns its full body so the agent can follow it. Skill invocations are tracked across compaction.

`PlayAudio` plays `.wav` files using host audio tools when available and silently no-ops on headless systems.

## Microcompaction

`supports_microcompaction` tells the compactor whether old tool results can be shortened after they are safely outside the prompt-cache hot path. Use `True` for large, reproducible, or low-value historical output. Use `False` when exact prior output must remain in context.

Sagent preserves recent clearable tool results, clears only cold results, and invalidates `Read` cache entries when old `Read` outputs are cleared.

## Tool-result storage

Sagent bounds large tool outputs before sending the next model request.

- Plain tool output is truncated at 400,000 characters by the generic tool wrapper.
- Non-exempt oversized tool results above 50,000 characters are persisted to disk.
- Persisted previews include the path and the first 2,000 characters.
- Aggregate tool output in one message is bounded to 200,000 characters.
- Session runs store persisted results under `session_dir/tool-results`; no-session runs fall back to `/tmp/sagent_results`.
- Persisted files are written with owner-only permissions.

`Read` is exempt from persistence because it already has line/page limits.
