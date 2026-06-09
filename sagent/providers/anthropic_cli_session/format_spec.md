# Claude Code session JSONL — format spec

**Pinned against:** `claude --version` → `2.1.168` (as of 2026-06-08).
Newer minor versions may add fields; the materializer must remain
forward-compatible (extra fields ignored, missing-but-required fields
treated as a tripwire failure).

**On-disk location.** Each session lives at:

```
~/.claude/projects/<encoded-cwd>/<session-uuid>.jsonl
```

`<encoded-cwd>` is the cwd at the time the CLI was spawned, with all
non-`[A-Za-z0-9-]` characters replaced by `-`. E.g. `/home/jp/blackjax-devs`
encodes to `-home-jp-blackjax-devs`. Leading `/` becomes a leading `-`.

**File format.** One JSON object per line, NDJSON. No trailing newline
required but accepted.

---

## Entry types

The CLI emits 10+ entry types in practice; the materializer handles
the four that matter for `--resume` replay and emits nothing else.
Everything else is sidecar (UI state, history snapshots, queue
management) that `--resume` does not need.

### `user` — operator or tool-result message

Two flavors share the type:

1. **operator text** (`message.content` is a plain string)
2. **tool results** (`message.content` is a list of `tool_result` blocks)

Required top-level fields:

| field | type | notes |
|---|---|---|
| `type` | `"user"` | discriminator |
| `parentUuid` | str \| null | UUID of the previous chain entry; null at session start |
| `isSidechain` | bool | always `false` for sagent-materialized entries |
| `uuid` | str | this entry's UUID |
| `timestamp` | str | ISO-8601 UTC with `Z` suffix, ms precision |
| `sessionId` | str | session UUID (matches the filename stem) |
| `cwd` | str | absolute path of the spawn cwd |
| `gitBranch` | str | current branch, or `"HEAD"` if detached |
| `userType` | `"external"` | always `"external"` for live user input |
| `entrypoint` | `"cli"` | always `"cli"` |
| `version` | str | claude CLI version when entry was written |
| `message.role` | `"user"` | |
| `message.content` | str \| list | text body OR list of `tool_result` blocks |

`promptId` is added when the entry came from a user prompt; the
materializer omits it (it is decorative).

### `assistant` — model response

Required top-level fields are the same as `user` PLUS:

| field | type | notes |
|---|---|---|
| `requestId` | str | API request id (e.g. `req_011C...`); materializer mints `mat_<n>` |
| `message.model` | str | model id served the response |
| `message.id` | str | API message id (e.g. `msg_01...`); materializer mints `msg_mat_<n>` |
| `message.type` | `"message"` | |
| `message.role` | `"assistant"` | |
| `message.content` | list | content blocks (see below) |
| `message.stop_reason` | str | typically `"end_turn"` or `"tool_use"` |
| `message.stop_sequence` | null | |
| `message.usage` | dict | token usage; materializer fills with zeros |

### `summary` — compaction summary

Not currently emitted by the materializer (sagent's `ContextSplice`
replaces the masked range inline). Reserved for future fidelity.

### `system` — system meta entry

Used by claude for compaction boundaries (`subtype:
"compact_boundary"`) and other state transitions. Not currently
emitted by the materializer; if needed, sagent's compaction is
already represented by inline `ContextSplice` payloads.

---

## Content blocks

Within `message.content` (list-typed entries):

### `text`

```json
{"type": "text", "text": "..."}
```

### `thinking`

```json
{
  "type": "thinking",
  "thinking": "<opaque body, may be empty>",
  "signature": "<base64-ish; opaque to operator>"
}
```

If `signature` is set and `thinking` is empty, the block is an
"orphan" and the assistant API rejects it. The materializer
suppresses orphans (mirrors `providers/anthropic.py:_is_orphan_thinking`).

### `tool_use`

```json
{
  "type": "tool_use",
  "id": "toolu_01...",
  "name": "Bash",
  "input": {...}
}
```

`caller` field is added by the CLI to indicate how the call was
dispatched (e.g. `{"type": "direct"}`). The materializer omits this;
the CLI tolerates its absence on replay.

### `tool_result`

```json
{
  "type": "tool_result",
  "tool_use_id": "toolu_01...",
  "content": [{"type": "text", "text": "..."}],
  "is_error": false   // optional, default false
}
```

Live CLI sometimes stores `content` as a plain string; the
materializer always emits the list-of-blocks form because Anthropic's
API accepts both and the round-trip parser handles either.

---

## parentUuid chain

For materializer outputs, the chain is strictly linear: each entry's
`parentUuid` is the previous entry's `uuid`. No branching, no
side-chains.

UUIDs are minted **deterministically** so the same input tape always
produces a byte-identical JSONL (modulo the volatile fields below).
The materializer uses UUIDv5 of `(session_id, tape_index)` against a
fixed namespace; this lets golden tests pin the entire output.

---

## Volatile fields (ignored when comparing)

When diffing a materializer-produced JSONL against one claude wrote,
ignore:

- `timestamp` (wall-clock; even ms-precise replays drift)
- `requestId` (provider-assigned; materializer mints synthetic ids)
- `uuid` / `parentUuid` (materializer uses UUIDv5; claude uses v4)
- `message.id` (provider-assigned)
- `message.usage` (depends on tokenization)
- `slug`, `forkedFrom`, `customTitle`, `aiTitle`, `agentName`,
  `lastPrompt`, `mode`, `permissionMode`, `agentSetting`,
  `last-prompt`, `queue-operation`, `attachment`,
  `file-history-snapshot` (UI/state sidecars; not chain-bearing)

The structural diff used by the tripwire and round-trip tests
compares the resulting Anthropic-style message list (role + content
blocks), not byte-for-byte JSONL.

---

## Sidecar types the materializer drops

These show up in real CLI files but contribute nothing to `--resume`:

| type | role |
|---|---|
| `attachment` | UI hint for re-fed compaction summaries |
| `custom-title` | UI title |
| `agent-name` | UI label |
| `mode` | normal/plan mode toggle |
| `permission-mode` | acceptEdits/default |
| `last-prompt` | UI history pointer |
| `file-history-snapshot` | file backup metadata |
| `queue-operation` | enqueue/dequeue marker |
| `ai-title` | UI title |
| `agent-setting` | per-role label |

If the round-trip parser encounters one of these in a claude-written
JSONL, it skips it without error.
