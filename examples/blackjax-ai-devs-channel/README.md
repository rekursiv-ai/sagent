# blackjax-chat (v2 — session-resume CLI; v2.1 — sagent-owned JSONL)

A multi-agent chat channel for BlackJAX, built as a plugin on top of
[sagent](https://github.com/rekursiv-ai/sagent). Five specialised
agents (`tl`, `swe`, `junior-swe`, `statistician`, `tech-writer`) run
as long-running asyncio tasks in one Python process, talking to each
other and a human operator through a typed inbox with mid-turn
preemption, on-the-fly status, and a web UI.

This is the **v2** layout. The frozen reference v1 lives at
`../sagent_anthropic_cli_v1/` and is documented separately;
the pivot v1 → v2 is summarised below. Install **v2** unless you
specifically need to inspect v1's `restart_notice` observer.

**v2.1** (default-on as of 2026-06-09) layers a *materializer* on top:
sagent's tape becomes the canonical history and the session JSONL is
just resume transport for `claude --print --resume`, with
sagent-driven compaction and resume-from-memory on restart. See
[§ v2.1: sagent owns the session JSONL](#v21-sagent-owns-the-session-jsonl-materializer)
below. Opt out per boot with `SAGENT_CLI_OWN_SESSION=0`.

For the full history of how we got here (the failed `channel/` tmux
runtime, the structural limits we hit, the external-MCP probe that
unblocked us), see
[`claude-config/project/worklog/threads/chat-to-sagent-migration.md`](../../../claude-config/project/worklog/threads/chat-to-sagent-migration.md).
This README sticks to what's shipping and how to run it.

---

## Pivot from v1: session-resume instead of history re-feed

Both `plugin/sagent_anthropic_cli_v1/` (frozen) and this directory
(v2, the recommended install) implement the same chat channel, but
v2 took a structural pivot mid-day 2026-06-02 → 06-03 morning. The
short version:

- **v1**'s CLI provider re-fed the full `agent.history` via stdin
  on every respawn EXCEPT `AssistantMessage` entries, which it
  stripped at `providers/anthropic_cli.py:537`. That meant the
  respawned subprocess saw peer replies but no record of its own
  prior delegations — which on `aborted_streaming` recovery made
  opus re-issue work it had already done. The v1 plugin worked
  around this with an in-tree `restart_notice` observer that
  recovered each prior `sagent_send` from a per-agent
  `outbound_log` and `runtime.append_splice`-ed synthetic
  reconstructions back into history.

- **v2** drops the re-feed + observer entirely. Each agent gets a
  stable `UUIDv5` session id; every `claude --print` subprocess
  spawns with `--session-id <uuid>` (first turn) or
  `--resume <uuid>` (subsequent turns). `claude` itself owns the
  on-disk transcript — assistant turns, `tool_use` blocks,
  `tool_result` blocks, all preserved — at
  `~/.claude/projects/-<encoded-cwd>/<uuid>.jsonl`. Sagent stops
  feeding history at all; only the new inbound is sent per turn.
  The `restart_notice` observer is deleted because the problem
  it papered over no longer exists.

Why pivot at all? Two reasons:

1. **The v1 observer was theatre on real-world history.** It
   walked `runtime.tape` looking for outbound `sagent_send` tool
   calls in `AssistantMessage.tool_calls`, but the CLI provider
   always returns `tool_calls=()` (`anthropic_cli.py:924`) — the
   MCP tool round-trip runs opaquely inside `claude --print`.
   The observer's bug-shape unit tests passed because they seeded
   synthetic `ToolCall` entries the provider never produces.
   v1 caught this on 2026-06-02 evening; the next iteration of
   the observer pulled outbounds from a separate `outbound_log`
   populated by `/api/post` instead, but it was still a
   reconstruction.

2. **Session-resume sidesteps the rest of the problem too.**
   On `aborted_streaming`, claude's session JSONL is already on
   disk with everything the respawned subprocess needs.
   `--resume <uuid>` picks up the conversation including
   tool_use/tool_result round-trips and thinking blocks; sagent
   doesn't have to reconstruct anything. Prompt cache hits stay
   warm across turns (~40k cache reads observed on resumed
   turns vs zero in v1's re-feed shape).

The v2 changes live on the `feat/cli-session-resume` branch:

- **Upstream sagent (`providers/anthropic_cli.py`):** opt-in
  `session_id` parameter on `AnthropicCLI.model(...)`. When set,
  argv swaps `--no-session-persistence` for `--session-id` /
  `--resume`, HotSpare is bypassed (spawn-on-demand per turn —
  pre-warming a spare with `--resume` branches the conversation
  tree), and history re-feed is disabled.
- **Plugin (`roles/common.py`):** every agent built with a
  stable `UUIDv5(namespace, f"blackjax-chat:{role_name}")`
  passed to `provider.model(session_id=…)`. Survives server
  restarts because the namespace + role label are deterministic.
- **Plugin runtime:** `runtime/restart_notice.py` deleted along
  with its 11 unit tests, the `outbound_log` plumbing, and the
  debug seed/inject endpoints that only existed for observer
  validation.

What still differs from v1 (other than the deletions):

- v2 inherits the operator's real `HOME` in session-persistent
  + single-account mode, so native tools (`Bash`-from-shell, `gh`,
  `git`, ssh) find `~/.config/`, `~/.gitconfig`, etc. (v1 had a
  hermetic per-spawn tmpdir but mounted those tools via sagent's
  HTTP bridge — the handler ran in the sagent server process
  with the operator's real HOME, so it was a non-issue. v2's
  tools run inside `claude --print`, so the env has to be
  right at the subprocess.)
- `--tools ""` is omitted from the argv in session-persistent
  mode (bisect 2026-06-03 found that flag becomes "allow NO
  tools, INCLUDING MCP ones" once `--session-id` is set, which
  silently broke structured tool dispatch). Stateless mode keeps
  the flag — no observed regression.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│ blackjax-chat serve.py (single Python process, asyncio)                  │
│                                                                          │
│ ┌─────────┐  ┌──────────┐  ┌─────────────┐  ┌──────────┐  ┌──────┐       │
│ │ tl      │  │ swe      │  │ junior-swe  │  │ statist. │  │ tech │       │
│ │ Agent   │  │ Agent    │  │ Agent       │  │ Agent    │  │ Agent│       │
│ └────┬────┘  └────┬─────┘  └─────┬───────┘  └────┬─────┘  └──┬───┘       │
│      │            │              │               │           │           │
│      ▼            ▼              ▼               ▼           ▼           │
│ ┌──────────────────────────────────────────────────────────────┐         │
│ │ per-agent claude --print --mcp-config <role>.mcp.json        │         │
│ └────────────────────────┬─────────────────────────────────────┘         │
│                          │ MCP stdio                                     │
│                          ▼                                               │
│ ┌──────────────────────────────────────────────────────────────┐         │
│ │ mcp_sagent/server.py (separate Python process per agent)     │         │
│ │   sagent_send / sagent_defer / sagent_self                   │         │
│ └────────────────────────┬─────────────────────────────────────┘         │
│                          │ HTTP POST to 127.0.0.1:8767                   │
│                          ▼                                               │
│ ┌──────────────────────────────────────────────────────────────┐         │
│ │ HTTP + web UI (Starlette+uvicorn)                            │         │
│ │   /api/{roles,agents,messages,trace,search,post,defer,restart}│        │
│ └──────────────────────────────────────────────────────────────┘         │
│                                                                          │
│  Runtime observers per agent:                                            │
│   • trace_writer    → sessions/<role>.trace.jsonl                        │
└──────────────────────────────────────────────────────────────────────────┘
```

Beyond `$SAGENT_DATA_DIR`, each agent's claude session JSONL (the
authoritative transcript under v2) lands at the operator's real
`~/.claude/projects/-<encoded-cwd>/<uuid>.jsonl` and survives
`serve.py` restarts — the next server boot `--resume`s the prior
conversation per agent. To start one agent fresh, delete its
session JSONL.

Three layers because each MCP server is its own Python process (spawned
by `claude --print` via `--mcp-config`) and can't reach the live
`agent_registry` in `serve.py`. HTTP loopback to `serve.py`'s
`/api/post` is the only synchronisation point all three layers share.
Cost is sub-millisecond per peer message, swamped by model-call latency.

**Data files** live under `$SAGENT_DATA_DIR` (set at launch), not the
plugin source tree. This lets the audit log co-locate with the legacy
`channel/main.jsonl` for `bin/merge_jsonl.py`:

```
$SAGENT_DATA_DIR/
├── main.jsonl                  ← audit log (channel/-compatible)
└── sessions/
    ├── <role>.trace.jsonl     ← per-agent runtime events
    ├── <role>.mcp.json        ← per-role MCP config
    └── mcp_calls.log          ← MCP server debug log
```

---

## What doesn't work out of the box

Three structural mismatches surfaced when we tried to drop our 5-agent
team onto stock sagent. Each one is the justification for one of the
patches in the next section. Full investigation in
[`worklog/threads/chat-to-sagent-migration.md`](../../../claude-config/project/worklog/threads/chat-to-sagent-migration.md);
the compressed version:

1. **Sagent's in-bridge tools (`AgentSend`, `Read`, `Bash` — bare
   names) don't get structurally dispatched by Sonnet/Opus.** Both
   models emit the tool calls as TEXT inside the assistant message
   (`<function_calls>` blocks, raw JSON, or prose) rather than as
   `tool_use` content blocks. Only Haiku used the structured channel
   reliably. This is independently reported in pydantic-ai#1904,
   jundot/omlx#159, and the Cursor forum — it's a `claude --print`
   streaming-mode issue, not specific to sagent. **The probe at
   `/tmp/sagent_probe/` (2026-06-01) found that external MCP tools
   surfaced via `--mcp-config` with the `mcp__<server>__<tool>` prefix
   ARE structurally dispatched by all three models.** That's why this
   plugin mounts `sagent_send` / `sagent_defer` / `sagent_self` via a
   separate MCP stdio server (`mcp_sagent/server.py`) instead of
   sagent's in-bridge tool registry, and why the three-process
   architecture exists at all.

2. **The CLI subprocess runs the entire MCP tool loop opaquely.** From
   sagent's POV, one `claude --print` turn is one
   `ModelCallStarted` → one `ModelResponseComplete` — the runtime
   never sees intermediate `tool_use` / `tool_result` blocks. So
   `_stop_all_tools` has no in-flight cohort to act on, and mid-turn
   `AgentSendMessage` arrivals just queue into `_mid_stream_queue`
   while the in-flight CLI keeps running. The headline "mid-turn
   preempt" sagent advertises doesn't work at subprocess
   granularity without a SIGINT path. That's the
   `preempt_in_flight=True` patch (override #2 below).

3. **Sagent's inbox-coalesce is single-user-shaped.** Upstream merges
   consecutive same-source `AgentSendMessage`s into one history entry.
   This is right for human typing (three lines typed in a row = one
   prompt) but wrong for distinct peer events: when TL sends a
   delegation, then a correction, then a hard `STOP`, the recipient
   must see those as three separate inbounds — not one 9 KB blob
   with `STOP` buried at the bottom. Override #1 inverts this.

Beyond those three: the CLI provider strips `AssistantMessage` entries
before re-feeding history to a respawned subprocess
(`providers/anthropic_cli.py:537`) — so on `aborted_streaming`
recovery, the model has no record of its own prior delegations.

In **v1** this was worked around in-plugin via the splice-based
`restart_notice` observer. In **v2** it's structurally bypassed:
each agent runs with `--session-id` / `--resume`, claude itself owns
the on-disk transcript including assistant turns + tool_use blocks,
sagent doesn't re-feed history at all, and respawn `--resume`s the
session that was already on disk. The stripping line still exists
in the provider for stateless-mode callers; session-persistent mode
just never reaches it. The override #3 below is now "session id
wiring" rather than the observer.

---

## Sagent behaviour overrides

Three places we deviate from upstream sagent. All justified by the
mismatches above.

### 1. `coalesce_inbox=False`  (upstream default: `True`)

Upstream merges consecutive same-source `AgentSendMessage`s into one
history entry. Correct for human typing, wrong for distinct peer events:
a delegation + correction + `STOP` from TL must arrive at SWE as three
separate inbounds, not one 9 KB blob with `STOP` buried at the bottom.

With the override, the runtime injects a synthetic
`AssistantMessage("(runtime: discrete-inbound boundary)")` between
consecutive peer messages instead — satisfies API alternation, keeps each
peer message distinct.

### 2. `preempt_in_flight=True`  (upstream default: `False`)

Sends SIGINT to the in-flight `claude --print` subprocess via
`model.cancel_in_flight()` before buffering. Required because the CLI
runs its MCP tool loop opaquely — sagent's runtime can't see in-flight
tool dispatches, so `_stop_all_tools` has nothing to act on. Without
this, mid-turn corrections wait for the current turn to drain.
Implementation lives on `feat/cli-preempt-via-sigint` in this fork.

### 3. `session_id=<uuid>` on `provider.model(...)`  (upstream, opt-in)

Each agent gets a stable `UUIDv5(namespace,
f"blackjax-chat:{role_name}")` passed into the provider at
construction time. The CLI provider then:

- Spawns `claude --print --session-id <uuid>` on the first turn
  and `--resume <uuid>` on every turn after that, in place of the
  upstream default `--no-session-persistence`.
- Sends only the newest user-like inbound to stdin per turn —
  claude has the rest in its session JSONL on disk.
- Bypasses HotSpare (each `stream()` call spawns its own
  subprocess; pre-warming a spare with `--resume <same-uuid>`
  branches the conversation tree).
- Inherits the operator's real HOME so native tools (Bash, gh,
  git, ssh) find `~/.config/`, `~/.gitconfig`, etc. (Stateless
  mode kept a hermetic per-spawn tmpdir for credential isolation
  — fine because its tools ran via the bridge in the sagent
  server process, with the operator's real HOME. Session-
  persistent mode runs tools inside `claude --print`, so the
  env has to be right at the subprocess.)
- Omits `--tools ""` from the argv: bisect probe 2026-06-03
  found that flag becomes "allow NO tools, INCLUDING MCP ones"
  once `--session-id` is set, silently breaking structured
  dispatch. Stateless mode keeps the flag.

With this in place:

- `aborted_streaming` recovery is structurally clean: the
  respawned `claude --print` `--resume`s the on-disk session,
  which already contains every assistant turn + tool_use block
  the prior subprocess emitted. No reconstruction needed; no
  observer needed.
- Prompt-cache hits stay warm across turns (~40k cache reads on
  resumed turns vs ~0 in v1's re-feed shape).
- `serve.py` restarts pick up the prior conversation per agent
  — the session JSONLs survive at
  `~/.claude/projects/-<encoded-cwd>/<uuid>.jsonl`.
- The plugin's `runtime/restart_notice.py` is deleted along
  with its 11 unit tests, the per-agent `outbound_log`, and the
  debug seed/inject endpoints.

For per-account use (`provider.account is not None`), the
provider still mints a per-construction-time hermetic tmpdir
with the renamed credentials file — claude's creds path is
hardcoded to `$HOME/.claude/.credentials.json` and there's no
env redirect.

---

## v2.1: sagent owns the session JSONL (materializer)

v2 lets **claude** own the on-disk session JSONL; sagent reads its
own tape for state but defers the transcript to claude. v2.1 inverts
that: **sagent's tape is canonical and the JSONL is just resume
transport** — the same mental model as direct-API mode, except the
"messages array" is materialized to a file so `claude --print
--resume` can read it. Default-on; set `SAGENT_CLI_OWN_SESSION=0`
(or `false`/`no`) to opt back to v2 CLI-owned mode.

Lives in `sagent/providers/anthropic_cli_session/` (upstream-eligible
core): `materializer.py` (tape → CLI-shaped NDJSON, deterministic
UUIDv5 chain, atomic write), `parser.py` (the inverse: JSONL → sagent
message list), `tripwire.py` (structural drift check + live canary),
`format_spec.md` (wire format pinned against `claude --version`).

**Before each `--resume` spawn**, the provider rewrites the JSONL from
`request.messages[:_last_sent_index]` (the resolved tape view), so
whatever claude appended last turn is superseded by sagent's canonical
record. New entries still go via stdin (as in v2). This makes sagent
the single source of truth and lets sagent's own `SummaryCompactor`
drive compaction (claude's auto-compact is disabled in this mode to
avoid the materializer clobbering claude's `compact_boundary`).

### Startup tripwire (canary)

On boot, `serve.py` runs a 1-turn `claude --print` canary, schema-checks
the JSONL claude wrote, and structurally round-trips it through the
materializer. On any drift it sets `SAGENT_CLI_OWN_SESSION=0` and falls
back to v2 for that boot — so a CLI-format change can never silently
corrupt sessions. Boot log on success:

```
materializer tripwire: PASS — sagent will own the session JSONL for this boot
```

### Resume-from-memory (tape rehydration)

sagent does **not** persist its own tape across a `serve.py` restart
(no `session_dir` wired). Naively, in materialize mode that would be
fatal: on the second post-restart turn the materializer rewrites each
JSONL from the short fresh tape and **clobbers the full history**. The
fix is to reconstruct the tape from the JSONL itself — the persisted
memory — at boot: `_rehydrate_agents_from_jsonl` runs after
`_build_all_agents` and before warmup, and for each agent:

```
parse_jsonl_to_messages(jsonl)
  → repair_dangling_tool_calls(...)         # fix any truncated mid-tool pairing
  → [ReferrableTapeEvent(TapeRef(sid, i), m) for i, m in ...]
  → runtime.replay_tape(records)            # seed the tape
  → model.seed_session(len(messages))       # on-disk prefix is synced
                                            #   → next spawn uses --resume
```

A restart then resumes the conversation exactly where it left off —
verified live by TL recalling its in-flight PR's commit SHAs after a
restart. No-op in v2 (opt-out) mode, where `claude --resume` already
owns the history; per-agent best-effort (a parse failure logs a warning
and that agent starts fresh).

### Things that bit us (and the fixes)

- **Compaction over-trigger.** `claude --print` reports `usage` tokens
  that are CUMULATIVE across its internal tool loop (one sagent turn =
  dozens of internal rounds), so `_last_input_tokens` (= input +
  cache_creation + cache_read) balloons to millions — dominated by
  cumulative `cache_read`, which is the prompt cache being *read*, not
  missed (the cache is healthy: ~0 misses observed). The compaction gate
  read that as "context is 5.6M tokens" and fired spuriously. Fix:
  the provider normalizes usage at its boundary — each internal round's
  `message_start` carries that round's request usage, and the LAST one
  IS the live context footprint, so `response.tokens`'s input side
  reports that (direct-API semantics) while `output_tokens` stays
  cumulative and billing rides `costUSD` untouched. The compaction gate
  then needs no special-casing at all (it even gains precision: exact
  server-side counts instead of a client-side estimate).
- **`signature_delta` dropped.** The stream parser ignored Anthropic's
  `signature_delta`, so `AssistantMessage.thinking_blocks` were unsigned.
  Inert in v2, fatal here (the materializer wrote unsigned thinking and
  the API rejected on `--resume` with `400 thinking.signature: Field
  required`). Fixed by capturing it alongside `thinking_delta`.
- **Two `role alternation` wedges.** (a) `SummaryCompactor` built a
  splice whose payload put a summary `UserMessage` adjacent to a kept
  `AgentSendMessage` — fixed by concatenating cross-type/cross-source
  adjacent user-side entries. (b) Claude writes ONE assistant turn as
  MULTIPLE consecutive `assistant` JSONL entries (one per content block);
  the parser made one `AssistantMessage` each, producing consecutive
  assistant-role messages that the tape rejects — fixed by coalescing
  consecutive assistant runs in `parse_jsonl_to_messages`.

### Materialize-mode caveats

- **Disk encoding depends on cwd.** The JSONL path is
  `~/.claude/projects/-<encoded-cwd>/<uuid>.jsonl` where `<encoded-cwd>`
  is the spawn cwd. Launch `serve.py` from the monorepo root (`cd
  /home/jp/blackjax-devs`) so the encoded dir matches across boots;
  rehydration computes the path from `Path.cwd()`.
- The provider's `materialize_session` kwarg still defaults to `False`
  (upstream-eligible signature). The plugin's `build_agent` is the one
  place that flips the default on for this deployment.

---

## Running it

Production form used during 2026-06-02 live testing:

```bash
tmux new-session -d -s sagent-chat -n serve \
  -c /home/jp/blackjax-devs \
  'SAGENT_DATA_DIR=/home/jp/blackjax-devs/claude-config/experimental/sagent \
   exec ~/rekursiv/sagent/.venv/bin/python \
   /home/jp/rekursiv/sagent/examples/blackjax-ai-devs-channel/bin/serve.py --port 8767'
```

Three things this form gets right:

1. `-c /home/jp/blackjax-devs` sets tmux pane cwd → bash → python →
   `Path.cwd()` at agent construction → each Bash tool's `start_cwd`
   is the monorepo root, not the plugin source dir. **In v2.1
   (materialize) mode this is load-bearing**: the session JSONL path
   is `~/.claude/projects/-<encoded-cwd>/<uuid>.jsonl`, and tape
   rehydration on restart computes that path from `Path.cwd()`. Launch
   from a different cwd and a restart silently fails to resume (it
   looks in the wrong encoded dir) — observed 2026-06-09 when a
   relaunch inherited the sagent-repo cwd from a leading `cd`.
2. `SAGENT_DATA_DIR=…experimental/sagent` lands audit log + traces
   beside the legacy `channel/main.jsonl` for end-of-day merge.
3. Absolute path to `serve.py` — relative paths wouldn't resolve with
   `-c` pointing at `~/blackjax-devs`.

Casual local run — **still `cd` to the monorepo root, not the plugin
dir**, so the encoded-cwd JSONL path stays stable across restarts:

```bash
cd /home/jp/blackjax-devs
SAGENT_DATA_DIR=/home/jp/blackjax-devs/claude-config/experimental/sagent \
  ~/rekursiv/sagent/.venv/bin/python \
  ~/rekursiv/sagent/examples/blackjax-ai-devs-channel/bin/serve.py --port 8767
```

Web UI at `http://127.0.0.1:8767/` — open via SSH tunnel:

```bash
ssh -L 8767:127.0.0.1:8767 <host>
```

`SERVE_HOST` is forced to `127.0.0.1` (loopback-only, no auth).

---

## Comparison: `channel/` vs sagent+CLI vs sagent+API

Today's plugin is the middle column. Column 1 is what we migrated away
from. Column 3 is the next plausible step (direct Anthropic SDK
instead of `claude --print` subprocesses), speculative and not built.

Marker key: ✅ works / materially better, ⚠️ works with caveats,
❌ broken or materially worse, 🔮 speculation.

The middle column is now **v2** (sagent + CLI in `--session-id` /
`--resume` mode). For the v1 numbers (stripping work-arounds via
`restart_notice`, no cache hits, etc.) see
`plugin/sagent_anthropic_cli_v1/README.md`.

| Dimension | `channel/` (tmux) | sagent+CLI v2 (today) | sagent+API (speculative) |
|---|---|---|---|
| **Process model** | ❌ One Python worker per agent, per tmux pane, per systemd cgroup. | ✅ Single process, asyncio task per agent. | 🔮 Same, but no CLI subprocesses at all. |
| **Cross-agent latency** | ❌ 5–15 s (poll cycle + cold CLI start). | ✅ Sub-second (in-process inbox + per-turn fresh subprocess). | 🔮 Sub-second, no subprocess to wait on. |
| **Per-turn token overhead** | ⚠️ Sessions are persistent (`--session-id` / `--resume` on every turn), so each turn reads the full prior conversation as cached prefix — same shape as v2, just generated by a thinner Python worker. The historical "300-500 tok reminder" claim turned out to be wrong; verified 2026-06-03 by reading `experimental/channel/chat:362-364`. | ✅ ~Zero marginal (system prompt + tool description cached); per-turn read scales with session size. | 🔮 ~Zero, with full operator control over `cache_control` markers. |
| **Mid-turn cancel** | ❌ `kill -9`, no clean shutdown. | ✅ SIGINT to subprocess (override #2); gated on per-message `urgent` flag (added 2026-06-04, see "Coordination interrupt model" row). | 🔮 Native — close the SSE stream. |
| **History feed on respawn** | ✅ `--resume <session_id>`; `claude` owns the JSONL on disk. Same mechanism as v2. | ✅ `--resume <uuid>`; `claude` owns the JSONL on disk. Assistant turns + `tool_use` + `tool_result` blocks all preserved. | ✅ History is just `messages=`; assistant turns + `tool_use` + `tool_result` blocks all go in verbatim. |
| **Outbound visibility on respawn** | ✅ `--resume` reads claude's session JSONL which contains the prior assistant turns + tool blocks. | ✅ `tool_use` blocks live in claude's own session JSONL; nothing to reconstruct on respawn. | ✅ Free — `tool_use` blocks in `messages=` verbatim. |
| **`aborted_streaming` recovery** | ❌ Worker had no respawn logic — each `claude -p` was a one-shot `subprocess.run`; an `aborted_streaming` was just a non-zero exit that the worker logged and the operator had to investigate. | ✅ **In-place retry via `send_with_retry`** (added 2026-06-04, commit `ac287d1`): transient `aborted_streaming` / `ede_diagnostic` events raise `AnthropicCLIRetryableError` and stay inside the retry budget instead of escalating to `ModelResponseError`. Prompt cache stays warm (same request bytes). **The runtime never sees the error; no synthetic `[Error: …]` UserMessage added to history.** Operator visibility via `WARNING sagent.agent.retry: API error (attempt N/5)` lines in server log. | 🔮 Same outcome via `messages=` retry; structurally identical to v2's path now. |
| **Tool results in history** | ✅ Same as v2 — preserved on disk via `--resume`. | ✅ Preserved across resume; the respawned subprocess can introspect prior tool calls + results. | ✅ First-class user-message block; replayable. |
| **Prompt-cache hit rate** | ✅ Same as v2 — `--resume` produces byte-identical prefixes, hits the cache. | ✅ High — ~40k cache reads on resumed turns observed in 2026-06-03 testing. | 🔮 High and operator-controllable. |
| **Observability** | ⚠️ Manual log scraping. | ✅ `/api/agents`, `/api/trace/<role>`, `/debug` console, web UI. `ToolLabel` events surfaced from the stream-json content blocks with name + args summary. | 🔮 Inherits the plugin's `/api/*` and traces — they observe runtime events, not transport. |
| **Implementation complexity** | ❌ Per-pane workers, mention router, polling, cgroup wiring. | ⚠️ Single binary, two overrides + opt-in `session_id` provider flag + HTTP MCP bridge. (v1's `restart_notice` observer deleted.) | 🔮 Direct SDK calls; all overrides become unnecessary. |
| **Survives `serve.py` restart** | ✅ No `serve.py` — each tmux worker was its own systemd unit; killing one didn't lose its session. | ✅ Claude session JSONLs at `~/.claude/projects/-<encoded-cwd>/<uuid>.jsonl` survive; next server boot `--resume`s per agent. | 🔮 Same — `messages=` can be loaded from anywhere. |
| **Native tool config (`gh`, `git`, ssh)** | ✅ Per-pane shell inherits operator env. | ✅ Session-persistent + single-account inherits real HOME so `~/.config/gh/`, `~/.gitconfig`, ssh keys all visible to claude's native tools. | 🔮 N/A — no subprocess tools. |
| **Coordination interrupt model** | ✅ Operator-only ingress; peers don't interrupt each other (channel/ had no inter-agent runtime — Agent-tool was the alternative path). | ✅ **Per-message `urgent: bool` flag** (added 2026-06-04). Default False: peer + operator messages buffer in `_mid_stream_queue` and drain at the recipient's natural turn boundary. Opt-in True: SIGINT preempt fires. Authority: TL/coordinator and same-sender corrections for peer; operator opts in via UI toggle or Ctrl+Enter. **Empirically, 72% of TL's pre-fix preempts were routine peer FYIs** (status updates, acks, completion reports) that shouldn't have interrupted — eliminating those saves ~$0.30 of opus compute per event. | 🔮 Same primitive; runtime-level decision either way. |
| **Preempt-induced cost** | N/A | ❌→✅ **Pre-2026-06-04**: every peer-message arrival mid-turn killed the recipient's in-flight subprocess and produced a synthetic `[Error: …]` UserMessage in their session JSONL. Cascade pattern (TL accumulated 50+ such messages over yesterday's run). **Post-fix**: only `urgent=true` messages preempt; non-urgent messages buffer cleanly. Plus the in-place retry layer means even legitimate preempts no longer pollute history. | 🔮 N/A — no subprocess to kill; cancellation closes the SSE stream cleanly without billing-side artifacts. |
| **Per-session volume** | ⚠️ Typical channel/ session: ~36 user turns / 52 assistant turns / 32 tool calls / 143 KB JSONL (a real example from the historical disk). Shorter sessions because peer coordination was simpler (text messages, no MCP delegation, no chained tool calls per turn). | ⚠️ Typical v2 session by end-of-day: ~939 user turns / 1,286 assistant turns / 387 tool calls / 5.5 MB JSONL — ~26-38× larger. Multi-agent peer coordination + richer tool catalog generates far more turns per outcome. | 🔮 Same volume as v2 (the work is the same); cost expressed as `messages=` bytes instead of session JSONL size. |
| **Measured per-day cost** | Lower in absolute terms because the sessions are smaller (less work being done per agent). 2026-06-03 v2 cost across 5 agents, 6 active hours: $6.30 total ($1.63 TL/opus, $2.55 statistician/sonnet, $2.11 swe/sonnet, ~$0.01 each haiku). Channel/ comparison numbers not captured. | ⚠️ Headline cost is competitive while the cache stays warm; cold-start cache rebuilds (5-min TTL expires) are visible as ~$10-15 spikes on TL. | 🔮 Same fundamentals as v2; operator controls cache markers more granularly. |

**Summary.** v2 was a structural pivot from v1's "patch around CLI
stripping" approach, NOT from channel/'s session model -- channel/
already used `--resume` and benefited from the same on-disk session
JSONL + prompt-cache wins. What v2 adds over channel/ is the
single-process asyncio runtime + structured MCP peer messaging +
**urgent-gated interrupt** + observability. What v2 costs over
channel/ is that the richer tool catalog + MCP delegation makes
per-session volume grow 25-30×: more turns per outcome, larger
cached prefix read per turn. Headline cost is similar at steady
state, but cold-start cache rebuilds (5-min TTL) are visible as
~$10-15 spikes on opus-TL.

**The 2026-06-04 fix stack closed the dominant operational pain.**
The day's three layered fixes (per-entry advance `8dc81f1`,
in-place retry `ac287d1`, urgent-gated peer + operator preempt
`774eb6b`/`c1b8fa9`) converted the substrate from "works but
pollutes history + cascades on every interrupt" to "transparent
under sustained operator + peer load." Critically, **most of what
we'd called "Anthropic stream instability" turned out to be our
own SIGINT preempt firing on routine peer FYIs** (validated by
0-input-token aborts that the API never saw). See
`worklog/lessons/tool-harness/2026-06-04-sagent-chat-runtime-fixes-and-corrected-framing.md`
for the corrected diagnosis + validation evidence.

**The remaining gap to sagent+API** is now small (spawn cost per
turn, 5-min ephemeral cache TTL, MCP catalog reload per turn).
The API column makes sense for headless / cron-driven workloads
where spawn cost dominates, NOT as a must-replace upgrade for
interactive chat-runtime use.

---

## Validation status (2026-06-04 evening, post-fix-stack)

**Closed (v1 carried over to v2)**:

- ✅ Mention-router duplicate-emit cannot reproduce (router is gone).
- ✅ `hello, ready` warmup-template regression cannot reproduce
  (warmup uses `sagent_self`, silent to peers).
- ✅ `sagent_defer` round-trip works (`tl` scheduled +30 s, SWE
  later self-deferred +300 s and +180 s for CI polling — both fired
  on time, no `bash sleep` hangs).
- ✅ Structured channel works on opus/sonnet/haiku via external MCP.
- ✅ Mid-turn preempt works (SIGINT path fires on organic
  `ModelResponseError`).
- ✅ Sub-second cross-agent latency.

**Closed by v2 pivot**:

- ✅ AssistantMessage stripping → re-delegation on respawn.
  Structurally impossible now — claude's session JSONL contains
  the full prior assistant turn including tool_use blocks, and
  `--resume` picks it up byte-for-byte from disk.
- ✅ Native tools find operator config. v1 mounted Bash/Read/Glob
  via the bridge so the issue never surfaced; v2 runs them inside
  `claude --print`, so HOME passthrough was needed. `gh auth
  status` from inside a v2 agent now shows the operator's
  authenticated session.
- ✅ Session survival across `serve.py` restart. Each agent's
  JSONL persists at `~/.claude/projects/-home-jp-blackjax-devs/`;
  next boot `--resume`s it.
- ✅ Prompt cache hits across turns. ~40k cache reads observed
  on a typical resumed turn vs ~0 in v1's re-feed shape.
- ✅ Structured tool dispatch under `--session-id` (after the
  `--tools ""` removal — bisect 2026-06-03 found that flag
  silently disables MCP tool dispatch once `--session-id` is
  set).

**Closed by the 2026-06-04 fix stack**:

- ✅ Drop-after-N message-delivery pattern (multiple buffered
  inbounds, first succeeds, second aborts → subsequent never
  delivered). Per-entry `_last_sent_index` advance (`8dc81f1`)
  guarantees the next retry picks up from the first undelivered
  entry without re-sending the prior ones. Regression test
  `test_session_persistent_advances_sent_index_per_entry_on_partial_failure`.
- ✅ Synthetic `[Error: …]` UserMessages polluting claude's
  session JSONL on every transient error. In-place retry
  (`ac287d1`) via `AnthropicCLIRetryableError` keeps transient
  events inside `send_with_retry`'s budget so the runtime never
  publishes `ModelResponseError`.
- ✅ "API instability" mis-diagnosis. Cross-correlation on
  2026-06-04 ~08:52 + ~08:57 proved the dominant cause of
  `aborted_streaming` was our own SIGINT preempt, not Anthropic
  stream stability. 0-input-token aborts are the smoking gun
  (API can't abort a request that never reached it).
- ✅ Routine peer messages interrupting recipient's in-flight
  work. Default `AgentSendMessage.urgent=False` (`774eb6b`)
  makes peer traffic buffer-and-drain by default; `urgent=True`
  reserved for STOP / pivot / same-sender corrections. ~72% of
  pre-fix TL preempts were routine peer FYIs that shouldn't
  have interrupted.
- ✅ Operator back-to-back typing interrupting in-flight
  responses. Plugin-layer default `UserMessage.urgent=False`
  via `/api/post` (`c1b8fa9`); web UI gains a persistent
  "interrupt" toggle button + Ctrl+Enter per-message override.
  Plain Enter buffers.

**Validation evidence** (2026-06-04 ~10:00–10:24 UTC, post-deploy
of the full fix stack, organic multi-agent coordination):

| Agent | Runtime-visible errors | Completed turns | Sends |
|---|---|---|---|
| TL | **0** | 8 | 4 |
| SWE | **0** | 2 | 2 |
| tech-writer | **0** | 4 | 4 |
| statistician | 0 | 0 | 0 (idle) |
| junior-swe | 0 | 0 | 0 (idle) |

2 transient `aborted_streaming` events caught silently by
`send_with_retry`; runtime never published a `ModelResponseError`.
Zero `urgent=True` invocations during the window — proving routine
coordination doesn't need interrupt-class messaging.

Compare to the same morning's 06:45–09:25 window (pre-fix-stack,
similar workload): TL had a 56.7% per-turn error rate (21 errors
/ 16 completes) with 50+ `[Error: ...]` synthetic UserMessages
accumulated in claude's session JSONL.

**Open**:

- [ ] `bin/merge_jsonl.py` round-trip across both streams (~30 min check).
- [ ] End-to-end PR drive (implementation phase, not just plan mode).
- [ ] Phase 6 cutover decision file at
  `claude-config/project/worklog/decisions/2026-06-02-blackjax-chat-cutover.md`.
- [ ] Long-running soak test (multi-hour session under organic
  `aborted_streaming` flare).

---

## Known issues

### A. Opus occasionally writes the reply as plain text without calling `sagent_send`

Observed 12:31. TL produced a 1.8 KB recap as plain assistant text
with empty `tool_calls`. The reply appears in `sessions/tl.trace.jsonl`
but NOT in `main.jsonl` or the chat view.

**Operator recovery:** when TL appears idle but the expected reply is
missing, check the most recent `ModelResponseComplete` in
`sessions/tl.trace.jsonl` — the body is there. Manually `POST` it via
`/api/post` with `from=tl, to=user` if you want it surfaced.

No structural fix yet. Stronger `PEER_MESSAGING` wording has been
tried twice with limited effect.

### B. `aborted_streaming` / `ede_diagnostic` errors (v1's biggest pain, structurally resolved in v2)

In v1 this was the dominant operational pain point. Across
2026-06-02 the Anthropic streaming API fired
`SubprocessTransportError: aborted_streaming` and `ede_diagnostic`
errors on opus and sonnet roughly once every 3–10 minutes during
sustained traffic. v1's failure mode chain:

- The agent's CLI subprocess dies mid-stream.
- Sagent publishes `ModelResponseError` + respawns.
- The respawn re-feeds history but **strips assistant turns**
  (`anthropic_cli.py:537`), so the new subprocess has no record
  of its own prior delegations and tends to re-issue them.
- Each respawn ate a full prompt-cache miss (sagent's re-fed
  bytes ≠ claude's session-resume bytes; the cache key didn't
  match).

v2 changes the structure:

- Sagent doesn't re-feed history. Each turn is a fresh
  `claude --print --resume <uuid>` subprocess.
- Claude's session JSONL on disk already contains every prior
  assistant turn including tool_use + tool_result blocks. The
  respawned subprocess `--resume`s that, byte-for-byte
  identical to what claude itself would write — so Anthropic's
  prompt-cache key matches and the resumed turn is a cache hit.
- The re-delegation symptom can't happen because the model
  sees its own prior outputs in the resumed session.

**What remains of the v1 pain in v2**:

- Each `aborted_streaming` event still adds latency — sagent
  publishes `ModelResponseError` to the runtime, and the next
  turn spawns a fresh process. But the model continuation is
  correct without operator intervention or observer scaffolding.
- The `retry_delay_ms` schedule the API itself emits is still
  ignored. Honouring it (upstream `sagent/agent/retry.py:345`
  whitelist expansion) would mean the SAME subprocess just
  retries instead of dying — even cheaper than `--resume`. Not
  yet staged.

**Operator playbook under v2**:

- `aborted_streaming` events that happen between turns are
  invisible to the operator now — the next turn just
  `--resume`s cleanly. Server log shows
  `ModelResponseError` followed by a fresh `ModelCallStarted`
  with no observer activity.
- `aborted_streaming` events that interrupt an in-flight turn
  surface as `agent.status == "hung"` briefly while the
  subprocess respawns. The next turn picks up via `--resume`.
- If many errors stack (rare in v2 but still possible),
  `/api/restart` wipes sagent's in-memory state AND
  ``agent.clear()`` reuses the same session_id — so the
  agent's claude session JSONL is **NOT** deleted unless you
  manually `rm` it. Useful when you want to flush sagent's
  inbox without losing claude's conversation history; risky
  when sagent and claude have drifted out of sync (rare).

---

## Status

Plugin is functional for daily operator use. The v2 pivot
removed the dominant residual risk that hung over v1
(`aborted_streaming` re-delegation cascades) at the cost of one
new dependency (the `--session-id` / `--resume` mode of
`claude --print`, which is documented but rarely used in
production tooling). `channel/` can be shut down in parallel
whenever ready. The Phase 6 decision file is the remaining
paperwork.

**v2.1 (materializer)** is default-on and has been live-validated:
sagent owns the session JSONL, the startup canary tripwire guards
against CLI-format drift, and restart resume-from-memory is proven
(TL recalled in-flight PR commit SHAs after a restart). The deep
debugging arc that hardened it — the cumulative-cache-read compaction
over-trigger, two `role alternation` wedges, the `signature_delta`
gap, and the tape-rehydration design — is recorded in
[`worklog/threads/v2.1-cli-session-materialize.md`](../../../claude-config/project/worklog/threads/v2.1-cli-session-materialize.md).
Known follow-ups: a `NoticeMessage` on retry-divergence so a model
learns to stop chaining `pre-commit && commit` past the subprocess
timeout (the divergence marker is trace-only today); and a live
`ContextSplice` round-trip check. The
`sagent/providers/anthropic_cli_session/` core + the provider-boundary
usage normalization are written to be upstream-PR-eligible — the whole
session slice lives inside `sagent/providers/` by design (see the
`2026-06-09-sagent-upstream-split` decision doc); the branch is rebased
on current `upstream/main` to keep that cheap.
