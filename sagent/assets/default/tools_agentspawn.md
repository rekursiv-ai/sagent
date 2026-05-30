Spawn a child agent; return its final assistant text.

Uses: parallel map-reduce, context isolation, reviewer/critic pass.

Before spawning, decide:
- **Deliverable** -- one sentence stating what the child returns.
- **Inputs** -- which files / sections the child may read.
- **Return form** -- bullets, JSON, citations, etc.
- **Non-goals** -- what the child must NOT do (edit files, spawn further, etc.).

A spawn prompt without these four is under-specified and the child will drift.

Tool allowlist by task type:
- Investigation / audit / review: `["Read", "Grep", "Glob", "List"]` plus any topic-specific tools (e.g. `PaperSearch` for lit review). Omit `Write`, `Edit`, `Bash` -- these mutate the worktree and a read-only spawn that drifts into edits corrupts shared state.
- Implementation: full default toolset OR explicit subset including write tools.
- Pure-text reasoning: `[]`.

Recycle before spawning fresh. Every spawn reloads a full system prompt (expensive in tokens). When the work fits an existing persistent child with the right system prompt, drive it via `AgentSend` instead of spawning a new one. Use `/tasks` to find idle persistent children. Fresh spawn only when you need an empty context ("fresh look") that prior history would bias.

- `prompt` -- self-contained instruction. Child sees none of your messages, files read, or scratch state.
- `system` -- override child system; defaults to inheriting.
- `provider` / `auth` / `model_id` / `account` -- backend overrides; inherit by default. Prefer `*Subscription` (e.g. `OpenAISubscription`) -- no API keys needed.
- `tools` -- whitelist; defaults to full toolset (incl. `AgentSpawn`). `[]` = pure text. For read-only work, restrict explicitly (see above).
- `max_tool_call_rounds` -- rarely set; cap on child rounds (one round = one model response, regardless of parallel-tool fan-out). Inherits from parent/CLI default.
- `max_depth` -- rarely set; cap on child sub-spawning (`0` = leaf i.e., child cannot sub-spawn). Inherits from parent/CLI default; `min`'d with ancestor caps.
- `persistent` -- long-running background agent. Drive via `AgentSend`; plain assistant text is invisible unless it `AgentSend`s back or `notify_on_asleep` fires.
- `notify_on_asleep` (persistent, default true) -- edge-triggered idle ping: `[<label> is idle] <last text>`.
- `label` -- explicit label; must be unique among live persistent agents.

Errors (round cap, exceptions, sub-failures) bubble up. Parallel children share per-path file locks; if a file changes under you, a staleness reminder fires -- re-read before editing. Child usage rolls into root `CostLedger`.
