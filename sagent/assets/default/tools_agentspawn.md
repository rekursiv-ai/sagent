Spawn a child agent; return its final assistant text.

Uses: parallel map-reduce, context isolation, reviewer/critic pass.

- `prompt` -- self-contained instruction. Child sees none of your messages, files read, or scratch state.
- `system` -- override child system; defaults to inheriting.
- `provider` / `auth` / `model_id` / `account` -- backend overrides; inherit by default. Prefer `*Subscription` (e.g. `OpenAISubscription`) -- no API keys needed.
- `tools` -- whitelist; defaults to full toolset (incl. `AgentSpawn`). `[]` = pure text.
- `max_tool_call_rounds` -- rarely set; cap on child rounds (one round = one model response, regardless of parallel-tool fan-out). Inherits from parent/CLI default.
- `max_depth` -- rarely set; cap on child sub-spawning (`0` = leaf i.e., child cannot sub-spawn). Inherits from parent/CLI default; `min`'d with ancestor caps.
- `persistent` -- long-running background agent. Drive via `AgentSend`; plain assistant text is invisible unless it `AgentSend`s back or `notify_on_asleep` fires.
- `notify_on_asleep` (persistent, default true) -- edge-triggered idle ping: `[<label> is idle] <last text>`.
- `label` -- explicit label; must be unique among live persistent agents.

Errors (round cap, exceptions, sub-failures) bubble up. Parallel children share per-path file locks; if a file changes under you, a staleness reminder fires -- re-read before editing. Child usage rolls into root `CostLedger`.
