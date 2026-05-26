Spawn a child agent to work on a subtask and return its final output.

Use this tool to:
- Decompose a large task into independent subtasks. Multiple AgentSpawn
  calls in a single response dispatch in parallel (map-reduce).
- Delegate work that shouldn't pollute your main conversation
  context — the child sees only its own prompt and tool results.
- Run a reviewer / critic / second-opinion pass after a draft.

Arguments:
- `prompt` (required) — the task for the child agent, as a complete
  self-contained instruction. The child does not see your messages
  or files you've read.
- `system` (optional) — override the child's system prompt. Defaults
  to inheriting this agent's system prompt.
- `provider` / `auth` / `model_id` / `account` (all optional) --
  swap the child's model backend. Mirrors the CLI flags:
  `provider` is a class name from `sagent.providers`
  (e.g. `Anthropic`, `Google`, `OpenAI`, `Moonshot`);
  `auth` is the suffix of a zero-argument `from_<auth>` classmethod on that
  provider (for example, `env` for API-key environment variables,
  `credentials` for subscription providers);
  `model_id` is the provider-specific model string (e.g.
  `claude-sonnet-4-6`, `gemini-3.1-pro-preview`); `account` selects among
  named credential slots. Each defaults to inheriting
  the parent's value, so passing none = same backend as parent.
  Prefer `*Subscription` providers when listed in the schema; they reuse the
  host's logged-in CLI subscription and don't need API-key env vars.
  Example: `provider="OpenAISubscription", model_id="gpt-5.5"` to
  delegate a review to GPT while staying on Anthropic for the main loop.
- `tools` (optional) — a list of tool names; the child gets only
  these. Defaults to inheriting this agent's full toolset
  (including this ``AgentSpawn`` tool, so children can spawn their own
  subagents). Pass `[]` to give the child no tools (pure text
  generation). Errors on unknown tool names.
- `max_tool_call_rounds` (optional) — cap on child's tool-call rounds
  (model responses that include one or more tool calls). A round with
  many parallel tool calls still counts as one round.
- `max_depth` (optional) — cap on the child's own sub-spawning. 0
  makes the child a leaf; 1 lets the child spawn one generation;
  omit for unbounded.
- `persistent` (optional) — run the child as a long-running background
  agent via `serve_forever()`. Returns immediately with the child's
  label. Send subsequent work via `AgentSend(to=<label>, ...)`; manage
  the lifecycle (cancel / list) via `BackgroundTask`. Reply path:
  unlike a non-persistent spawn, a persistent child's plain assistant
  text is **invisible** to you — it's logged only to the child's own
  history. The child must call `AgentSend(to=<your label>, ...)` for
  its words to reach your inbox. The idle-notification payload (see
  `notify_on_asleep`) is the only automatic fallback.
- `notify_on_asleep` (optional, persistent only, default true) — when
  true, the parent's inbox receives a `UserMessage` every time the
  child becomes idle (drained inbox with no work in flight). The
  notification carries the child's last assistant text:
  `[<label> is idle] <last text>`. This is the safety net for children
  that emit plain assistant text instead of calling `AgentSend` back.
  Pass `false` to suppress idle pings entirely. Edge-triggered: one
  notification per idle transition.
- `label` (optional) — explicit label for the child agent. Auto-
  generated if omitted. Required to be unique across live persistent
  agents.

Return value is the child's final assistant message text. If the
child hits `max_tool_call_rounds`, raises an unhandled error, or its own
sub-agent fails, that error bubbles up as the tool result.

Concurrency: multiple AgentSpawn calls in one response run in parallel.
Children edit files under shared per-path locks — a child can't
clobber another child's in-flight edit on the same file. See the
file-staleness system reminders: if a file changes under your feet
(via a child's Edit, a linter, or the user), you'll be notified and
should re-read before editing.

Cost tracking: child usage aggregates into the root agent's
CostLedger automatically. No accounting needed on your end.
