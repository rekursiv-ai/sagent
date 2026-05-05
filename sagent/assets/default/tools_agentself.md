Mutate the current agent's own state.

Use this tool for:
- `operation="status"` — update the terminal titlebar when starting a task
  or changing focus. Keep status short: 3-7 words, sentence case.
- `operation="diagnostics"` — inspect token, cost, model, provider, auth,
  and context-limit state.
- `operation="compact"` — queue conversation summarization to free context.
- `operation="recompact"` — redo the previous compaction with new guidance.
- `operation="model"` — switch the current agent to another model backend.
- `operation="limits"` — change this agent's context-token limits.
- `operation="clear"` — request a destructive conversation-history clear.
  Use only when the user explicitly asks for a clear or fresh start.

Model switching:
- Usually pass only `model_id`. Known model prefixes infer provider/auth.
- Omit `provider`, `auth`, and `account` unless the user explicitly asks for
  a provider/account or inference would choose the wrong backend.
- `auth` is the suffix of a zero-argument `from_<auth>()` constructor on the
  provider, for example `env` for API-key environment variables.

Limit changes:
- Use `operation="limits"` when changing this agent's token limits.
- With `operation="limits"`, include `max_request_tokens`,
  `max_response_tokens`, or both.
- `max_request_tokens` and `max_response_tokens` are token counts, not bytes,
  chars, or percentages.
- Do not attach limit fields to `status`, `clear`, `compact`, `recompact`,
  `diagnostics`, or `model`; those operations do not change limits.
- The current values are the active agent's current token limits, initialized
  from the active model and possibly changed by an earlier limits operation.
