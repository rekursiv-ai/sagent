Patch the current agent's own state.

All fields are optional. Omitted fields are left unchanged.

Fields:
- `status` -- update the terminal/status text. Keep it short: 3-7 words,
  sentence case.
- `context` -- queue a context mutation: `"clear"`, `"compact"`, or
  `"recompact"`. Omit to preserve context.
- `context_prompt` -- optional clear reason or compaction/recompaction
  guidance. Only used with `context`.
- `model_id` -- optional model ID. Known model prefixes infer provider/auth.
- `provider` -- optional provider class name override.
- `auth` -- optional auth method suffix, for example `env` or `credentials`.
- `account` -- optional credential account name.
- `max_request_tokens` -- optional request-token limit.
- `max_response_tokens` -- optional response-token limit.
- `model_options` -- optional provider/model-specific settings. To see possible
  keys for the active or selected model, call with `diagnostics=true`.
- `diagnostics` -- set true to include current model, usage, limits, cache,
  and supported `model_options` in the result.
- `catalog` -- optional read-only diagnostics query: `"providers"` lists known
  provider names; `"models"` lists known models for `catalog_provider` or the
  active provider.
- `catalog_provider` -- provider name for `catalog="models"`.

Top-level fields are stable Sagent semantics. Provider-specific controls such
as `thinking`, `effort`, and `cache_ttl` belong under `model_options` and are
validated against the active or selected model.
