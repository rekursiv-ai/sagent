Patch own agent state.

All arguments optional:
- `status` -- 3-7 words, sentence case. Used for session summary and delineation. Also UI title. Should be set after first user message and periodically when session focus changes.
- `context` -- `"clear"` drops history; `"compact"` summarizes; `"recompact"` re-summarizes. Runtime auto-manages near budget -- do NOT invoke defensively on budget warnings. Manual use is destructive and rare. Pair with `context_prompt`.
- `model_id` -- known prefixes infer `provider`/`auth`.
- `provider` -- optional provider class name override.
- `auth` -- optional auth method suffix, for example `env` or `credentials`.
- `account` -- optional credential account name.
- `max_request_tokens` / `max_response_tokens` -- per-call limits.
- `model_options` -- provider-specific (`thinking`, `effort`, `cache_ttl`, `service_tier`). `diagnostics=true` lists supported keys.
- `diagnostics` -- current model, usage, limits, cache, options.
- `catalog` -- `"providers"` or `"models"` (scoped by `catalog_provider`).
