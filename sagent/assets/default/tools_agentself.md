Patch own agent state.

All arguments optional:
- `status` -- 3-7 words, sentence case. Used for session summary and delineation. Also UI title. Should be set after first user message and periodically when session focus changes.
- `context` -- `"clear"` drops history; `"compact"` summarizes; `"recompact"` re-summarizes. Runtime auto-manages near budget -- do NOT invoke defensively on budget warnings. Manual use is destructive and rare. Pair with `context_prompt`.
- `model_id` -- known prefixes infer `provider`/`auth`. Context-window size is part of the id, not a limit: a `+1m` / `+200k` suffix selects the large-window variant (e.g. `claude-opus-4-8+1m` is the 1M-window model). To get a bigger window, switch `model_id`; do NOT raise `max_request_tokens`.
- `provider` -- optional provider class name override.
- `auth` -- optional auth method suffix, for example `env` or `credentials`.
- `account` -- optional credential account name.
- `max_request_tokens` / `max_response_tokens` -- per-call limits. Cannot exceed the active model's own window; raising this never unlocks a larger context (use a `+1m`/`+200k` `model_id` for that).
- `model_options` -- provider-specific (`thinking`, `effort`, `cache_ttl`, `service_tier`, `latency`). `latency: "fast"` selects fast serving on models that support it. `diagnostics=true` lists supported keys.
- `diagnostics` -- current model, usage, limits, cache, options.
- `catalog` -- `"providers"` or `"models"` (scoped by `catalog_provider`).
