Patch own agent state. All fields optional.

- `status` -- 3-7 words, sentence case.
- `context` -- `"clear"` drops history; `"compact"` summarizes; `"recompact"` re-summarizes. Runtime auto-manages near budget -- do NOT invoke defensively on budget warnings. Manual use is destructive and rare. Pair with `context_prompt`.
- `model_id` -- known prefixes infer `provider`/`auth`.
- `provider` / `auth` / `account` -- backend overrides.
- `max_request_tokens` / `max_response_tokens` -- per-call limits.
- `model_options` -- provider-specific (`thinking`, `effort`, `cache_ttl`, `service_tier`). `diagnostics=true` lists supported keys.
- `diagnostics` -- current model, usage, limits, cache, options.
- `catalog` -- `"providers"` or `"models"` (scoped by `catalog_provider`).
