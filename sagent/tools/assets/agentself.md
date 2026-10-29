Patch your own agent state.

All arguments optional:
- `status` -- 3-7 words, sentence case. The session's title in the terminal. Set it after the first user message, and again when the session's focus changes.
- `provider` / `model` -- switch model; the next model call uses it, keeping your window and settings where it offers them. `model` is a name or a role (`best`, `default`, `everyday`, `utility`), with tags in any order: a window (`+fullcontext`, `+200k`), an effort (`+highthink`), a tier (`+fast`). To get a bigger window, switch `model`.
- `effort` / `budget` / `output` / `service_tier` -- reasoning and serving settings, validated against the model in force after any switch; an invalid one rejects the whole patch and nothing changes. `budget: none` stops reasoning; `output` is what of it comes back.
- `diagnostics` -- report your label, model, settings, limits, and directory.
- `catalog` -- `"providers"` lists providers; `"models"` lists `catalog_provider`'s models (yours by default).
