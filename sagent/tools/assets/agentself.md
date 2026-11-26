Patch your own or a descendant's agent state.

All arguments optional:
- `agent` -- agent label or relation; omit for yourself. Your permissions apply;
  the target need not have AgentSelf.
- `checkpoint` reports `progress`, `pending`, and `user_action` as strings. Use empty strings when nothing is pending or no user action is needed. Send null to clear a completed handoff. The report survives resume and appears in /pending.
- `status` -- 3-7 words, sentence case. The session's title in the terminal. Set it after the first user message, and again when the session's focus changes.
- `provider` / `model` -- switch model; the next model call uses it, keeping your window and settings where it offers them. `model` is a name or a role (`best`, `default`, `everyday`, `utility`), with tags in any order: a window (`+fullcontext`, `+200k`), an effort (`+highthink`), a tier (`+fast`). To get a bigger window, switch `model`.
- `effort` / `budget` / `output` / `service_tier` -- reasoning and serving settings, validated against the model in force after any switch; an invalid one rejects the whole patch and nothing changes. `budget: none` stops reasoning; `output` is what of it comes back.
- `compaction` -- who summarizes a full context: `bespoke` (sagent's own prompt) or `provider` (the vendor's API, where the model offers it). `provider` takes no instructions, so a compaction given what to keep, a session audit, or one keeping the head runs `bespoke`.
- `detach` -- when long calls wake you: `{default: KNOBS, per_tool: {Bash: KNOBS}}`. KNOBS sets `first_gap_sec` (first wake), `gap_growth` (next gap as a multiple of time run; 0 fixes the gap), `max_gap_sec`, or `off: true`. The first wake detaches the calls; later ones show what is new.
- `diagnostics` -- report your label, model, settings, limits, and directory.
- `catalog` -- `"providers"` lists providers; `"models"` lists `catalog_provider`'s models (yours by default).
