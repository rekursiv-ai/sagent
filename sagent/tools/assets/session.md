Read and edit your own session: what you see now, and everything before it. Read another agent's with `agent`.

Actions:
- `list` -- each entry of your context, one line: its id, what it is, its tokens, and how it starts. Use it to find what you no longer need.
- `read` -- the entries `ids`, whole; or the records `records`, by the number `search` or a watch feed gives, whole even if pruned or compacted away. A call's record comes with its result.
- `search` -- find `text` (case ignored) in the whole session, compacted windows included.
- `estimate` -- what pruning the tool results `ids` would save, and what it costs: dropping a result breaks the prompt cache at it, so everything after it is written once more.
- `prune` -- replace the tool results `ids` with a short stub, after this call. Say why in `reason`. Prune what you will not need again; the result is gone from your context, though the session's journal keeps it.
- `compact` -- start over from a summary of the session, after this call; `prompt` says what it must keep. `style` is `summary` (condense what you see) or `session-audit` (audit the whole session: every user message verbatim, the steering, the decisions, the work, and what is pending); omitted, your compactor's style. The style chosen stays chosen.
- `clear` -- start over with nothing, after this call. Rare and destructive.

`agent` -- a label, or `parent`, `children`, ...: `list`, `read` and `search` look at that agent's session instead of yours. Only reads reach another's.

Your context is compacted and pruned automatically as it fills; do not do either defensively. Use these when the user asks, or when you know a result is spent.
