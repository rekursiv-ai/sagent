Read and edit an agent's session: its context and everything before it.
Set `agent` to a label or relation; omit it for your own session.

Actions:
- `list` -- each entry of your context, one line: its id, what it is, its tokens, and how it starts. Use it to find what you no longer need.
- `read` -- the entries `ids`, whole; or the records `records`, by the number `search` or a watch feed gives, whole even if pruned or compacted away. A call's record comes with its result.
- `search` -- find `text` (case ignored) in the whole session, compacted windows included.
- `estimate` -- what pruning the tool outputs `ids` (results, or a long call's news) would save, and what it costs: dropping a result breaks the prompt cache at it, so everything after it is written once more.
- `prune_queue` -- queue the tool outputs `ids` to drop. They stay readable; all queued outputs drop together, in one cache rewrite, once dropping them pays; a descendant's at rest drop at once. Say why in `reason`. Queue what you will not need again: once dropped, a result is gone from your context, and to attend to it again you must reread the file or rerun the command, though the session's journal keeps it.
- `prune_unqueue` -- take `ids` off the queue.
- `prune_list` -- the queued outputs: each id, its tokens, and why it was queued; then what dropping them all saves, and how far that is from paying.
- `compact` -- start over from a summary of the session: yours after this call, a descendant's before it returns; `prompt` says what it must keep. `style` is `summary` (condense what you see) or `session-audit` (audit the whole session: every user message verbatim, the steering, the decisions, the work, and what is pending); omitted, your compactor's style. The style chosen stays chosen.
- `clear` -- start over with nothing: yours after this call, a descendant's before it returns. Rare and destructive.

Mutations and estimates may target yourself or descendants; list, read, and
search may target any agent in your team. Your permissions apply, and the
target need not have AgentSession.

Your context is compacted as it fills, and a `prune` note lists the results whose pruning would pay; do neither defensively. Queue prunes in reply to that note or when the user asks. Cost never justifies partial reads or dropping rules you follow.
