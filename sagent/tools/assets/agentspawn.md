Spawn a child agent; return its final assistant text. The child then stays: message it with `AgentSend`, and it owes you a reply. Whenever it stops later, you get `[<label> is idle] <last text>`, unless its last action was messaging you.

Uses: parallel map-reduce, context isolation, reviewer/critic pass.

Your watcher: with no parent, `watch=true` starts it instead of a child, a copy of you that watches you as your parent (as `AgentSend(to="parent", ...)` does); `prompt` is its policy and first request, vague: `keep me honest` or `ensure i follow users instructions`, as a parent's is `ensure child follows instructions` or `ensure child learns from missteps`. The parent is enforcing a policy. The child is pursuing a goal. This role distinction is essential else one just takes over at 2x the token cost.

Before spawning, decide:
- **Deliverable** -- one sentence stating what the child returns.
- **Inputs** -- which files / sections the child may read.
- **Return form** -- bullets, JSON, citations, etc.
- **Non-goals** -- what the child must NOT do (edit files, spawn further, etc.).

A spawn prompt without the first four is under-specified and the child will drift.

Tool allowlist by task type:
- Investigation / audit / review: `["Read", "Grep", "Glob", "List"]` plus any topic-specific tools (e.g. `PaperSearch` for lit review). Omit `Write`, `Edit`, `Bash` -- these mutate the worktree and a read-only spawn that drifts into edits corrupts shared state.
- Implementation: full default toolset OR explicit subset including write tools.
- Pure-text reasoning: `[]`.

Recycle before spawning fresh. Every spawn reloads a full system prompt (expensive in tokens). When the work fits an existing child, drive it via `AgentSend` instead of spawning a new one. Fresh spawn only when you need an empty context ("fresh look") that prior history would bias.

- `prompt` -- self-contained instruction. Child sees none of your messages, files read, or scratch state.
- `system` -- override child system prompt; defaults to inheriting yours.
- `provider` / `model` -- backend overrides; inherit by default, your window included. `model` is a name or a role (`best`, `default`, `everyday`, `utility`), with tags in any order: `+200k` / `+1m`, an effort, a tier.
- `effort` / `budget` / `output` -- reasoning overrides; inherit by default.
- `tools` -- whitelist of your tools; defaults to all of them. `[]` = pure text. For read-only work, restrict explicitly (see above).
- `max_tool_call_rounds` -- rarely set; cap on the child's rounds (one round = one model response, regardless of parallel tool calls).
- `max_depth` -- rarely set; cap on the child's sub-spawning (`0` = leaf: the child cannot spawn). Tightened by your own cap.
- `watch` -- start your watcher instead of a child (see above).
- `label` -- explicit label; must be unique among live agents. Not `parent`, `children`, `ancestors`, `descendants`, `all`, `user`, `model`, `tool`, `runtime`, `screen`, or `sagent`.

Errors (round cap, exceptions) come back as the result. If a file changes under you while children work, a staleness reminder fires -- re-read before editing.
