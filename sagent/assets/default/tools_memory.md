Save or forget a persistent memory for this project. Memories carry over to future sessions, so they give those sessions full context: who the user is, how they work, what to avoid or repeat, and why.

Save immediately when the user asks you to remember something. Forget the memory when asked to.

- `action: save` writes the memory and its line in `MEMORY.md`; saving a name again replaces it.
- `action: forget` removes the memory named `name` and its line.

## Memory types

Each memory belongs to exactly one type:

- **user** -- role, goals, preferences, expertise. Save when you learn something about the person.
- **feedback** -- corrections and confirmed approaches. Save when the user corrects you or validates a non-obvious method. Structure: rule, then **Why:** and **How to apply:** lines.
- **project** -- decisions, ownership, timelines, incidents not derivable from code or git. Save when you learn who/what/why/when. Use absolute dates. Structure: fact, then **Why:** and **How to apply:**.
- **reference** -- pointers to external systems (issue trackers, dashboards, channels). Save when you learn where information lives.

## What NOT to save

Do not persist anything derivable from the current project state:
- Code patterns, architecture, file layout -- read the code.
- Git history, authorship -- use `git log` / `git blame`.
- Bug fixes, debugging recipes -- the fix is in the code.
- Content already in AGENTS.md files.
- Ephemeral work: in-progress state, conversation context.

## Using memories

- Read memories when they seem relevant or the user references prior work.
- You MUST read memory when the user explicitly asks you to recall.
- If the user says to *ignore* memory, do not apply or cite it.
- Memories go stale. Before acting on a remembered function, file, or flag, verify it still exists with Read or Grep.
