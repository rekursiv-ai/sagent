# Sessions

Sessions persist agent conversation and state across runs. They are plaintext local state and may contain prompts, model responses, tool results, file snippets, paths, metadata, and cost counters.

## Default location

The CLI stores sessions under a project-scoped directory derived from the current working directory:

```text
<data-dir>/rekursiv-ai/sagent/projects/<cwd-slug>/<session-id>/session.jsonl
```

`<data-dir>` is the user's data directory, such as `~/.local/share` on Linux. The slug is based on the resolved working directory. Long slugs are shortened with a hash. Each spawned child keeps its own session under `<session-id>/<label>/`.

## Explicit sessions

Use an explicit journal file when scripts, tests, or applications should control persistence:

```bash
sagent --session /tmp/my-sagent-session/session.jsonl
```

```python
from pathlib import Path

from sagent.agent.agent import Agent
from sagent.session.jsonl import JsonlJournal

path = Path("/tmp/my-sagent-session/session.jsonl")
async with JsonlJournal(path) as journal:
    async with Agent(model=model, journal=journal) as agent:
        print(await agent.send("Pick up where we left off."))
```

Opening the agent resumes what the journal holds. A journal is locked to one process: a second open raises `JournalInUseError`.

To restore the model, account, and system prompt the session last used, call `recall` before opening the agent:

```python
agent = Agent(model=model, journal=journal)
await agent.recall(catalog)
async with agent:
    ...
```

## Resume and continue

```bash
sagent --continue
sagent --resume
sagent --continue-all
sagent --resume-all
```

| Flag | Behavior |
| --- | --- |
| `--continue` | Resume the newest session for the current working directory. |
| `--resume [ID]` | Resume by id prefix, or open a picker for sessions from the current working directory. |
| `--continue-all` | Resume the newest session across all project directories. |
| `--resume-all` | Open a picker across all project directories. |

## Disable persistence

```bash
sagent --ephemeral
```

Use this for prompts that should not write conversation state or memories to disk. Library users omit `journal` for an in-memory run, and leave out the `Memory` tool.

## What persists

The journal is one JSON line per session record, in order:

- user, assistant, and system messages;
- thinking, tool calls, and typed tool results;
- token usage and spend, a child's included;
- compactions and clears, each starting a new window;
- standing records: the model, its settings, the account, and the system prompt.

A compaction or clear starts a new window; the journal keeps every earlier one. `journal.history()` yields every record, and iterating the journal yields the last window.

## Loading sessions

On open, the agent reads the journal's last window back into its session, and counts the spend recorded before. `journal.history()` reads a line cut off mid-write as an `IncompleteRecord`.
