# Sessions

Sessions persist agent conversation and state across runs. They are plaintext local state and may contain prompts, model responses, tool results, file snippets, paths, metadata, and cost counters.

## Default location

The CLI stores sessions under a project-scoped directory derived from the current working directory:

```text
~/.sagent/projects/<cwd-slug>/<session-id>/
```

The slug is based on the resolved working directory. Long slugs are shortened with a hash.

## Explicit sessions

Use an explicit directory when scripts, tests, or applications should control persistence:

```bash
sagent --session /tmp/my-sagent-session
```

```python
agent = Agent(
    model=model,
    system="You are concise.",
    tools=[],
    session_dir="/tmp/my-sagent-session",
)
```

Passing `session_dir` loads existing state if present and stores large persisted tool results under that directory.

## Resume and continue

```bash
sagent --continue
sagent --resume
sagent --continue-all
sagent --resume-all
```

| Flag | Behavior |
| --- | --- |
| `--continue` | Resume the newest session for the current working directory, or start fresh. |
| `--resume` | Open a picker for sessions from the current working directory. |
| `--continue-all` | Resume the newest session across all project directories. |
| `--resume-all` | Open a picker across all project directories. |

`--session PATH` always wins over resume and continue flags.

## Disable persistence

```bash
sagent --ephemeral
```

Use this for prompts that should not write conversation state or auto-memory to disk. Library users can omit `session_dir` for in-memory runs and pass `include_memory=False` to `build_system_dict` when persistent memory should be disabled too.

No-session runs may still persist oversized tool results temporarily under `/tmp/sagent_results` when needed to keep model requests within context budgets.

## What persists

Sagent writes session metadata and conversation state, including:

- messages;
- event log;
- provider, auth, model ID, and account recipe when known;
- token limits and last token counts;
- total cost;
- tool-call round counts;
- compaction count and summary pointers;
- status string;
- Bash current working directory;
- tool-result replacement state.

Before compaction, Sagent writes numbered `pre_compact_<N>.jsonl` transcripts when a session directory exists.

Large tool outputs are stored under:

```text
<session-dir>/tool-results/
```

## Loading sessions

On load, Sagent restores messages, status, cost and token counters, compaction state, tool-result replacement state, and Bash cwd. It also repairs dangling tool calls so the next provider request is well-formed.

If saved provider metadata is available, Sagent can restore the provider/model recipe rather than only replaying text history.

## Slack sessions

`sagent-slack` uses a separate session root:

```text
~/.sagent/slack
```

It creates timestamped session directories with a manifest of active agents and persona metadata. `sagent-slack --continue` resumes the newest Slack session and exits if none exists.

See [Slack service](slack.md).
