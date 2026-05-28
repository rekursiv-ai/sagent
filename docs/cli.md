# CLI

Sagent ships two console entry points:

- `sagent`: terminal and scriptable agent interface.
- `sagent-slack`: Slack Socket Mode service that routes Slack messages to persistent agents.

Both entry points share provider, model, tool, compaction, budget, and prompt flags.

## Quick start

```bash
export GOOGLE_API_KEY=...
sagent --provider Google --model gemini-3.1-pro-preview
```

The public package defaults to API-key auth with `--auth env`. Provider API keys are read from the provider's environment variable.

For non-interactive use, pipe a prompt on stdin:

```bash
printf 'Summarize this repository in five bullets.' | \
  sagent --provider Google --model gemini-3.1-pro-preview --output-format json
```

`sagent` starts the REPL only when both `--input-format` and `--output-format` are `text`. Any machine-readable format runs one headless request and exits.

## Provider and model flags

```bash
sagent --provider Anthropic --model claude-sonnet-4-6
sagent --provider OpenAI --model gpt-5.5
sagent --provider Google --model gemini-3.1-pro-preview
sagent --provider Moonshot --model kimi-k2.6
sagent --provider DashScope --model qwen3.6-plus
sagent --provider MiniMax --model MiniMax-M2.7
sagent --provider SelfHosted
sagent --provider SelfHosted --model /opt/models/qwen3.6-27b+bfloat16+cuda
```

| Flag | Meaning |
| --- | --- |
| `--provider NAME` | Provider class from `sagent.providers`, such as `Anthropic` or `Google`. |
| `--auth METHOD` | Calls `Provider.from_<METHOD>()`; default is `env`. |
| `--account NAME` | Optional named credential slot for providers that support one. |
| `--model ID` | Provider-specific model ID. Anthropic IDs may include `+1m` or `+200k`; SelfHosted IDs may include `+cuda`, `+bfloat16`, or `+compile`. |
| `--system TEXT` | Extra system prompt instructions appended to Sagent's default prompt. |
| `--effort LEVEL` | Provider-specific reasoning effort. Anthropic accepts `low`, `medium`, `high`, `xhigh`, `max`. |
| `--max-response-tokens N` | Limit response tokens for each model call. |
| `--log-level LEVEL` | Enable stderr diagnostics. Same values as Python logging levels. |

If no `from_<auth>` factory exists, Sagent treats `--auth` as a literal API key and passes it to `from_key(...)`. Prefer environment variables for shell history safety.

For self-hosted HuggingFace models, `SelfHosted` defaults to
`Qwen/Qwen3.6-27B` and treats `--model` as a repo ID or local snapshot path with
optional `+` settings:

```bash
sagent --provider SelfHosted --model Qwen/Qwen3.6-27B+bfloat16+cuda
sagent --provider SelfHosted --model Qwen/Qwen3.6-27B+cuda+bfloat16
sagent --provider SelfHosted --model /opt/models/qwen3.6-27b+bfloat16+cuda
```

For short local Qwen smoke tests, use `--effort none` to disable reasoning
traces:

```bash
sagent --provider SelfHosted --model Qwen/Qwen3-0.6B+float16+cuda \
  --effort none --max-response-tokens 32 --max-tool-call-rounds 1
```

Add `--log-level DEBUG` or set `SAGENT_LOG_LEVEL=DEBUG` when debugging local
model load, device placement, prompt rendering, generation, or tool-call
parsing.

Supported SelfHosted options are `cpu`, `cuda`, `mps`, `auto`, `float16`,
`bfloat16`, `float32`, and `compile`. Options can appear in any order, but each
category can appear only once.

## Tool flags

```bash
sagent --tools Read Glob Grep WebSearch
sagent --tools Read Bash BackgroundTask
```

`--tools` accepts exact class names exported from `sagent.tools`. The default terminal tool set is:

```text
AgentSpawn AgentSend AgentSelf Bash Read Write Edit Grep Glob List WebSearch WebFetch PaperSearch PaperDetails PaperAuthor PaperFetch PlayAudio Skill
```

`Wiki` is available only when requested explicitly, for example
`--tools Read Grep Wiki`.

Use `--tools none` for a model-only run with no tools.

`Bash` is wired with its sibling tools so it can suggest dedicated alternatives, such as `Grep` instead of shell grep, when those tools are available.

`--add-dir DIR...` adds extra directories whose `AGENTS.md` files extend the prompt beyond the current working directory.

## Sessions

```bash
sagent --session /tmp/sagent-session
sagent --continue
sagent --resume
sagent --continue-all
sagent --resume-all
sagent --ephemeral
```

| Flag | Behavior |
| --- | --- |
| `--session PATH` | Use an explicit session directory. Overrides resume/continue. |
| `--continue` | Resume the newest session for the current working directory, or start fresh if none exists. |
| `--resume` | Open an interactive picker for sessions from the current working directory. |
| `--continue-all` | Resume the newest session across all projects. |
| `--resume-all` | Open an interactive picker across all projects. |
| `--resume-persistent` | Restart live persistent subagents recorded in the session (default). |
| `--no-resume-persistent` | Skip persistent-subagent restart when resuming a session. |
| `--ephemeral` | Keep conversation state in memory and disable auto-memory. |

When resuming, Sagent re-reads the parent session's persistent-subagent
lifecycle records and restarts every child whose latest durable state is
``running``. Cancelled, completed, or stopped children stay archived.
Pass ``--no-resume-persistent`` to suppress restart -- the archived
lifecycle metadata is left intact for later inspection.

Default sessions live under a project-scoped directory derived from the current working directory. See [Sessions](sessions.md).

## Compaction, budgets, and limits

| Flag | Behavior |
| --- | --- |
| `--compact` / `--no-compact` | Enable or disable automatic conversation compaction. Default: enabled. |
| `--max-budget-usd USD` | Stop once cumulative model cost reaches this run budget. |
| `--max-tool-call-rounds N` | Limit model/tool-call rounds for one prompt. Default: unlimited. |

Compaction keeps long sessions within the model context window and writes pre-compaction transcripts when session persistence is enabled. See [Compaction](compaction.md).

## Machine-readable I/O

```bash
printf '{"prompt":"Say hi"}\n' | \
  sagent --provider Google --model gemini-3.1-pro-preview \
  --input-format stream-json --output-format stream-json
```

| Format | Input behavior | Output behavior |
| --- | --- | --- |
| `text` | Read all stdin as one prompt. | Print final content as text. |
| `json` | Not accepted for input. | Print `{"content": "..."}`. |
| `stream-json` | Read NDJSON objects and join each `prompt` field. | Print event records, then final `{"descriptor":"result","content":"..."}`. |

`stream-json` output uses message descriptors such as `text/plain`, `text/x-tool-label`, `multipart/x-tool-result`, and `application/x-done`. See [Streaming](streaming.md).

## REPL flags and keys

```bash
sagent --name reviewer --history ~/.sagent-history-reviewer
```

| Flag | Behavior |
| --- | --- |
| `--name NAME` | Agent name used for the REPL and live-agent registry. |
| `--history PATH` | Prompt-toolkit input history file. |
| `--advisor MODEL` | Add an `advisor` tool backed by another model. |
| `--advisor-max-uses N` | Cap advisor calls for the session. |

REPL keys:

- `Enter`: submit.
- `Alt+Enter`: insert newline.
- `Up`: move queued input back into the buffer.
- `Ctrl+X Ctrl+E`: edit in `$EDITOR`.
- `Ctrl+C`: cancel current input or operation.

### Steering subagents

`/send`, `/halt`, and `/kill` accept a shared target syntax for live
persistent subagents:

| Syntax | Matches |
| --- | --- |
| `label` | Exact agent label. |
| `fix-*` | Glob-style match against agent labels. |
| `{a,b,c}` | Explicit comma-separated list. |
| `/regex/` | Regex search over labels. |

```text
/send fix-tools continue from the last failing test
/send fix-* /model claude-sonnet-4-7
/send {fix-tools,fix-compact} continue
/halt /fix-.*/
/kill fix-tools
```

`/send <target> <text>` delivers a plain message to each matched
subagent's inbox; if the body starts with `/`, the supported
subagent-control verbs are `/model`, `/thinking`, `/halt`, and `/quit`.
`/halt <target>` halts the matched subagents (bare `/halt` halts the
local agent). `/kill <qid|all>` keeps its tool-task meaning, and
`/kill <target>` cancels matched persistent subagents through the same
graceful path as `BackgroundTask cancel persistent:<label>` -- a
terminal `cancelled` lifecycle record is written before the child shuts
down. Zero matches surface a visible error.

### Service suspensions

When a model provider returns a recoverable backoff response (HTTP 429,
529, or `retry-after`), Sagent publishes a `ModelServiceSuspended`
runtime event rather than streaming a retry banner into the assistant
text. The REPL renders a single dim line:

```
[model service suspended: rate-limited; resumes at 11:40:00 (in 32m 18s)]
```

Short waits (under a minute) show ``resumes in Ns`` instead. The event
is durable: the resume path reads the latest `retry_at` from the
session log and sleeps the remaining time before the next provider
call. Activity accounting pauses while suspended so per-agent
elapsed-seconds reflects real work, not retry sleep.

The `stream-json` output channel forwards the suspension as a
`application/x-model-service-suspended` record carrying provider,
model, retry timestamps, and a sanitized error snapshot.

## Slack service

`sagent-slack` connects to Slack via Socket Mode, listens for messages, and routes work to persistent Sagent agents.

```bash
export SLACK_APP_TOKEN=xapp-...
export SLACK_BOT_TOKEN=xoxb-...
export GOOGLE_API_KEY=...
sagent-slack --provider Google --model gemini-3.1-pro-preview
```

Slack app setup:

1. Create a Slack app at <https://api.slack.com/apps>.
2. Enable Socket Mode.
3. Generate an app-level token with `connections:write`.
4. Add bot scopes: `chat:write`, `chat:write.customize`, `channels:manage`, `channels:read`, `channels:history`, `app_mentions:read`, `im:history`, `im:read`, `im:write`, `users:read`, `groups:history`, `reactions:read`.
5. Subscribe to `app_mention`, `message.im`, and `reaction_added` events.

Slack-specific flags:

| Flag | Behavior |
| --- | --- |
| `--app-token TOKEN` | Socket Mode app token. Default: `$SLACK_APP_TOKEN`. |
| `--bot-token TOKEN` | Bot token. Default: `$SLACK_BOT_TOKEN`. |
| `--persona-dir DIR` | Directory of persona `.md` files. |
| `--log-prefix PREFIX` | Prefix for per-agent log channel names. |
| `--router-log-channel NAME` | Channel for routing decisions. Default: `router-log`; empty disables. |
| `--cwd PATH` | Change working directory before starting agents. |
| `--session-dir DIR` | Slack session root. Default: `~/.sagent/slack`. |
| `--continue` | Resume agents from the newest Slack session. Exits if none exists. |

Slack commands sent to the bot:

```text
help
list
create <persona>
create <persona> as <label>
stop <name>
```

Routing order:

1. Reactions route to the cached owner of the reacted-to agent message.
2. Messages in an agent log channel route to that owning agent.
3. A message starting with an agent name routes to that agent.
4. Thread replies route to the agent that owns the thread.
5. Human commands are handled by the router.
6. If exactly one agent is active, ordinary messages route to it.
7. Otherwise the router replies with guidance.

Slack agents use persona files from `--persona-dir`. `create sara` loads `sara.md`, then `default.md`, then falls back to `You are sara.`
