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
sagent --provider SelfHosted --model /opt/models/qwen3.6-27b
```

| Flag | Meaning |
| --- | --- |
| `--provider NAME` | Provider class from `sagent.providers`, such as `Anthropic` or `Google`. |
| `--auth METHOD` | Calls `Provider.from_<METHOD>()`; default is `env`. |
| `--account NAME` | Optional named credential slot for providers that support one. |
| `--model ID` | Provider-specific model ID. Anthropic IDs may include `+1m` or `+200k` context tags. |
| `--system TEXT` | Extra system prompt instructions appended to Sagent's default prompt. |
| `--effort LEVEL` | Provider-specific reasoning effort. Anthropic accepts `low`, `medium`, `high`, `xhigh`, `max`. |
| `--max-response-tokens N` | Limit response tokens for each model call. |
| `--log-level LEVEL` | Enable stderr diagnostics. Same values as Python logging levels. |

If no `from_<auth>` factory exists, Sagent treats `--auth` as a literal API key and passes it to `from_key(...)`. Prefer environment variables for shell history safety.

For self-hosted HuggingFace models, `SelfHosted` defaults to
`Qwen/Qwen3.6-27B` and treats `--model` as either a repo ID or local snapshot
path:

```bash
sagent --provider SelfHosted --model Qwen/Qwen3.6-27B
sagent --provider SelfHosted --model /opt/models/qwen3.6-27b
```

For short local Qwen smoke tests, use `--effort none` to disable reasoning
traces:

```bash
sagent --provider SelfHosted --model Qwen/Qwen3-0.6B \
  --tools none --effort none --max-response-tokens 32 --max-tool-call-rounds 1
```

Add `--log-level DEBUG` or set `SAGENT_LOG_LEVEL=DEBUG` when debugging local
model load, device placement, prompt rendering, generation, or tool-call
parsing.

When `SAGENT_SELFHOSTED_DEVICE` is unset, SelfHosted uses MPS if available,
then CUDA if available, then the PyTorch CPU default.

Set `SAGENT_SELFHOSTED_COMPILE=1` to opt into `torch.compile` for the loaded
SelfHosted model. It is disabled by default because compile can add significant
first-request latency and backend-specific variance.

## Tool flags

```bash
sagent --tools Read Glob Grep WebSearch
sagent --tools Read Bash BackgroundTask
```

`--tools` accepts exact class names exported from `sagent.tools`. The default terminal tool set is:

```text
AgentSpawn AgentSend AgentSelf Bash Read Write Edit Grep Glob List WebSearch WebFetch PaperSearch PaperDetails PaperAuthor PaperFetch PlayAudio Skill Wiki
```

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
sagent --no-session-persistence
```

| Flag | Behavior |
| --- | --- |
| `--session PATH` | Use an explicit session directory. Overrides resume/continue. |
| `--continue` | Resume the newest session for the current working directory, or start fresh if none exists. |
| `--resume` | Open an interactive picker for sessions from the current working directory. |
| `--continue-all` | Resume the newest session across all projects. |
| `--resume-all` | Open an interactive picker across all projects. |
| `--no-session-persistence` | Keep conversation state in memory and disable auto-memory. |

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
