# CLI

Sagent ships one console entry point, `sagent`: a terminal and scriptable agent interface.

## Quick start

```bash
export GOOGLE_API_KEY=...
sagent --provider Google --model gemini-pro-3.1
```

Provider API keys are read from the provider's environment variable.

Every interactive session shows the Sagent welcome before provider or model
configuration, including a first launch without credentials and a resumed
session. The single frame contains the SAGENT identity, project folder, and
in-session help hints: `/help`, `/tasks` (running tool calls), `/quit`, and
Left on an empty prompt line (`←`), which opens the agent tree. The greeting
sits outside the frame.

After successful setup, the resolved provider/model line appears below the
welcome, followed by the normal chat prompt. Missing credentials keep the
existing setup error and instructions. No new setup wizard, credential store,
or provider-selection behavior is introduced. The welcome makes no model
request and does not verify credentials with the remote service.

The terminal background and body text follow your terminal settings.
`NO_COLOR=1` disables color. The welcome prints at once, with no animation.
Narrow, short, dumb, or non-Unicode terminals use a compact SAGENT heading.

Welcome layout follows a terminal-cell grid:

- The icon sits left of the wordmark, separated by three cells.
- The frame spans the terminal, up to 80 columns, with one empty row above
  and below its content and two cells of horizontal padding.
- Long paths mark omitted ancestors with an ellipsis and retain the project
  suffix. Command descriptions stay with their commands when wrapping.
- The greeting stays outside the frame, flush with its left border.

Piped input does not show a welcome. Redirected stderr keeps plain error output
or the provider/model line. `sagent --help` remains plain argparse help.

For non-interactive use, pipe a prompt on stdin:

```bash
printf 'Summarize this repository in five bullets.' | \
  sagent --provider Google --model gemini-pro-3.1 --output-format json
```

`sagent` starts the REPL when stdin is a terminal. Piped stdin runs one headless request, printed as `--output-format` says, and exits.

## Provider and model flags

```bash
sagent --provider Anthropic --model sonnet-4.6
sagent --provider OpenAI --model sol-6
sagent --provider Google --model gemini-pro-3.1
sagent --provider Moonshot --model kimi-2.6
sagent --provider DashScope --model qwen-plus-3.7
sagent --provider MiniMax --model minimax-2.7
sagent --provider SelfHosted
sagent --provider SelfHosted --model /opt/models/qwen3.6-27b+bfloat16+cuda
```

| Flag | Meaning |
| --- | --- |
| `--provider NAME` | Provider name in the model catalog, in any case, such as `anthropic` or `Google`, or an alias such as `gapi`. |
| `--account NAME` | Optional named credential slot for providers that support one. |
| `--model ID` | Catalog model ID. Bare IDs select the largest context window. Models with smaller profiles accept `+200k` or `+272k`; an explicit `+1m` remains valid when the bare profile is already at least one million tokens. SelfHosted IDs may include `+cuda`, `+bfloat16`, or `+compile`. |
| `--system-prompt TEXT` | System prompt replacing Sagent's default prompt. |
| `--append-system-prompt TEXT` | Instructions added after the system prompt. |
| `--effort LEVEL` | Reasoning effort: `none`, `min`, `low`, `medium`, `high`, `xhigh`, or `max`. Each model accepts a subset (`gpt-5.6` takes all but `min`; `astra-6.0` also has no `none`), and an unsupported value raises. |
| `--log-level LEVEL` | Log level; interactive logs go to the session's `.log` file. |
| `--allow-providers LIST` | Comma-separated providers this agent and its children may use. |

`--provider`, `--account`, `--model`, and `--effort` are settings: every setting `/help` lists is a flag too, as `--NAME VALUE`. If no provider answers to the name, Sagent reports an error. Keep API keys in the environment so secrets do not land in shell history.

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
  --effort none --max-tool-call-rounds 1
```

Add `--log-level DEBUG` or set `SAGENT_LOG_LEVEL=DEBUG` when debugging local
model load, device placement, prompt rendering, generation, or tool-call
parsing.

Supported SelfHosted options are `cpu`, `cuda`, `mps`, `shard`, `float16`,
`bfloat16`, `float32`, `compile`, and a context size such as `200k`. Options can
appear in any order, but each category can appear only once.

## Tool flags

```bash
sagent --tools Read Glob Grep WebSearch
sagent --tools Read Bash BackgroundTask
```

`--tools` accepts tool names, MCP tools included. The default is every tool:

```text
Bash Read Write Edit Grep Glob List WebSearch WebFetch PaperSearch PaperDetails PaperAuthor PaperFetch Memory PlayAudio AgentSpawn AgentSend AgentSelf Session Skill BackgroundTask
```

`Memory` is left out under `--ephemeral`; naming it there is an error.

Use `--tools none` for a model-only run with no tools.

`Bash` is wired with its sibling tools so it can suggest dedicated alternatives, such as `Grep` instead of shell grep, when those tools are available.

`--add-dir DIR...` adds extra directories whose `AGENTS.md` files extend the prompt beyond the current working directory.

## Sessions

```bash
sagent --session /tmp/sagent-session/session.jsonl
sagent --continue
sagent --resume
sagent --continue-all
sagent --resume-all
sagent --ephemeral
```

| Flag | Behavior |
| --- | --- |
| `--session PATH` | Journal file to write and resume. |
| `--continue` | Resume the newest session for the current working directory. |
| `--resume [ID]` | Resume by id prefix, or open a picker for sessions from the current working directory. |
| `--continue-all` | Resume the newest session across all projects. |
| `--resume-all` | Open an interactive picker across all projects. |
| `--resume-persistent` / `--no-resume-persistent` | Restart the persistent agents a resumed session left running. Default: on. |
| `--resume-limit N` | Sessions the picker lists. Default: 20. |
| `--ephemeral` | Keep no journal and no memories. |

A spawned child stands in its parent's journal until it is stopped. When
resuming, Sagent starts each standing child again from its own journal;
`--no-resume-persistent` leaves them stopped, still standing.

Default sessions live under a project-scoped directory derived from the current working directory. See [Sessions](sessions.md).

## Compaction, budgets, and limits

| Flag | Behavior |
| --- | --- |
| `--compact` / `--no-compact` | Enable or disable automatic conversation compaction. Default: enabled. |
| `--compactor {summary,session-audit}` | Condense what the model sees, or audit the whole session from its journal. Default: `summary`. |
| `--prune` / `--no-prune` | Drop old tool output once that costs less than keeping it cached. Default: enabled. |
| `--max-budget-usd USD` | Stop once cumulative model cost reaches this run budget. |
| `--max-tool-call-rounds N` | Limit tool-call rounds for one prompt. Default: unlimited. |

Compaction keeps long sessions within the model context window; the journal keeps every earlier window. See [Compaction](compaction.md).

## Machine-readable I/O

```bash
printf '{"prompt":"Say hi"}\n' | \
  sagent --provider Google --model gemini-pro-3.1 \
  --input-format stream-json --output-format stream-json
```

| Format | Input behavior | Output behavior |
| --- | --- | --- |
| `text` | Read all stdin as one prompt. | Print final content as text. |
| `json` | Not accepted for input. | Print `{"content": "..."}`. |
| `stream-json` | Read NDJSON objects and join each `prompt` field. | Print event records, then final `{"descriptor":"result","content":"..."}`. |

`stream-json` output uses descriptors such as `text/plain`, `text/x-tool-label`, and `application/x-tool-result`. See [Streaming](streaming.md).

## REPL flags and keys

```bash
sagent --name reviewer --history ~/.sagent-history-reviewer
```

| Flag | Behavior |
| --- | --- |
| `--name NAME` | The agent's label in its team. Default: `Agent`. |
| `--history PATH` | Input history file. Default: the per-user one. |

REPL keys:

- `Enter`: submit.
- `Alt+Enter`: insert newline.
- `Up`: move queued input back into the buffer.
- `Ctrl+X Ctrl+E`: edit in `$EDITOR`.
- `Ctrl+C`: cancel current input or operation.

### Steering subagents

`/send` and `/halt` take an `AGENTS` target:

| Syntax | Matches |
| --- | --- |
| `label` | Exact agent label. |
| `parent`, `children`, `ancestors`, `descendants`, `all` | Agents by relation. |
| `fix-*` | Glob-style match against agent labels. |
| `{a,b,c}` | Explicit comma-separated list. |
| `/regex/` | Regex search over labels. |

```text
/send fix-tools continue from the last failing test
/send fix-* /model sonnet
/send {fix-tools,fix-compact} continue
/halt /fix-.*/
/kill fix-tools
```

`/send AGENTS TEXT` messages each matched agent; a body starting with `/`
runs that command on them. `/LABEL.NAME ARGS` is short for
`/send LABEL /NAME ARGS`, as in `/children.effort high`. `/halt AGENTS`
halts the matched agents (bare `/halt` halts the local agent).
`/kill ID|LABEL|all` cancels tool calls, or stops a persistent child.
`/send parent` from a root with no parent starts its watcher.

### Service suspensions

When a model call fails in a way that may pass, such as a rate limit,
Sagent publishes a `ModelCallSuspended` runtime event rather than
streaming a retry banner into the assistant text. The turn waits, then
calls again; the user's message, a new model, or a sign-in calls again
at once. The REPL shows one line, such as:

```
Weekly limit reached. Resumes Sat Oct 31 06:00:00 (1d 22h 3m 5s).
```

The `stream-json` output channel forwards the same text as a
`text/x-notice` record.
