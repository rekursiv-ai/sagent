# Security

Sagent executes the tools you give it. It does not sandbox model behavior, shell commands, filesystem access, provider requests, or network calls.

Use the smallest tool set that can solve the task. For untrusted prompts or repositories, run Sagent inside an OS, container, VM, or network sandbox that matches the risk.

## Capability model

| Capability | Risk |
| --- | --- |
| Provider APIs | Receive system prompts, conversation history, and selected tool results. |
| File tools | Read and write paths the current process can access. |
| `Bash` | Executes shell commands with current process permissions. |
| Web and paper tools | Send queries/URLs/paper identifiers to external services. |
| Slack and Linear | Read and mutate configured external workspaces. |
| Sessions | Persist plaintext local state. |

## Recommended tool profiles

For low-risk Q&A over copied text:

```python
tools=[]
```

For read-only repository analysis:

```python
tools=[Read(), Glob(), Grep(), List()]
```

For local code editing:

```python
tools=[Read(), Write(), Edit(), Glob(), Grep(), List(), Bash()]
```

For external service automation, add only the service tools required for the task and use narrowly scoped credentials.

## Session privacy

By default, the CLI stores sessions under a project-scoped directory derived from the current working directory. Session state can contain prompts, model responses, tool results, file snippets, local paths, status, model metadata, and cost counters.

Use one of these when persistence is not appropriate:

```bash
sagent --no-session-persistence
sagent --session /tmp/disposable-sagent-session
```

`--no-session-persistence` also disables the auto-memory system prompt section, so existing project memories are not sent to providers and the model is not instructed to write new memories. Library users can leave `session_dir` unset for in-memory runs; call `build_system_dict(..., include_memory=False)` when persistent memory should be disabled too.

Large tool outputs may be persisted separately to keep model requests within budget. Session runs store those under `session_dir/tool-results`; no-session runs use `/tmp/sagent_results`. Persisted files are written with owner-only permissions.

Sagent also writes always-on error traces for provider wire-layer failures to `~/.sagent/debug.log` by default. Records include compact prompt and tool-result previews truncated to 200 characters. Set `SAGENT_DEBUG_LOG` to redirect the log; set `SAGENT_DEBUG=1` to add verbose request traces.

## Credentials

Public providers read API keys from environment variables through `--auth env` or `Provider.from_env()`. Keep keys in your shell, secret manager, or CI secret store.

Common environment variables:

```bash
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
MOONSHOT_API_KEY=...
DASHSCOPE_API_KEY=...
MINIMAX_API_KEY=...
SEMANTIC_SCHOLAR_API_KEY=...
OPENALEX_EMAIL=...
SAGENT_DEBUG=...
SAGENT_DEBUG_LOG=...
SLACK_APP_TOKEN=...
SLACK_BOT_TOKEN=...
LINEAR_API_KEY=...
```

Avoid passing literal API keys through `--auth` unless you control shell history and process-list exposure.

## Shell and filesystem safety

`Skill` loads local/project-authored instruction files into the model context. Treat skill directories as trusted prompt sources, not user-upload locations.

`Wiki` reads local/project-authored wiki pages into the model context. Treat wiki roots as trusted content sources.

`Bash` runs commands through the local shell. File tools use the current process permissions. Sagent adds safety checks for common editing mistakes, such as requiring `Write` to read existing files first and serializing writes per path, but these are not a sandbox.

Use OS-level isolation for untrusted repositories, generated commands, or broad write access.

## Network safety

`WebFetch` accepts only HTTP(S) URLs and rejects hosts that resolve to loopback, private, link-local, multicast, reserved, or unspecified addresses. Redirect targets are checked too. This reduces SSRF risk when Sagent is exposed through hosted surfaces such as Slack.

Search, fetch, paper, Slack, and Linear tools still send user/model-provided data to external services. Scope credentials and tool access accordingly.

## Slack and Linear

`Slack` can post messages, list/read channels and threads, list users, and create channels. `sagent-slack` can route workspace messages into agents and create per-agent log channels.

`Linear` can list, read, create, update issues, and add comments.

These actions are externally visible. Use dedicated bot accounts, least-privilege workspace installs, and explicit human review for sensitive workspaces.
