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
| Sessions | Persist plaintext local state. |

## Recommended tool profiles

For low-risk Q&A over copied text:

```python
tools = []
```

For read-only repository analysis:

```python
tools = [Read(), Glob(), Grep(), List()]
```

For local code editing:

```python
tools = [Read(), Write(), Edit(), Glob(), Grep(), List(), Bash()]
```

For external service automation, add only the service tools required for the task and use narrowly scoped credentials.

## Session privacy

By default, the CLI stores sessions under a project-scoped directory derived from the current working directory. Session state can contain prompts, model responses, tool results, file snippets, local paths, status, model metadata, and cost counters.

Use one of these when persistence is not appropriate:

```bash
sagent --ephemeral
sagent --session /tmp/disposable-sagent-session/session.jsonl
```

`--ephemeral` also drops the `Memory` tool, so existing project memories are not sent to providers and the model cannot write new ones. Library users omit `journal` for in-memory runs, and leave out the `Memory` tool when memories should stay off too.

With a journal file, `Bash` writes a cut command's whole output beside it. Journals are written with owner-only permissions.

## Credentials

Public providers read API keys from environment variables. Keep keys in your shell, secret manager, or CI secret store.

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
```

Keep keys in the environment so secrets stay out of shell history and process listings.

## Shell and filesystem safety

`Skill` loads local/project-authored instruction files into the model context. Treat skill directories as trusted prompt sources, not user-upload locations.

`Bash` runs commands through the local shell. File tools use the current process permissions. Sagent adds safety checks for common editing mistakes, such as requiring `Write` to read existing files first and serializing writes per path, but these are not a sandbox.

Use OS-level isolation for untrusted repositories, generated commands, or broad write access.

### Running the shell through a wrapper

`Bash(exec_prefix=..., env=...)` (see [Tools](tools.md#running-the-shell-through-a-wrapper))
starts the shell through a wrapper command, for example one that runs it as another
user. Sagent does not isolate anything itself; the wrapper does.

What the option does not cover:

- File tools. `Read`, `Write`, `Edit`, `Grep`, `Glob`, `List`, and `Skill` run inside
  the Sagent process with its permissions and see its files. An agent whose shell is
  meant to be kept apart must not be given them.
- Anything else the Sagent process reads or writes for the model, such as `AGENTS.md`
  includes, session files, and saved tool results.
- The environment. `env` is the whole environment of the spawned process; keep
  provider keys and other secrets out of it unless the shell should hold them.

Stopping a shell that Sagent cannot signal. A process group cannot be signaled across
users, so a timeout or a cancelled call cannot rely on `killpg`. Sagent instead holds
the write end of a pipe that is the shell's standard input, and a small guard that
the shell script starts reads the other end:

- A normal end writes one line, and the guard leaves. A background job started with
  its output redirected survives, as it does without a wrapper.
- A timeout, a cancelled call, or the death of the Sagent process closes the pipe with
  nothing written. The guard then ends the shell's process group: terminate, and kill
  two seconds later for anything that ignores it. A killed Sagent process leaves no
  shell behind, unless the command defeats the guard (see below).
- Sagent waits at most five seconds for the wrapper to end. After that it tries to kill
  the wrapper process and logs a warning. The kill is refused when the wrapper belongs
  to another user, so Sagent then waits at most two seconds more for the wrapper's
  output and exit status, and returns. If the wrapper is still running by then, the
  result says that the shell did not stop and carries none of its output. A call never
  waits on a shell that will not stop.

A command that starts its own session or process group, kills the guard, or writes the
line that tells it to stand down to the guard's pipe, can outlive a stop, as a command
that starts its own session can without a wrapper. The guard runs with the shell's own
permissions, so it cannot stop a command that means to escape it; only something that
acts on every process of the other user can.

## Network safety

`WebFetch` accepts only HTTP(S) URLs and rejects hosts that resolve to loopback, private, link-local, multicast, reserved, or unspecified addresses. Redirect targets are checked too. This reduces SSRF risk when Sagent is exposed through hosted surfaces.

Search, fetch, and paper tools still send user/model-provided data to external services. Scope credentials and tool access accordingly.
