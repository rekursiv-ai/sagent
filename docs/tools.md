# Tools

Tools are ordinary Python objects. They let an agent read files, run commands, search the web, inspect papers, send service requests, coordinate with other agents, and call application-specific code.

Pass only the tools an agent needs. File, shell, and network tools act with the current process permissions.

## Tool protocol

A tool (`sagent.tools.types.Tool`) exposes:

- `name`: what the model calls it by.
- `description`: what the model reads about when and how to call it.
- `schema`: JSON Schema of its arguments.
- `display`: how its calls show in the terminal, a `ToolDisplay`.
- `summary(arguments)`: short label for a call, such as `Edit foo.py`.
- `serialize_key(context, arguments)`: which calls this one must not overlap; `None` runs beside anything, `"*"` runs alone.
- `call(context, arguments)`: run, and return the session record of what it did.

A tool holds no agent's state. Each call gets the calling agent's `ToolContext`: its working directory, the files it has seen, and its result size limit. Arguments that do not fit the schema raise `ValueError`; the runtime returns the message to the model as an error.

Optional protocols add standing context: `Briefing` (a system prompt section, read once a window) and `Reminding` (a note each round).

## Function tools

Use the decorator for plain functions. The signature is the schema; the docstring is the description, and its `Args:` lines describe each parameter.

```python
from sagent.tools.function import tool


@tool
def add(a: int, b: int) -> str:
    """Add two numbers.

    Args:
      a: The first number.
      b: The second number.

    """
    return str(a + b)
```

A `str` return is cut to what one result holds. A parameter named `context` receives the `ToolContext` instead of an argument. `tool(fn, name="Add")` renames it.

## Class-based tools

Use a class when the tool needs state, a custom summary, or a typed result. Declare the arguments once, as a dataclass whose fields are `Annotated[T, "description"]`; `json_schema` builds the schema from it and `parse` checks the model's arguments against it.

```python
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated

from sagent.session.types import UncategorizedToolResult
from sagent.tools.display import ToolDisplay
from sagent.tools.types import Arg, ToolContext, json_schema, parse
from treekle import PlainTree


@dataclass(frozen=True, slots=True, kw_only=True)
class Echo:
    @dataclass(frozen=True, slots=True, kw_only=True)
    class Arguments:
        text: Annotated[str, "What to say back."]
        times: Annotated[int, "How many times to say it.", Arg(minimum=1)] = 1

    name = "Echo"
    description = "Say the text back."
    schema = json_schema(Arguments)
    display = ToolDisplay()

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        return "Echo"

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        return None

    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> UncategorizedToolResult:
        a = parse(arguments, Echo.Arguments)
        return UncategorizedToolResult(call_id="", content=" ".join([a.text] * a.times))
```

The runtime fills in `call_id`.

## Built-ins

Each built-in lives in its own module under `sagent.tools`, such as `sagent.tools.read.Read` and `sagent.tools.glob_tool.Glob`. `AgentSpawn`, `AgentSend`, `AgentSelf`, and `Session` take the session's `Team`: `AgentSpawn(team=team)`.

| Tool | Purpose | Key inputs |
| --- | --- | --- |
| `Read` | Read text, images, PDFs, notebooks. | `file_path`; optional `offset`, `limit`, `last_lines`, `pages`. |
| `Write` | Create or overwrite files. | `file_path`, `content`. Existing files must have been read first. |
| `Edit` | Exact string replacement. | `file_path`, `old_string`, `new_string`; optional `replace_all`. |
| `Glob` | Find files by filename pattern. | `pattern`; optional `path`, `sort`, `long`, `max_results`, `offset`. |
| `Grep` | Search file contents with ripgrep. | `pattern`; optional `path`, filters (`glob`, `type`, `exclude`), and output controls (`output_mode`, context, paging). |
| `List` | List directory entries. | `path`; optional `show_hidden`, `long`, `sort`, `max_results`, `keep_last`. |
| `Bash` | Run shell commands. | `command`, `description`; optional `timeout` (ms). |
| `WebSearch` | Query web search backends. | `query`; optional `backend`, `categories`, `transport`, `allowed_domains`, `blocked_domains`. |
| `WebFetch` | Fetch and extract readable URL content. | `url`; optional `method`, `transport`, `extractor`, `json`, `form`. |
| `PaperSearch` | Search scholarly papers. | `query`; optional `source`, `limit`, `year_from`, `year_to`, `open_access_only`, `abstract_chars`. |
| `PaperDetails` | Paper metadata, references, citations. | `ids`; optional `operation`, `source`, `influential_only`, `year_from`, `limit`, `abstract_chars`. |
| `PaperAuthor` | Author search and author papers. | `query` or `ids`; optional `operation="papers"`, `limit`, year bounds, `abstract_chars`. |
| `PaperFetch` | Download paper PDFs. | `ids`. |
| `Memory` | Save or forget a project memory, keeping `MEMORY.md` in step. | `action`, `name`; a save also `type`, `description`, `content`. |
| `AgentSelf` | Let an agent inspect/mutate itself. | Flat patch object; all fields optional. |
| `AgentSpawn` | Run child agents. | `prompt`; optional `system`, `provider`, `model`, `account`, reasoning settings, `tools`, `max_tool_call_rounds`, `max_depth`, `label`, `watch`. |
| `AgentSend` | Send to a live named agent inbox. | `to`, `content`; optional `delay`. |
| `BackgroundTask` | Manage background tool jobs. | `operation=list|cancel|foreground`; optional `id`. |
| `Session` | List, read, search, prune, compact, or clear the context. | `action`; optional `ids`, `records`, `agent`, `text`, `reason`, `prompt`, `style`. |
| `Skill` | Load a user-authored skill body. | `skill`; optional `args`. |
| `PlayAudio` | Play a short local WAV notification. | `path`. |

## File tools

`Read` supports line windows for text files, page ranges for PDFs, visual rendering for common image formats, and notebook extraction for `.ipynb`.

`Write` is intentionally stricter than normal file writes: if the target file already exists, the agent must have read it in the current tool state. This prevents accidental clobbering of unseen user work.

`Edit` uses exact string matching as its safety gate. A non-unique `old_string` fails unless `replace_all` is true. Edits and writes are serialized per path so concurrent agents do not write the same file at once.

`Glob`, `Grep`, and `List` are preferred over shelling out for discovery, search, and directory listing.

## Bash

`Bash` runs `/bin/bash -c <command>` with the agent process permissions. `timeout` is in milliseconds; the default and the maximum are 600,000 (10 minutes).

When `Bash` is constructed with peer tools, it can warn the model when a dedicated tool should be used instead of a shell command.

### Running the shell through a wrapper

Library users can start the shell through a wrapper command, such as one that runs it
as another user or in a container, and can give the spawned process an explicit
environment. Both options are off by default and cannot be set from the command line.

```python
from sagent.tools.bash import Bash

shell = Bash(
    exec_prefix=("/usr/local/bin/run-isolated",),
    env={"TERM": "xterm-256color", "PATH": "/usr/bin:/bin"},
)
```

- `exec_prefix`: words placed before `/bin/bash`. The wrapper receives the shell
  command as its arguments and must run them with the exit status and the output
  and error streams unchanged. It must also leave the shell's standard input
  connected, because Sagent uses it to stop the shell (see
  [Security](security.md#running-the-shell-through-a-wrapper)). Empty runs the shell
  directly.
- `env`: the environment of the spawned process. `None` inherits the Sagent process's
  environment. A mapping replaces it entirely, so a variable that is not in it, such
  as a provider key, is not passed to the wrapper. The shell sees whatever the
  wrapper carries across.
- Working directory: the wrapper starts in `/`. The shell enters the agent's
  directory itself, falling back to `/` when it cannot.
  Sagent takes the directory the shell reports without checking that it exists,
  because the shell may run where Sagent cannot look.
- The shell's standard input is empty, as it is without a wrapper.

The wrapper changes where the shell runs. It does not change what the other tools can
reach: `Read`, `Write`, `Edit`, `Grep`, `Glob`, `List`, and `Skill` still run inside
the Sagent process.

## Web tools

`WebSearch` tries each backend in turn -- DuckDuckGo, SearXNG (when
`SEARXNG_URL` is set), then Brave (when `BRAVE_API_KEY` is set) -- skipping a
blocked, unconfigured, or empty backend. Answers are cached host-wide for an hour.

Use `allowed_domains` and `blocked_domains` to scope results.

`WebFetch` fetches HTTP(S) URLs, extracts main content, and caches GET results for
15 minutes. Retrieval itself -- site-specific rewrites, the bot-wall fallback
ladder, and transport selection -- belongs to `wesearch`, which treats an
agent-supplied URL as untrusted: every host, including each redirect target,
must resolve to a public address or the fetch is refused.

## Paper tools

Paper tools use Semantic Scholar, OpenAlex, arXiv, and open-access PDF metadata.

`PaperSearch`:

- `source="fused"` reciprocal-rank-fuses Semantic Scholar and OpenAlex. This is the default.
- `source="s2"` queries Semantic Scholar only.
- `source="openalex"` queries OpenAlex only.
- `source="searxng"` queries a self-hosted SearXNG science metasearch.
- Optional filters: `year_from`, `year_to`, `open_access_only`, `limit`, `abstract_chars`.

`PaperDetails` takes `ids`, DOIs or arXiv IDs, one string or a list. Omit `operation` for metadata, use `operation="references"` for cited papers, and `operation="citations"` for citing papers, each with exactly one id. `source` picks the citation backend (`s2` by default, or `openalex`). Citation mode also supports `influential_only` and `year_from`.

`PaperAuthor` searches authors by `query`, returns metadata by `ids`, or lists publications with one id plus `operation="papers"`.

`PaperFetch` downloads PDFs from arXiv first, then open-access URLs. PDF bytes are cached under `<cache-dir>/rekursiv-ai/wesearch/papers`, such as `~/.cache/rekursiv-ai/wesearch/papers` on Linux.

## Agent coordination tools

`AgentSelf` accepts a single patch object. All fields are optional; omitted fields are unchanged:

- `status`: update visible status text.
- `provider`, `model`: switch model; `model` takes a name or role with tags such as `+200k` or `+highthink`.
- `effort`, `budget`, `output`, `service_tier`: reasoning and serving settings, checked against the model.
- `diagnostics`: set `true` to report model, settings, limits, and directory.
- `catalog`, `catalog_provider`: `"providers"` or `"models"` for read-only discovery.

The `Session` tool lists, prunes, compacts, or clears the agent's context.

`AgentSpawn` creates child agents. Children inherit the parent's model, settings, tools, system prompt, and directory, or override provider, model, account, effort, tools, max tool-call rounds, and max depth. A child runs its prompt to idle, returns its last text, then stays alive, receiving messages through `AgentSend`, until the team stops it.

A child with its parent's tools builds its system blocks from its parent's settings, so its first request reads the prefix the parent already cached -- unless what the blocks render changed since the parent last built them (its directory, an `AGENTS.md`, a saved memory). What differs per agent -- its spawn depth, its label, the agents it can message, a persistent child's reply rule -- is told as a reminder with its first turn and retold when it changes, never in the system blocks.

A child of a parent with a journal file stands in that journal, with its own journal path, until it is stopped. Resuming the parent's session starts such children again, each from its own journal. Closing the session, as `Ctrl-D` / `/quit` does, leaves them standing. See [CLI](cli.md) for the slash target grammar.

`AgentSend` sends text to a live named agent inbox, optionally after a delay.

## Background tasks

If `BackgroundTask` is available to an agent, Sagent injects `background` and `delay` fields into other tool schemas.

```json
{"command": "make", "description": "Build it", "background": true}
{"command": "echo done", "description": "Remind me", "delay": 60}
```

A backgrounded tool immediately returns a placeholder result. When the tool completes, its real result is delivered to the agent inbox. Use:

- `BackgroundTask(operation="list")`
- `BackgroundTask(operation="cancel", id="...")`
- `BackgroundTask(operation="foreground", id="...")`

`delay` implies backgrounding.

## Skills and audio

`Skill` loads a user-authored skill by name and returns its full body so the agent can follow it. Skill invocations are tracked across compaction.

`PlayAudio` plays `.wav` files using host audio tools when available and silently no-ops on headless systems.

## Pruning

`Pruner` (`sagent.agent.pruner`) replaces old tool results with a stub once keeping them has cost more than fetching them again. `agent.model.settings.pruning`, a `PruningPolicy`, says which may go: the newest `keep_recent` results and tools in `keep_tools` (`Skill` by default) are never pruned. See [Compaction](compaction.md#pruning).

## Result size

Each result is cut to fit `ToolContext.max_result_chars`, 50,000 characters by default. A tool may hold its own to less: `Bash` keeps 30,000, `Grep` 20,000. A cut keeps whole lines, matches, or paths, so the model can ask for the rest.

When the agent has a journal file, `Bash` writes a cut command's whole output beside it and names the file in the result.
