# Tutorial

This tutorial builds a small file-analysis agent. It uses Sagent's public API, local files, and one provider key.

## 1. Install Sagent

From a checkout:

```bash
uv sync
```

From PyPI:

```bash
pip install sagent
```

Sagent requires Python 3.12.

Set one provider key. The examples below use Google, but any public provider works.

```bash
export GOOGLE_API_KEY=...
```

## 2. Start the CLI

```bash
sagent --provider Google --model gemini-pro-3.1
```

Ask a question at the prompt. The CLI stores a session for the current working directory by default.

For one-shot use, pipe a prompt on stdin:

```bash
printf 'Say hi in one sentence.' | \
  sagent --provider Google --model gemini-pro-3.1 --output-format json
```

## 3. Create a tiny corpus

```bash
mkdir -p /tmp/sagent-demo
cat > /tmp/sagent-demo/notes.md <<'EOF'
# Demo notes

Sagent is a typed Python agent library. It can call tools, keep sessions, compact history, and use multiple model providers.
EOF
```

## 4. Run a Python agent

Create `/tmp/sagent-demo/analyze.py`:

```python
import asyncio

from sagent.agent.agent import Agent
from sagent.providers.catalog import ModelCatalog
from sagent.tools.glob_tool import Glob
from sagent.tools.grep import Grep
from sagent.tools.read import Read


async def main() -> None:
    async with ModelCatalog() as catalog:
        async with Agent(
            model=catalog["google"]["gemini-pro-3.1"],
            system_prompt="You summarize local project files concisely.",
            tools=[Read(), Glob(), Grep()],
        ) as agent:
            print(await agent.send("Read /tmp/sagent-demo/notes.md and summarize it."))


asyncio.run(main())
```

Run it:

```bash
python /tmp/sagent-demo/analyze.py
```

The agent can read the file through the `Read` tool instead of receiving file content in the prompt.

## 5. Persist a session

Pass a journal when later runs should resume the conversation:

```python
from pathlib import Path

from sagent.session.jsonl import JsonlJournal

path = Path("/tmp/sagent-demo/session/session.jsonl")
async with ModelCatalog() as catalog, JsonlJournal(path) as journal:
    async with Agent(
        model=catalog["google"]["gemini-pro-3.1"],
        tools=[Read(), Glob(), Grep()],
        journal=journal,
    ) as agent:
        print(await agent.send("What did you summarize last time?"))
```

CLI equivalent, resuming the same journal file:

```bash
sagent --provider Google --model gemini-pro-3.1 --session /tmp/sagent-demo/session/session.jsonl
```

Use `--ephemeral` for prompts that should write no journal and no memories.

## 6. Add machine-readable output

For scripts, use `--output-format json` or `stream-json`:

```bash
printf 'Summarize /tmp/sagent-demo/notes.md' | \
  sagent --provider Google --model gemini-pro-3.1 \
  --tools Read \
  --output-format stream-json
```

`stream-json` prints newline-delimited event records plus a final `descriptor: "result"` record.

## 7. Add a reviewer child

`AgentSpawn` lets one agent delegate isolated work to a child agent. Children join the parent's `Team`. Give them only the tools and depth they need.

```python
from sagent.agent.team import Team
from sagent.tools.agent_spawn import AgentSpawn

async with ModelCatalog() as catalog, Team(catalog=catalog) as team:
    async with Agent(
        model=catalog["google"]["gemini-pro-3.1"],
        system_prompt="Draft a short answer, then spawn a reviewer before finalizing.",
        tools=[AgentSpawn(team=team)],
    ) as agent:
        team.join(agent)
        print(await agent.send("Explain what a context window is."))
```

The parent can call `AgentSpawn` with `tools=[]` and `max_depth=0` for a pure text review.

## 8. Use a custom tool

For deterministic functions, use `@tool`. The docstring is what the model reads; its `Args:` lines describe each parameter.

```python
from sagent.tools.function import tool


@tool
def word_count(text: str) -> str:
    """Count the words in some text.

    Args:
      text: The text to count.

    """
    return str(len(text.split()))
```

Then pass `word_count` into `Agent(..., tools=[word_count])`.

## Troubleshooting

- Missing API key: set the provider key or choose another provider with `--provider`.
- Unknown model: pass a model ID supported by the selected provider.
- Session surprises: use `--session PATH` to name a journal file, or `--ephemeral` to keep no journal and no memories.
- Tool access: tools operate in the user's environment. Only pass tools the agent needs.
- Long context: keep compaction enabled or lower the task scope with narrower tools/prompts.

## Next steps

- [Python API](api.md)
- [Tools](tools.md)
- [CLI](cli.md)
- [Providers](providers.md)
- [Streaming](streaming.md)
- [Compaction](compaction.md)
- [Security](security.md)
