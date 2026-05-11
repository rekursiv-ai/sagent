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
sagent --provider Google --model gemini-3.1-pro-preview
```

Ask a question at the prompt. The CLI stores a session for the current working directory by default.

For one-shot use, pipe a prompt on stdin:

```bash
printf 'Say hi in one sentence.' | \
  sagent --provider Google --model gemini-3.1-pro-preview --output-format json
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

from sagent import tools
from sagent.agent import Agent
from sagent.lib.json import json_freeze
from sagent.providers import Google


async def main() -> None:
    agent = Agent(
        model=Google.from_env().model("gemini-3.1-pro-preview"),
        system="You summarize local project files concisely.",
        tools=[tools.Read(), tools.Glob(), tools.Grep()],
    )
    result = await agent.run(
        json_freeze({"prompt": "Read /tmp/sagent-demo/notes.md and summarize it."})
    )
    print(result.content)


asyncio.run(main())
```

Run it:

```bash
python /tmp/sagent-demo/analyze.py
```

The agent can read the file through the `Read` tool instead of receiving file content in the prompt.

## 5. Persist a session

Use an explicit session directory when follow-up calls should share state:

```python
agent = Agent(
    model=Google.from_env().model("gemini-3.1-pro-preview"),
    system="You summarize local project files concisely.",
    tools=[tools.Read(), tools.Glob(), tools.Grep()],
    session_dir="/tmp/sagent-demo/session",
)
```

CLI equivalent:

```bash
sagent --provider Google --model gemini-3.1-pro-preview --session /tmp/sagent-demo/session
sagent --provider Google --model gemini-3.1-pro-preview --continue
```

Use `--no-session-persistence` for prompts that should not write conversation state or auto-memory to disk.

## 6. Add machine-readable output

For scripts, use `--output-format json` or `stream-json`:

```bash
printf 'Summarize /tmp/sagent-demo/notes.md' | \
  sagent --provider Google --model gemini-3.1-pro-preview \
  --tools Read \
  --output-format stream-json
```

`stream-json` prints newline-delimited event records plus a final `descriptor: "result"` record.

## 7. Add a reviewer child

`AgentSpawn` lets one agent delegate isolated work to a child agent. Give children only the tools and depth they need.

```python
agent = Agent(
    model=Google.from_env().model("gemini-3.1-pro-preview"),
    system="Draft a short answer, then spawn a reviewer before finalizing.",
    tools=[tools.AgentSpawn()],
)
```

The parent can call `AgentSpawn` with `tools=[]` and `max_depth=0` for a pure text review.

## 8. Use a custom tool

For deterministic functions, use `@tool`:

```python
from sagent.tools import tool


@tool
def word_count(text: str) -> int:
    return len(text.split())
```

Then pass `word_count` into `Agent(..., tools=[word_count])`.

## Troubleshooting

- Missing API key: set the provider key or choose another provider with `--provider`.
- Unknown model: pass a model ID supported by the selected provider.
- Session surprises: use `--session PATH` for an explicit location or `--no-session-persistence` for stateless runs with auto-memory disabled.
- Tool access: tools operate in the user's environment. Only pass tools the agent needs.
- Long context: keep compaction enabled or lower the task scope with narrower tools/prompts.

## Next steps

- [Python API](api.md)
- [Tools](tools.md)
- [CLI](cli.md)
- [Providers](providers.md)
- [Streaming](streaming.md)
- [Compaction](compaction.md)
- [Slack service](slack.md)
- [Security](security.md)
