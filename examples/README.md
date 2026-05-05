# Sagent examples

These examples are small and copyable. They use the same API as a normal application: construct a provider, choose a model, construct an `Agent`, and pass typed tools into it.

## Prerequisites

Install Sagent. `offline_custom_tool.py` needs no provider key. The other
examples use the provider shown in their source:

```bash
pip install sagent
export GOOGLE_API_KEY=...
```

Set only the key for the example you run.

If a script fails before the first model call, check that the matching key is set and that the requested model exists for that provider.

## Start here

1. `offline_custom_tool.py` -- no-key model/tool loop.
2. `decorator_tool.py` -- smallest custom-tool path.
3. `custom_tool.py` -- full class-based `Tool` contract.
4. `multi_agent_reviewer.py` -- isolated child-agent review.
5. `openai_compatible_provider.py` -- custom OpenAI-compatible backend.

## Custom tools

`decorator_tool.py` wraps a deterministic function with `@tool` and lets Sagent infer the JSON Schema from the function signature.

`custom_tool.py` shows the full class-based `Tool` contract in one file:

- `name`, `tool_id`, `description`, `directive_schema`, `supports_microcompaction`
- `summary(msg)` for human-readable status
- `prompt()` for optional per-request instructions
- `run(msg)` returning `TextMessage`

Run them with:

```bash
python -m examples.decorator_tool
python -m examples.custom_tool
```

Use this shape for application-specific tools: database lookups, product APIs, internal services, or deterministic local functions.

## Multi-agent reviewer

`multi_agent_reviewer.py` shows the smallest useful `AgentSpawn` pattern: a parent agent drafts an answer, then spawns a reviewer with its own system prompt, no tools, one tool-call round, and `max_depth=0` so review stays isolated.

Run it with:

```bash
python -m examples.multi_agent_reviewer
```

Use this shape for map-reduce research, independent review, and context-isolated subtasks. Give children only the tools and depth they need.

## OpenAI-compatible provider

`openai_compatible_provider.py` shows the smallest provider extension for a chat-completions-compatible endpoint such as vLLM, SGLang, LiteLLM, or a hosted OpenAI-compatible API.

Run it with:

```bash
export LOCAL_OPENAI_API_KEY=...
export LOCAL_OPENAI_BASE_URL=http://localhost:8000/v1
python -m examples.openai_compatible_provider
```

If your endpoint has OpenAI-compatible chat completions, subclassing
`OpenAICompat` is usually enough. Override the provider class attributes, then
let Sagent reuse the same agent loop, tools, sessions, compaction, and cost
tracking.

For HuggingFace models loaded locally through `transformers`, use
`SelfHosted` instead:

```bash
pip install "sagent[selfhosted]"
hf download Qwen/Qwen3.6-27B --local-dir /opt/models/qwen3.6-27b
sagent --provider SelfHosted
sagent --provider SelfHosted --model /opt/models/qwen3.6-27b
sagent --provider SelfHosted --model Qwen/Qwen3-0.6B \
  --tools none --effort none --max-tool-call-rounds 1
```

## More docs

- [Tutorial](../docs/tutorial.md)
- [Tools](../docs/tools.md)
- [Providers](../docs/providers.md)
- [Architecture](../docs/architecture.md)
