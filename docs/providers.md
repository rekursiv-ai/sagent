# Providers

Sagent separates providers from models. A provider owns authentication and creates model objects. A model sends typed requests and returns typed responses.

## Public provider matrix

| Provider class | Environment variable | Default model | Utility model | Notes |
| --- | --- | --- | --- | --- |
| `Anthropic` | `ANTHROPIC_API_KEY` | `claude-opus-4-7+1m` | `claude-haiku-4-5` | Anthropic API-key provider. |
| `OpenAI` | `OPENAI_API_KEY` | `gpt-5.5` | `gpt-5.4-mini` | OpenAI API provider. |
| `Google` | `GOOGLE_API_KEY` | `gemini-3.1-pro-preview` | `gemini-3-flash-preview` | Google Gemini provider. |
| `Moonshot` | `MOONSHOT_API_KEY` | `kimi-k2.6` | provider-defined | OpenAI-compatible Kimi provider. |
| `DashScope` | `DASHSCOPE_API_KEY` | `qwen3.6-plus` | provider-defined | Alibaba DashScope provider. |
| `MiniMax` | `MINIMAX_API_KEY` | `MiniMax-M2.7` | provider-defined | MiniMax provider. |
| `OpenAICompat` | subclass-defined | subclass-defined | subclass-defined | Base class for chat-completions-compatible APIs. |

The public package is designed around API-key providers.

## Basic usage

```python
from sagent.providers import Google

provider = Google.from_env()
model = provider.model("gemini-3.1-pro-preview")
utility = provider.utility_model()
```

All public API-key providers support:

```python
ProviderClass.from_env()
ProviderClass.from_key("...")
ProviderClass.from_env().model("model-id")
ProviderClass.from_env().utility_model()
```

`model(None)` uses the provider's default model. Unknown model IDs raise with the provider's known model list.

## CLI dispatch

```bash
sagent --provider Google --auth env --model gemini-3.1-pro-preview
```

`--provider` is the provider class name from `sagent.providers`. `--auth env` calls `Google.from_env()`.

If the named factory does not exist, Sagent treats `--auth` as a literal API key and calls `from_key(...)`:

```bash
sagent --provider Google --auth "$GOOGLE_API_KEY" --model gemini-3.1-pro-preview
```

Prefer environment variables so keys do not land in shell history.

## Provider inference

Agent tools can infer a provider switch from model ID prefixes:

| Prefix | Provider |
| --- | --- |
| `claude` | `Anthropic` |
| `gemini` | `Google` |
| `gpt`, `chatgpt`, `o1`, `o3`, `o4`, `codex` | `OpenAI` |
| `kimi`, `moonshot` | `Moonshot` |
| `qwen` | `DashScope` |
| `minimax` | `MiniMax` |

This is used by model-switching tools so callers can usually pass just `model_id`.

## Context-window tags

Anthropic model IDs may include window tags such as:

```bash
sagent --provider Anthropic --model claude-sonnet-4-6+200k
sagent --provider Anthropic --model claude-opus-4-7+1m
```

The provider strips the tag for API calls and uses it to select the request-token budget.

## OpenAI-compatible endpoints

Subclass `OpenAICompat` for endpoints that implement OpenAI chat completions.

```python
from sagent.providers import OpenAICompat
from sagent.providers.lib.cost import ModelProfile, Pricing


class LocalProvider(OpenAICompat):
    ENV_VAR = "LOCAL_OPENAI_API_KEY"
    BASE_URL = "http://localhost:8000/v1"
    DEFAULT_MODEL = "local-model"
    DEFAULT_UTILITY_MODEL = "local-model"
    KNOWN_MODELS = {
        "local-model": ModelProfile(
            max_request_tokens=128_000,
            max_response_tokens=8_192,
            pricing=Pricing(),
        ),
    }
```

Use it like any other provider:

```python
model = LocalProvider.from_env().model("local-model")
```

`OpenAICompat.from_env(base_url=...)` and `from_key(api_key, base_url=...)` can override the class `BASE_URL` at construction time.

See `examples/openai_compatible_provider.py` for a runnable version.

## Model contract

A model exposes:

- `buffer(request)`: send one request and return a complete response.
- `stream(request, on_text=None)`: stream text chunks through `on_text` while returning a complete response.
- `is_context_overflow(error)`: classify provider context-window errors.

`ModelResponse` includes content, stop reason, token counts, response identifiers, cache counts, and cost fields.

## Pricing and costs

Provider model profiles include token limits and per-million-token pricing. Sagent records request, response, cache-write, and cache-read tokens for every response. `Agent.total_cost_usd` and CLI `--max-budget-usd` use these costs.
