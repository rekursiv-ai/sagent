# Providers

Sagent separates providers from models. A provider owns authentication and creates model objects. A model sends typed requests and returns typed responses.

## Public provider matrix

| Provider class | Environment variable | Default model | Utility model | Notes |
| --- | --- | --- | --- | --- |
| `Anthropic` | `ANTHROPIC_API_KEY` | `opus-5.5` | `haiku-5.5` | Anthropic API-key provider; `best` resolves to `fable-5.1`. |
| `OpenAI` | `OPENAI_API_KEY` | `sol-6.1` | `luna-6.0` | OpenAI API provider; `best` resolves to `astra-6.0`. |
| `Google` | `GOOGLE_API_KEY` or `GEMINI_API_KEY` | `gemini-argon-4.0` | `gemini-flash-lite-3.5` | Google Gemini provider. |
| `Moonshot` | `MOONSHOT_API_KEY` | `kimi-3.0` | `kimi-2.6` | OpenAI-compatible Kimi provider. |
| `DashScope` | `DASHSCOPE_API_KEY` | `qwen-plus-3.7` | `qwen-flash-3.8` | Alibaba DashScope provider. |
| `MiniMax` | `MINIMAX_API_KEY` | `minimax-3.0` | `minimax-3.0` | MiniMax provider. |
| `SelfHosted` | none | `Qwen/Qwen3.6-27B` | same as default | Local HF transformers provider; `SelfHosted(model=...)` names what every role means. |
| `OpenAICompatible` | subclass-defined | subclass-defined | subclass-defined | Base class for chat-completions-compatible APIs. |

The public package is designed around API-key providers.

## Basic usage

A `ModelCatalog` offers every provider by name, sharing one HTTP client and process pool:

```python
from sagent.providers.catalog import ModelCatalog

async with ModelCatalog() as catalog:
    model = catalog["google"]["gemini-pro-3.1"]
    utility = catalog["google"]["utility"]
```

A provider also works alone. API-key providers read their key from the environment when a request is made:

```python
from sagent.providers.google.api import Google

async with Google() as google:
    model = google["gemini-pro-3.1"]
```

Looking a model up does no I/O. An unknown model ID raises `UnknownModelError`.

Catalog keys are exact names of the form `family-major.minor`, such as
`sol-6.1`, `luna-6.0`, and `opus-5.5`. A prefix names the newest model it
starts: `sol` and `sol-6` both mean `sol-6.1`, and `opus-4` means `opus-4.8`.
The roles `default`, `utility`, and `best` resolve the same way. Vendor wire
IDs remain accepted as compatibility aliases. A bare model ID selects its
largest context window. Append `+272k` for a smaller
OpenAI window or `+200k` for a smaller Anthropic window when the model offers
one. An explicit `+1m` is also accepted when the bare model already provides
that window. Both OpenAI providers use the Responses API. Supported reasoning
efforts vary by model.

API-key requests send output-token limits and temperature for non-reasoning
models. The subscription backend omits those unsupported fields. Both paths
replay local conversation history with `store: false`, preserving encrypted
reasoning and image-bearing tool results. Existing sessions need no conversion.

## CLI dispatch

```bash
sagent --provider google --model gemini-pro-3.1
```

`--provider` takes the name `ModelCatalog` offers the provider under, in any case and with or without separators (`Google`, `openai_sub`), or an alias (`gapi`, `acli`).

If no provider answers to the name, Sagent reports an error. Keep API keys in
the environment so secrets do not land in shell history.

## Provider inference

`ModelCatalog.model(name, current=...)` finds the provider that serves a model
name: it asks the current provider first, then each other provider in the order
offered. Short IDs such as `opus-4.8` and `luna-6.0`, and vendor wire IDs such
as `claude-sonnet-4-6`, switch providers without a separate provider argument.
`PROVIDER/MODEL` asks only that provider. `SelfHosted` answers to any name with
a `/`, a Hugging Face repo or a local path; a repo whose org names a provider
is `selfhosted/ORG/NAME`.

`AgentSelf(model=...)` and `--model` resolve names this way.

## Context-window tags

Anthropic model IDs may include window tags such as:

```bash
sagent --provider Anthropic --model sonnet-4.6+200k
sagent --provider Anthropic --model opus-4.7+1m
```

The provider strips the tag for API calls and uses it to select the request-token budget.

## OpenAI-compatible endpoints

Subclass `OpenAICompatible` (`sagent.providers.openai.compat`) for endpoints that implement OpenAI chat completions.

Set the subclass's `package` and `section` (where its model catalog lives), `url`, and `key_variable`. Override `encode` for whatever its server reads differently.

```python
from sagent.providers import moonshot
from sagent.providers.openai.compat import OpenAICompatible


class Kimi(OpenAICompatible):
    package, section = moonshot, "moonshot"
    url = "https://api.moonshot.ai/v1"
    key_variable = "MOONSHOT_API_KEY"
    reasoning_field = "reasoning_content"


async with Kimi() as kimi:
    model = kimi["default"]
```

`Kimi(api_key=..., base_url=...)` overrides the key and the class `url` at construction time.

## Self-hosted HuggingFace models

Use `SelfHosted` for HuggingFace causal LMs loaded through `transformers`:

```bash
uv sync --extra selfhosted
# Or: pip install "sagent[selfhosted]"
hf download Qwen/Qwen3.6-27B --local-dir /opt/models/qwen3.6-27b
sagent --provider SelfHosted
sagent --provider SelfHosted --model /opt/models/qwen3.6-27b+bfloat16+cuda
sagent --provider SelfHosted --model Qwen/Qwen3-0.6B+float16+cuda \
  --effort none --max-tool-call-rounds 1
```

Python API:

```python
from sagent.providers.selfhosted.server import SelfHosted

async with SelfHosted() as local:
    model = local["Qwen/Qwen3.6-27B+bfloat16+cuda"]
```

Pass a local snapshot path as the model name or `--model` when you want to use an
already-populated cache. SelfHosted options such as `+cuda`, `+bfloat16`, and
`+compile` can appear in any order. Cloud Qwen IDs continue to infer
`DashScope`; select `SelfHosted` explicitly for local model paths.

Examples of frontier open-weight HuggingFace repos to evaluate:

| Model | Repo ID |
| --- | --- |
| DeepSeek V4 Flash | `deepseek-ai/DeepSeek-V4-Flash` |
| DeepSeek V4 Pro | `deepseek-ai/DeepSeek-V4-Pro` |
| Qwen 3.6 35B-A3B | `Qwen/Qwen3.6-35B-A3B` |
| Qwen 3.6 27B | `Qwen/Qwen3.6-27B` |
| Kimi K2 Thinking | `moonshotai/Kimi-K2-Thinking` |
| GLM 4.6 | `zai-org/GLM-4.6` |
| Gemma 4 31B IT | `google/gemma-4-31B-it` |

Check each model card for license, hardware, quantization, chat-template, and
`trust_remote_code` requirements. Some newly released architectures require the
latest `transformers` build or a serving runtime such as vLLM or SGLang before
they work with `AutoModelForCausalLM`; multimodal or custom-code models may need
additional runtime support.

The `selfhosted` extra tracks the released HuggingFace runtime stack needed by
these examples: `transformers`, `accelerate`, `torchvision`,
`compressed-tensors`, `sentencepiece`, `protobuf`, `safetensors`, and `torch`.
Kimi checkpoints are tagged as custom-code models, so they still require an
explicit `trust_remote_code=True` load.

## Model contract

A model exposes:

- `capability`: every setting it offers, its limits, and its prices.
- `settings`: the settings chosen; assign to change them.
- `count(request)`: tokens a request would send, exact where the vendor says.
- `stream(request)`: yields `TextChunk`s as the reply is written, then the finished `Response`.

A transient failure raises a `RetryableError` subclass such as `RateLimitedError`; others raise errors such as `AuthError` and `ContextOverflowError`. `Response` holds the reply's records, token counts, spend, and stop reason.

## Pricing and costs

Provider model profiles include token limits and per-million-token pricing. Sagent records request, response, cache-write, and cache-read tokens for every response. `Agent.spend_usd` and CLI `--max-budget-usd` use these costs.
