# Providers

Sagent separates providers from models. A provider owns authentication and creates model objects. A model sends typed requests and returns typed responses.

## Public provider matrix

| Provider class | Environment variable | Default model | Utility model | Notes |
| --- | --- | --- | --- | --- |
| `Anthropic` | `ANTHROPIC_API_KEY` | `opus-5.5` | `sonnet-5.5` | Anthropic API-key provider; `best` resolves to `fable-5.1`. |
| `OpenAI` | `OPENAI_API_KEY` | `sol-6.1` | `luna-6.0` | OpenAI API provider. |
| `Google` | `GOOGLE_API_KEY` | `gemini-pro-3.1` | `gemini-flash-lite-3.5` | Google Gemini provider. |
| `Moonshot` | `MOONSHOT_API_KEY` | `kimi-3.0` | `kimi-2.6` | OpenAI-compatible Kimi provider. |
| `DashScope` | `DASHSCOPE_API_KEY` | `qwen-plus-3.7` | `qwen-flash-3.8` | Alibaba DashScope provider. |
| `MiniMax` | `MINIMAX_API_KEY` | `minimax-3.0` | `minimax-3.0` | MiniMax provider. |
| `SelfHosted` | none | `Qwen/Qwen3.6-27B` | configured snapshot | Local HF transformers provider. |
| `OpenAICompat` | subclass-defined | subclass-defined | subclass-defined | Base class for chat-completions-compatible APIs. |

The public package is designed around API-key providers.

## Basic usage

```python
from sagent.providers import Google

provider = Google.from_env()
model = provider.model("gemini-pro-3.1")
utility = provider.model("utility")
```

All public API-key providers support:

```python
ProviderClass.from_env()
ProviderClass.from_key("...")
ProviderClass.from_env().model("model-id")
ProviderClass.from_env().model("utility")
```

`model(None)` uses the provider's default model. Unknown model IDs raise with the provider's known model list.

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
sagent --provider Google --auth env --model gemini-pro-3.1
```

`--provider` is the provider class name from `sagent.providers`. `--auth env` calls `Google.from_env()`.

If the named factory does not exist, Sagent reports an error. Use `--auth env`
for API keys so secrets do not land in shell history.

## Provider inference

Agent tools first infer a provider from catalog membership, so short IDs such
as `opus-4.8` and `luna-6.0` can switch providers without a separate provider
argument. Vendor prefixes remain as compatibility fallbacks:

| Prefix | Provider |
| --- | --- |
| `claude` | `Anthropic` |
| `gemini` | `Google` |
| `gpt`, `chatgpt`, `o1`, `o3`, `o4`, `codex` | `OpenAI` |
| `kimi`, `moonshot` | `Moonshot` |
| `qwen` | `DashScope` |
| `minimax` | `MiniMax` |
| `/`, `./`, `../`, `~/` | `SelfHosted` |

This is used by model-switching tools so callers can usually pass just `model_id`.

## Context-window tags

Anthropic model IDs may include window tags such as:

```bash
sagent --provider Anthropic --model sonnet-4.6+200k
sagent --provider Anthropic --model opus-4.7+1m
```

The provider strips the tag for API calls and uses it to select the request-token budget.

## OpenAI-compatible endpoints

Subclass `OpenAICompat` for endpoints that implement OpenAI chat completions.

Set the subclass's `ENV_VAR`, `BASE_URL`, and `catalog`. The catalog must map
the `default` and `utility` roles to capability rows alongside its model IDs.

Use it like any other provider:

```python
model = LocalProvider.from_env().model("local-model")
```

`OpenAICompat.from_env(base_url=...)` and `from_key(api_key, base_url=...)` can override the class `BASE_URL` at construction time.

See `examples/openai_compatible_provider.py` for a runnable version.

### Hosted gateway example: A2Agent

`examples/a2agent_provider.py` defines an `A2Agent` subclass of `OpenAICompat`
and sends one prompt using the existing transport. It is a standalone Python
example, not a new built-in `sagent --provider` option or an endorsement.

The [API reference](https://a2agent.me/llms.txt) documents the base URL and
Bearer authentication. Supply a private key through `A2AGENT_API_KEY` and an
exact model ID available to that key. To inspect the model list:

```bash
curl --fail-with-body https://api.a2agent.me/v1/models \
  -H "Authorization: Bearer $A2AGENT_API_KEY"
```

Model discovery alone does not establish context limits, prices, or tool
support. Confirm input/output limits for the chosen route and all five token
rates for the key's group before running the example. The following variables
must be set to those confirmed values; prices are USD per million tokens:

```bash
uv run python examples/a2agent_provider.py \
  --model "$A2AGENT_MODEL" \
  --input-limit "$A2AGENT_INPUT_LIMIT" \
  --output-limit "$A2AGENT_OUTPUT_LIMIT" \
  --input-price "$A2AGENT_INPUT_PRICE" \
  --output-price "$A2AGENT_OUTPUT_PRICE" \
  --cache-read-price "$A2AGENT_CACHE_READ_PRICE" \
  --cache-write-price "$A2AGENT_CACHE_WRITE_PRICE" \
  --cache-write-1h-price "$A2AGENT_CACHE_WRITE_1H_PRICE"
```

Do not copy another vendor's rates or use zero for unknown prices. Explicit
zero rates are accepted only for meters confirmed to be unbilled. The example
has no baked-in model list, default model choice, or assumed pricing. Its
`default` and `utility` roles both resolve to the one model you configure;
separate calls to `create_provider` keep their catalogs independent.

`--base-url` overrides the complete API base (default
`https://api.a2agent.me/v1`); only use endpoints you trust with the key. The
transport appends `/chat/completions`. `buffer()` still uses SSE internally,
so this example requires streaming support even though it prints a final
response. Model/route-specific reasoning controls and embeddings are outside
this example's scope. Check tool calling separately before attaching agent
tools.

The accompanying tests use mock HTTP responses and dummy keys, not live
A2Agent inference. Test with your own key before relying on the integration.
Requests send prompt content to a hosted service and may incur charges;
keep keys out of source files, public CI, logs, and issue reports.

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
from sagent.providers import SelfHosted, SelfHostedModel

provider = SelfHosted.from_key("Qwen/Qwen3.6-27B+bfloat16+cuda")
model: SelfHostedModel = provider.model()
```

Pass a local snapshot path to `from_key` or `--model` when you want to use an
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

- `buffer(request)`: send one request and return a complete response.
- `stream(request, on_text=None)`: stream text chunks through `on_text` while returning a complete response.
- `is_context_overflow(error)`: classify provider context-window errors.

`ModelResponse` includes content, stop reason, token counts, response identifiers, cache counts, and cost fields.

## Pricing and costs

Provider model profiles include token limits and per-million-token pricing. Sagent records request, response, cache-write, and cache-read tokens for every response. `Agent.total_cost_usd` and CLI `--max-budget-usd` use these costs.
