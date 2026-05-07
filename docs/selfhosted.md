# Self-hosted Models

Sagent can run HuggingFace causal LMs through the `SelfHosted` provider. The
default public model is `Qwen/Qwen3.6-27B`, loaded through `transformers`.

Install the optional local runtime dependencies:

```bash
pip install "sagent[selfhosted]"
```

Use the default Qwen 3.6 model:

```bash
sagent --provider SelfHosted
```

Or download a Qwen 3.6 snapshot and load it from a local cache path:

```bash
hf download Qwen/Qwen3.6-27B --local-dir /opt/models/qwen3.6-27b
```

Run it from Python:

```python
from sagent.agent import Agent
from sagent.lib.json import json_freeze
from sagent.providers import SelfHosted, SelfHostedModel

provider = SelfHosted.from_key("Qwen/Qwen3.6-27B+bfloat16+cuda")
model: SelfHostedModel = provider.model()
agent = Agent(
    model=model,
    system="Answer concisely.",
    tools=[],
)
result = await agent.run(json_freeze({"prompt": "Say hi."}))
print(result.content)
```

Run Qwen 3.6 27B from the CLI:

```bash
sagent --provider SelfHosted --model Qwen/Qwen3.6-27B+bfloat16+cuda
sagent --provider SelfHosted --model Qwen/Qwen3.6-27B+cuda+bfloat16
```

Test Qwen 3 0.6B on macbook:

```bash
# To reduce memory: shorten response tokens, use shorter bare prompt, and load no tools
sagent --provider SelfHosted --model Qwen/Qwen3-0.6B+float16+mps --max-response-tokens 128  --max-tool-call-rounds 1 --tools none
```

Note that on the first turn, the model might need extra warmup time (e.g., run compile).

SelfHosted model specs are the HuggingFace repo ID or local path followed by
`+` options in any order:

```text
MODEL[+DEVICE][+DTYPE][+compile]
```

Supported devices are `cpu`, `cuda`, `mps`, and `auto`. `auto` delegates
placement to `accelerate` through `device_map="auto"` so large models can shard
across visible GPUs. When no device option is provided, SelfHosted uses MPS if
available, then CUDA if available, then the PyTorch CPU default.

Supported dtypes are `float16`, `bfloat16`, and `float32`; `torch.float16`,
`torch.bfloat16`, and `torch.float32` are accepted too. `compile` wraps the
loaded model with `torch.compile`; it is off by default because it can add a
large first-request cost and varies by device/backend.

Each option category may appear at most once. These are invalid:

```bash
# Invalid
sagent --provider SelfHosted --model Qwen/Qwen3.6-27B+bfloat16+float16
sagent --provider SelfHosted --model Qwen/Qwen3.6-27B+cuda+mps
```

To force a local cache path, pass it as the model:

```bash
sagent --provider SelfHosted --model /opt/models/qwen3.6-27b+bfloat16+cuda
```

Cloud Qwen model IDs such as `qwen3.6-plus` still route to DashScope. Use
`SelfHosted` explicitly for local paths and HuggingFace snapshots.

Other frontier open-weight HuggingFace repos to evaluate with the same provider:

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
