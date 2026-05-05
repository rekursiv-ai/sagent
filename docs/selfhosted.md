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

provider = SelfHosted.from_hf("Qwen/Qwen3.6-27B")
model: SelfHostedModel = provider.model()
agent = Agent(
    model=model,
    system="Answer concisely.",
    tools=[],
)
result = await agent.run(json_freeze({"prompt": "Say hi."}))
print(result.content)
```

Run it from the CLI:

```bash
sagent --provider SelfHosted --model Qwen/Qwen3.6-27B
```

For small local Qwen smoke tests, disable thinking and tools so the model can
spend its short output budget on the answer:

```bash
SAGENT_SELFHOSTED_DTYPE=float16 \
  sagent --provider SelfHosted --model Qwen/Qwen3-0.6B \
  --tools none --effort none --max-response-tokens 32 --max-tool-call-rounds 1
```

Add `--log-level DEBUG` to see load timing, selected device, prompt token
counts, generation timing, and ignored malformed or unadvertised tool calls.

For repeated use, configure the model once. The value can be either a
HuggingFace repo ID or a local snapshot path:

```bash
export SAGENT_SELFHOSTED_MODEL=Qwen/Qwen3.6-27B
export SAGENT_SELFHOSTED_DEVICE=cuda
export SAGENT_SELFHOSTED_DTYPE=bfloat16
export SAGENT_SELFHOSTED_COMPILE=0
sagent --provider SelfHosted
```

`SAGENT_SELFHOSTED_DEVICE`, `SAGENT_SELFHOSTED_DTYPE`, and
`SAGENT_SELFHOSTED_COMPILE` are optional. When no device is set, SelfHosted uses
MPS if available, then CUDA if available, then the PyTorch CPU default.
Supported dtype values are `float16`, `bfloat16`, and `float32`. Compile is off
by default; set `SAGENT_SELFHOSTED_COMPILE=1` to wrap the loaded model with
`torch.compile`. Keep it opt-in because compile can add a large first-request
cost and may vary by device/backend.

To force a local cache path, pass it in the same places:

```bash
sagent --provider SelfHosted --model /opt/models/qwen3.6-27b
export SAGENT_SELFHOSTED_MODEL=/opt/models/qwen3.6-27b
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
