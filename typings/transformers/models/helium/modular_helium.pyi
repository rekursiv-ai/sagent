from torch import nn

from .configuration_helium import HeliumConfig
from ..gemma.modeling_gemma import (
    GemmaForCausalLM,
    GemmaForSequenceClassification,
    GemmaForTokenClassification,
)
from ..granite.modeling_granite import GraniteAttention
from ..llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRotaryEmbedding,
)

logger = ...

class HeliumRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, hidden_states): ...
    def extra_repr(self):  # -> str:
        ...

class HeliumRotaryEmbedding(LlamaRotaryEmbedding): ...
class HeliumMLP(LlamaMLP): ...

def rotate_half(x):  # -> Tensor:
    ...
def apply_rotary_pos_emb(
    q, k, cos, sin, position_ids=..., unsqueeze_dim=...
):  # -> tuple[Any, Any]:
    ...

class HeliumAttention(GraniteAttention):
    def __init__(self, config: HeliumConfig, layer_idx: int | None = ...) -> None: ...

class HeliumDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: HeliumConfig, layer_idx: int | None = ...) -> None: ...

class HeliumPreTrainedModel(LlamaPreTrainedModel): ...

class HeliumModel(HeliumPreTrainedModel, LlamaModel):
    def __init__(self, config: HeliumConfig) -> None: ...

class HeliumForCausalLM(GemmaForCausalLM): ...
class HeliumForSequenceClassification(GemmaForSequenceClassification): ...
class HeliumForTokenClassification(GemmaForTokenClassification): ...

__all__ = [
    "HeliumForCausalLM",
    "HeliumForSequenceClassification",
    "HeliumForTokenClassification",
    "HeliumModel",
    "HeliumPreTrainedModel",
]
