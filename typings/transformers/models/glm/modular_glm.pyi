from .configuration_glm import GlmConfig
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
    LlamaForTokenClassification,
)
from ..phi3.modeling_phi3 import Phi3MLP

logger = ...
_CHECKPOINT_FOR_DOC = ...

class GlmMLP(Phi3MLP): ...

def rotate_half(x):  # -> Tensor:
    ...
def apply_rotary_pos_emb(
    q, k, cos, sin, position_ids=..., unsqueeze_dim=...
):  # -> tuple[Tensor, Tensor]:
    ...

class GlmAttention(LlamaAttention):
    def __init__(self, config: GlmConfig, layer_idx: int | None = ...) -> None: ...

class GlmForCausalLM(LlamaForCausalLM): ...
class GlmForSequenceClassification(LlamaForSequenceClassification): ...
class GlmForTokenClassification(LlamaForTokenClassification): ...

__all__ = [
    "GlmForCausalLM",
    "GlmForSequenceClassification",
    "GlmForTokenClassification",
    "GlmModel",
    "GlmPreTrainedModel",
]
