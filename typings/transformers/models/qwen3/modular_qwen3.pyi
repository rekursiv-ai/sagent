import torch

from .configuration_qwen3 import Qwen3Config
from ..gemma.modeling_gemma import GemmaMLP
from ..llama.modeling_llama import LlamaAttention
from ..qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer,
    Qwen2ForCausalLM,
    Qwen2ForQuestionAnswering,
    Qwen2ForSequenceClassification,
    Qwen2ForTokenClassification,
    Qwen2Model,
    Qwen2PreTrainedModel,
    Qwen2RMSNorm,
)
from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import CausalLMOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs
from ...utils.deprecation import deprecate_kwarg

"""PyTorch Qwen3 model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...

class Qwen3RMSNorm(Qwen2RMSNorm): ...
class Qwen3MLP(GemmaMLP): ...

class Qwen3Attention(LlamaAttention):
    def __init__(self, config: Qwen3Config, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class Qwen3DecoderLayer(Qwen2DecoderLayer): ...
class Qwen3PreTrainedModel(Qwen2PreTrainedModel): ...
class Qwen3Model(Qwen2Model): ...

class Qwen3ForCausalLM(Qwen2ForCausalLM):
    def forward(
        self, **super_kwargs: Unpack[TransformersKwargs]
    ) -> CausalLMOutputWithPast: ...

class Qwen3ForSequenceClassification(Qwen2ForSequenceClassification): ...
class Qwen3ForTokenClassification(Qwen2ForTokenClassification): ...
class Qwen3ForQuestionAnswering(Qwen2ForQuestionAnswering): ...

__all__ = [
    "Qwen3ForCausalLM",
    "Qwen3ForQuestionAnswering",
    "Qwen3ForSequenceClassification",
    "Qwen3ForTokenClassification",
    "Qwen3Model",
    "Qwen3PreTrainedModel",
]
