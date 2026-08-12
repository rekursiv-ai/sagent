from typing import Any
import torch

from .configuration_bitnet import BitNetConfig
from ..gemma.modeling_gemma import GemmaMLP
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaRMSNorm,
)
from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import CausalLMOutputWithPast
from ...processing_utils import Unpack
from ...utils.deprecation import deprecate_kwarg

"""PyTorch BitNet model."""
logger = ...

class BitNetRMSNorm(LlamaRMSNorm): ...

class BitNetMLP(GemmaMLP):
    def __init__(self, config: BitNetConfig) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class BitNetAttention(LlamaAttention):
    def __init__(self, config: BitNetConfig, layer_idx: int) -> None: ...
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

class BitNetDecoderLayer(LlamaDecoderLayer): ...
class BitNetModel(LlamaModel): ...

class BitNetForCausalLM(LlamaForCausalLM):
    _tied_weights_keys = ...
    _tp_plan = ...
    _pp_plan = ...
    def forward(self, **super_kwargs) -> CausalLMOutputWithPast: ...
    def __call__(self, *args: Any, **kwargs: Any) -> CausalLMOutputWithPast: ...

__all__ = ["BitNetForCausalLM", "BitNetModel", "BitNetPreTrainedModel"]
