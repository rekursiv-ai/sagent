import torch

from ..deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Attention,
    DeepseekV3ForCausalLM,
    DeepseekV3MLP,
    DeepseekV3Model,
    DeepseekV3MoE,
    DeepseekV3PreTrainedModel,
    DeepseekV3RMSNorm,
    DeepseekV3RotaryEmbedding,
    DeepseekV3TopkRouter,
)
from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...processing_utils import Unpack
from ...utils import TransformersKwargs

logger = ...

class LongcatFlashRMSNorm(DeepseekV3RMSNorm): ...
class LongcatFlashRotaryEmbedding(DeepseekV3RotaryEmbedding): ...

class LongcatFlashMLP(DeepseekV3MLP):
    def __init__(self, config, hidden_size=..., intermediate_size=...) -> None: ...

class LongcatFlashTopkRouter(DeepseekV3TopkRouter):
    def __init__(self, config) -> None: ...
    @torch.no_grad()
    def get_topk_indices(self, scores):  # -> Tensor:
        ...
    def forward(self, hidden_states):  # -> tuple[Tensor, Any]:
        ...

class LongcatFlashMoE(DeepseekV3MoE):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Tensor:
        ...

class LongcatFlashMLA(DeepseekV3Attention):
    def __init__(self, config, layer_idx: int) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class LongcatFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx: int) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.Tensor: ...

class LongcatFlashPreTrainedModel(DeepseekV3PreTrainedModel):
    _can_record_outputs = ...

class LongcatFlashModel(DeepseekV3Model):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        cache_position: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ):  # -> BaseModelOutputWithPast:
        ...

class LongcatFlashForCausalLM(DeepseekV3ForCausalLM):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config) -> None: ...

__all__ = [
    "LongcatFlashForCausalLM",
    "LongcatFlashModel",
    "LongcatFlashPreTrainedModel",
]
