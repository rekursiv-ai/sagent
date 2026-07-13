from torch import nn

import torch

from ..mixtral.configuration_mixtral import MixtralConfig
from ..mixtral.modeling_mixtral import (
    MixtralAttention,
    MixtralDecoderLayer,
    MixtralForCausalLM,
    MixtralForQuestionAnswering,
    MixtralForSequenceClassification,
    MixtralForTokenClassification,
    MixtralModel,
    MixtralPreTrainedModel,
    MixtralRMSNorm,
    MixtralSparseMoeBlock,
)
from ...cache_utils import Cache, DynamicCache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import MoeModelOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs
from ...utils.deprecation import deprecate_kwarg

"""PyTorch MiniMax model."""
logger = ...

class MiniMaxConfig(MixtralConfig):
    def __init__(
        self,
        layer_types=...,
        block_size=...,
        full_attn_alpha_factor=...,
        full_attn_beta_factor=...,
        linear_attn_alpha_factor=...,
        linear_attn_beta_factor=...,
        mlp_alpha_factor=...,
        mlp_beta_factor=...,
        **super_kwargs,
    ) -> None: ...

class MiniMaxRMSNorm(MixtralRMSNorm): ...

class MiniMaxCache(DynamicCache):
    def __init__(self) -> None: ...
    def set_linear_cache(self, layer_idx, linear_cache):  # -> None:
        ...
    def get_linear_cache(self, layer_idx: int):  # -> Tensor | None:
        ...
    def __len__(self):  # -> int:
        ...
    def __getitem__(self, layer_idx: int):  # -> tuple[Tensor] | tuple[Tensor, Tensor]:
        ...
    def __iter__(
        self,
    ):  # -> Generator[tuple[Tensor] | tuple[Tensor, Tensor], Any, None]:
        ...
    def batch_repeat_interleave(self, repeats: int):  # -> None:
        ...
    def batch_select_indices(self, indices: torch.Tensor):  # -> None:
        ...
    def crop(self, max_length: int): ...

class MiniMaxLightningAttention(nn.Module):
    def __init__(self, config: MiniMaxConfig, layer_idx: int) -> None: ...
    def get_slope_rate(self):  # -> Tensor:
        ...
    def decay_factors(self, slope_rate):  # -> tuple[Tensor, Tensor, Tensor]:
        ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class MiniMaxAttention(MixtralAttention): ...
class MiniMaxSparseMoeBlock(MixtralSparseMoeBlock): ...

class MiniMaxDecoderLayer(MixtralDecoderLayer, GradientCheckpointingLayer):
    def __init__(self, config: MiniMaxConfig, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool | None = ...,
        output_router_logits: bool | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[
        torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None
    ]: ...

class MiniMaxPreTrainedModel(MixtralPreTrainedModel):
    _can_compile_fullgraph = ...
    _can_record_outputs = ...

class MiniMaxModel(MixtralModel):
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: MiniMaxCache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast: ...

class MiniMaxForCausalLM(MixtralForCausalLM):
    def forward(self, **super_kwargs):  # -> MoeCausalLMOutputWithPast:
        ...

class MiniMaxForSequenceClassification(MixtralForSequenceClassification): ...
class MiniMaxForTokenClassification(MixtralForTokenClassification): ...
class MiniMaxForQuestionAnswering(MixtralForQuestionAnswering): ...

__all__ = [
    "MiniMaxConfig",
    "MiniMaxForCausalLM",
    "MiniMaxForQuestionAnswering",
    "MiniMaxForSequenceClassification",
    "MiniMaxForTokenClassification",
    "MiniMaxModel",
    "MiniMaxPreTrainedModel",
]
