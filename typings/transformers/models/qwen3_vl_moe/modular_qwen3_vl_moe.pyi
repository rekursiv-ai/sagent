from torch import nn

import torch

from ..qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeDecoderLayer,
    Qwen3MoePreTrainedModel,
    Qwen3MoeRMSNorm,
)
from ..qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig, Qwen3VLVisionConfig
from ..qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLTextAttention,
    Qwen3VLTextModel,
    Qwen3VLVisionModel,
)
from ...cache_utils import Cache
from ...configuration_utils import PretrainedConfig
from ...processing_utils import Unpack
from ...utils import TransformersKwargs

"""PyTorch Qwen3-VL-MOE model."""
logger = ...

class Qwen3VLMoeTextConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
    keys_to_ignore_at_inference = ...
    base_model_tp_plan = ...
    base_model_pp_plan = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_key_value_heads=...,
        hidden_act=...,
        max_position_embeddings=...,
        initializer_range=...,
        rms_norm_eps=...,
        use_cache=...,
        tie_word_embeddings=...,
        rope_theta=...,
        attention_bias=...,
        attention_dropout=...,
        decoder_sparse_step=...,
        moe_intermediate_size=...,
        num_experts_per_tok=...,
        num_experts=...,
        norm_topk_prob=...,
        router_aux_loss_coef=...,
        mlp_only_layers=...,
        rope_scaling=...,
        head_dim=...,
        **kwargs,
    ) -> None: ...

class Qwen3VLMoeVisionConfig(Qwen3VLVisionConfig): ...

class Qwen3VLMoeConfig(Qwen3VLConfig):
    model_type = ...
    sub_configs = ...

class Qwen3VLMoeTextRMSNorm(Qwen3MoeRMSNorm): ...

class Qwen3VLMoeTextExperts(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        router_indices: torch.Tensor,
    ) -> torch.Tensor: ...

class Qwen3VLMoeTextSparseMoeBlock(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class Qwen3VLMoeTextAttention(Qwen3VLTextAttention): ...
class Qwen3VLMoeTextDecoderLayer(Qwen3MoeDecoderLayer): ...

class Qwen3VLMoePreTrainedModel(Qwen3MoePreTrainedModel):
    config: Qwen3VLMoeConfig
    _no_split_modules = ...

class Qwen3VLMoeVisionModel(Qwen3VLVisionModel): ...
class Qwen3VLMoeTextModel(Qwen3VLTextModel): ...

class Qwen3VLMoeCausalLMOutputWithPast(Qwen3VLCausalLMOutputWithPast):
    aux_loss: torch.FloatTensor | None = ...

class Qwen3VLMoeModel(Qwen3VLModel): ...

class Qwen3VLMoeForConditionalGeneration(Qwen3VLForConditionalGeneration):
    def forward(
        self,
        input_ids: torch.LongTensor = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        pixel_values: torch.Tensor | None = ...,
        pixel_values_videos: torch.FloatTensor | None = ...,
        image_grid_thw: torch.LongTensor | None = ...,
        video_grid_thw: torch.LongTensor | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs: Unpack[TransformersKwargs],
    ):  # -> Qwen3VLMoeCausalLMOutputWithPast:
        ...

__all__ = [
    "Qwen3VLMoeConfig",
    "Qwen3VLMoeForConditionalGeneration",
    "Qwen3VLMoeModel",
    "Qwen3VLMoePreTrainedModel",
    "Qwen3VLMoeTextConfig",
    "Qwen3VLMoeTextModel",
    "Qwen3VLMoeVisionModel",
]
