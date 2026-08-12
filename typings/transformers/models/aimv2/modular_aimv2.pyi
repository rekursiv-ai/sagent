from typing import Any
from torch import nn

import torch

from ..clip.modeling_clip import CLIPModel, CLIPTextEmbeddings
from ..llama.modeling_llama import LlamaMLP, LlamaRMSNorm
from ..siglip.configuration_siglip import (
    SiglipConfig,
    SiglipTextConfig,
    SiglipVisionConfig,
)
from ..siglip.modeling_siglip import SiglipAttention, SiglipEncoder, SiglipOutput
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import check_model_inputs

"""Pytorch implementation of AIMv2 Model"""

class Aimv2VisionConfig(SiglipVisionConfig):
    def __init__(
        self,
        hidden_size: int = ...,
        intermediate_size: int = ...,
        num_hidden_layers: int = ...,
        num_attention_heads: int = ...,
        num_channels: int = ...,
        image_size: int = ...,
        patch_size: int = ...,
        rms_norm_eps: float = ...,
        attention_dropout: float = ...,
        qkv_bias: bool = ...,
        mlp_bias: bool = ...,
        hidden_act: str = ...,
        initializer_range: float = ...,
        use_head: bool = ...,
        is_native: bool = ...,
        **kwargs,
    ) -> None: ...

class Aimv2TextConfig(SiglipTextConfig):
    def __init__(
        self,
        vocab_size: int = ...,
        hidden_size: int = ...,
        intermediate_size: int = ...,
        num_hidden_layers: int = ...,
        num_attention_heads: int = ...,
        rms_norm_eps: float = ...,
        attention_dropout: float = ...,
        qkv_bias: bool = ...,
        mlp_bias: bool = ...,
        hidden_act: str = ...,
        pad_token_id: int | None = ...,
        bos_token_id: int | None = ...,
        eos_token_id: int = ...,
        max_position_embeddings: int = ...,
        initializer_range: bool = ...,
        **kwargs,
    ) -> None: ...

class Aimv2Config(SiglipConfig):
    def __init__(
        self,
        text_config=...,
        vision_config=...,
        projection_dim=...,
        logit_scale_init_value=...,
        **kwargs,
    ) -> None: ...

class Aimv2Output(SiglipOutput): ...
class Aimv2RMSNorm(LlamaRMSNorm): ...
class Aimv2MLP(LlamaMLP): ...

class Aimv2VisionEmbeddings(nn.Module):
    def __init__(self, config: Aimv2VisionConfig) -> None: ...
    @staticmethod
    def build_2d_sincos_position_embedding(
        height, width, embed_dim=..., temperature=..., device=..., dtype=...
    ) -> torch.Tensor: ...
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Aimv2TextEmbeddings(CLIPTextEmbeddings): ...

class Aimv2Attention(SiglipAttention):
    def __init__(self, config) -> None: ...

class Aimv2EncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Aimv2VisionConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class Aimv2Encoder(SiglipEncoder): ...

class Aimv2AttentionPoolingHead(nn.Module):
    def __init__(self, config: Aimv2VisionConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

@auto_docstring
class Aimv2PreTrainedModel(PreTrainedModel):
    config: Aimv2Config
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _supports_sdpa = ...
    _supports_flash_attn = ...
    _supports_flex_attn = ...

@auto_docstring(custom_intro=...)
class Aimv2VisionModel(Aimv2PreTrainedModel):
    config: Aimv2VisionConfig
    main_input_name = ...
    _can_record_outputs = ...
    def __init__(self, config: Aimv2VisionConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    @deprecate_kwarg("attention_mask", version="v4.58.0")
    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        pixel_values,
        attention_mask: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling: ...

@auto_docstring(custom_intro=...)
class Aimv2TextModel(Aimv2PreTrainedModel):
    main_input_name = ...
    _can_record_outputs = ...
    def __init__(self, config: Aimv2TextConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids,
        attention_mask: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling: ...

@auto_docstring
class Aimv2Model(CLIPModel):
    _supports_flash_attn = ...
    def __init__(self, config: Aimv2Config) -> None: ...
    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Aimv2Output: ...

__all__ = [
    "Aimv2Config",
    "Aimv2Model",
    "Aimv2PreTrainedModel",
    "Aimv2TextConfig",
    "Aimv2TextModel",
    "Aimv2VisionConfig",
    "Aimv2VisionModel",
]
