from dataclasses import dataclass

from torch import nn

import torch

from .configuration_mimi import MimiConfig
from ...cache_utils import Cache
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_rope_utils import dynamic_rope_update
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring
from ...utils.deprecation import deprecate_kwarg

"""PyTorch Mimi model."""
logger = ...

@dataclass
@auto_docstring
class MimiOutput(ModelOutput):
    audio_codes: torch.LongTensor | None = ...
    audio_values: torch.FloatTensor | None = ...
    encoder_past_key_values: Cache | list[torch.FloatTensor] | None = ...
    decoder_past_key_values: Cache | list[torch.FloatTensor] | None = ...

class MimiConv1dPaddingCache:
    def __init__(
        self,
        num_layers: int,
        per_layer_padding: list[int],
        per_layer_padding_mode: list[str],
        per_layer_in_channels: list[int],
    ) -> None: ...
    def update(self, hidden_states: torch.Tensor, layer_idx: int):  # -> Tensor | None:
        ...

@dataclass
@auto_docstring
class MimiEncoderOutput(ModelOutput):
    audio_codes: torch.LongTensor | None = ...
    encoder_past_key_values: Cache | list[torch.FloatTensor] | None = ...
    padding_cache: MimiConv1dPaddingCache | None = ...

@dataclass
@auto_docstring
class MimiDecoderOutput(ModelOutput):
    audio_values: torch.FloatTensor | None = ...
    decoder_past_key_values: Cache | list[torch.FloatTensor] | None = ...

class MimiConv1d(nn.Module):
    def __init__(
        self,
        config,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = ...,
        dilation: int = ...,
        groups: int = ...,
        pad_mode: str | None = ...,
        bias: bool = ...,
        layer_idx: int | None = ...,
    ) -> None: ...
    def apply_weight_norm(self):  # -> None:
        ...
    def remove_weight_norm(self):  # -> None:
        ...
    def forward(self, hidden_states, padding_cache=...):  # -> Any:
        ...

class MimiConvTranspose1d(nn.Module):
    def __init__(
        self,
        config,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = ...,
        groups: int = ...,
        bias=...,
    ) -> None: ...
    def apply_weight_norm(self):  # -> None:
        ...
    def remove_weight_norm(self):  # -> None:
        ...
    def forward(self, hidden_states):  # -> Any:
        ...

class MimiResnetBlock(nn.Module):
    def __init__(self, config: MimiConfig, dim: int, dilations: list[int]) -> None: ...
    def forward(self, hidden_states, padding_cache=...):  # -> Any:
        ...

class MimiEncoder(nn.Module):
    def __init__(self, config: MimiConfig) -> None: ...
    def forward(self, hidden_states, padding_cache=...):  # -> Any:
        ...

class MimiLayerScale(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, x: torch.Tensor):  # -> Tensor:
        ...

class MimiRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, config: MimiConfig, device=...) -> None: ...
    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):  # -> tuple[Tensor, Tensor]:
        ...

def rotate_half(x):  # -> Tensor:
    ...
def apply_rotary_pos_emb(
    q, k, cos, sin, position_ids=..., unsqueeze_dim=...
):  # -> tuple[Any, Any]:
    ...

class MimiMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor: ...

class MimiAttention(nn.Module):
    def __init__(self, config: MimiConfig, layer_idx: int | None = ...) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool = ...,
        use_cache: bool = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class MimiFlashAttention2(MimiAttention):
    def __init__(self, *args, **kwargs) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool = ...,
        use_cache: bool = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class MimiSdpaAttention(MimiAttention):
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool = ...,
        use_cache: bool = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

MIMI_ATTENTION_CLASSES = ...

class MimiTransformerLayer(GradientCheckpointingLayer):
    def __init__(self, config: MimiConfig, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs,
    ) -> tuple[
        torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None
    ]: ...

class MimiTransformerModel(nn.Module):
    def __init__(self, config: MimiConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | list[torch.FloatTensor] | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple | BaseModelOutputWithPast: ...

class MimiDecoder(nn.Module):
    def __init__(self, config: MimiConfig) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class MimiEuclideanCodebook(nn.Module):
    def __init__(self, config: MimiConfig, epsilon: float = ...) -> None: ...
    @property
    def embed(self) -> torch.Tensor: ...
    def quantize(self, hidden_states):  # -> Tensor:
        ...
    def encode(self, hidden_states):  # -> Tensor:
        ...
    def decode(self, embed_ind):  # -> Tensor:
        ...

class MimiVectorQuantization(nn.Module):
    def __init__(self, config: MimiConfig) -> None: ...
    def encode(self, hidden_states):  # -> Tensor:
        ...
    def decode(self, embed_ind):  # -> Tensor:
        ...

class MimiResidualVectorQuantizer(nn.Module):
    def __init__(
        self, config: MimiConfig, num_quantizers: int | None = ...
    ) -> None: ...
    def encode(
        self, embeddings: torch.Tensor, num_quantizers: int | None = ...
    ) -> torch.Tensor: ...
    def decode(self, codes: torch.Tensor) -> torch.Tensor: ...

class MimiSplitResidualVectorQuantizer(nn.Module):
    def __init__(self, config: MimiConfig) -> None: ...
    def encode(
        self, embeddings: torch.Tensor, num_quantizers: float | None = ...
    ) -> torch.Tensor: ...
    def decode(self, codes: torch.Tensor) -> torch.Tensor: ...

@auto_docstring
class MimiPreTrainedModel(PreTrainedModel):
    config: MimiConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _can_compile_fullgraph = ...

@auto_docstring(
    custom_intro="""
    The Mimi neural audio codec model.
    """
)
class MimiModel(MimiPreTrainedModel):
    def __init__(self, config: MimiConfig) -> None: ...
    def get_encoder(self):  # -> MimiEncoder:
        ...
    def get_encoded_length(
        self, input_length: torch.LongTensor
    ) -> torch.LongTensor: ...
    def get_audio_codes_mask(
        self, padding_mask: torch.Tensor, padding_side: str = ...
    ): ...
    def encode(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor | None = ...,
        num_quantizers: float | None = ...,
        encoder_past_key_values: Cache | list[torch.FloatTensor] | None = ...,
        padding_cache: MimiConv1dPaddingCache | None = ...,
        use_streaming: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | MimiEncoderOutput: ...
    def decode(
        self,
        audio_codes: torch.Tensor,
        padding_mask: torch.Tensor | None = ...,
        decoder_past_key_values: Cache | list[torch.FloatTensor] | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor] | MimiDecoderOutput: ...
    @auto_docstring
    def forward(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor | None = ...,
        num_quantizers: int | None = ...,
        audio_codes: torch.Tensor | None = ...,
        encoder_past_key_values: Cache | list[torch.FloatTensor] | None = ...,
        decoder_past_key_values: Cache | list[torch.FloatTensor] | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor] | MimiOutput: ...

__all__ = ["MimiModel", "MimiPreTrainedModel"]
