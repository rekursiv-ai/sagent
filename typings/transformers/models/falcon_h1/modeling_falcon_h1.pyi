from typing import Any

from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from torch import nn

import torch

from .configuration_falcon_h1 import FalconH1Config
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...integrations import use_kernel_forward_from_hub
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_rope_utils import dynamic_rope_update
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.deprecation import deprecate_kwarg
from ...utils.import_utils import is_causal_conv1d_available, is_mamba_2_ssm_available

if is_mamba_2_ssm_available(): ...
else:
    selective_state_update = ...
if is_causal_conv1d_available(): ...
else: ...
logger = ...

class FalconHybridMambaAttentionDynamicCache:
    is_compileable = ...
    def __init__(
        self,
        config: FalconH1Config,
        batch_size: int,
        dtype: torch.dtype = ...,
        devices: list[str] | None = ...,
    ) -> None: ...
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def reorder_cache(self, beam_idx: torch.LongTensor):  # -> None:
        ...
    def get_seq_length(self, layer_idx: int | None = ...) -> int: ...
    def update_conv_state(
        self,
        layer_idx: int,
        new_conv_state: torch.Tensor,
        cache_position: torch.LongTensor,
    ) -> torch.Tensor: ...
    def reset(self):  # -> None:
        ...

class FalconH1RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, config: FalconH1Config, device=...) -> None: ...
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
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor: ...
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = ...,
    **kwargs: Unpack[TransformersKwargs],
):  # -> tuple[Tensor, Tensor]:
    ...

class FalconH1Attention(nn.Module):
    def __init__(self, config: FalconH1Config, layer_idx: int) -> None: ...
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

class FalconH1RMSNormGated(torch.nn.Module):
    def __init__(
        self, hidden_size, eps=..., n_groups=..., norm_before_gate=...
    ) -> None: ...
    def forward(self, hidden_states, gate=...): ...

def pad_tensor_by_size(input_tensor: torch.Tensor, pad_size: int):  # -> Tensor:
    ...
def reshape_into_chunks(input_tensor, pad_size, chunk_size):  # -> Tensor:
    ...
def segment_sum(input_tensor):  # -> Tensor:
    ...

is_fast_path_available = ...

def apply_mask_to_padding_states(hidden_states, attention_mask): ...

class FalconH1Mixer(nn.Module):
    def __init__(self, config: FalconH1Config, layer_idx: int) -> None: ...
    def cuda_kernels_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: FalconHybridMambaAttentionDynamicCache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
    ):  # -> Any:
        ...
    def torch_forward(
        self,
        input_states,
        cache_params: FalconHybridMambaAttentionDynamicCache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
    ):  # -> Any:
        ...
    def forward(
        self,
        hidden_states,
        cache_params: FalconHybridMambaAttentionDynamicCache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
    ):  # -> Any:
        ...

class FalconH1MLP(nn.Module):
    def __init__(self, config: FalconH1Config) -> None: ...
    def forward(self, x):  # -> Any:
        ...

@use_kernel_forward_from_hub("RMSNorm")
class FalconH1RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, hidden_states): ...
    def extra_repr(self):  # -> str:
        ...

class FalconH1DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: FalconH1Config, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        mamba_attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: FalconHybridMambaAttentionDynamicCache | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
        **kwargs,
    ) -> tuple[
        torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None
    ]: ...

@auto_docstring
class FalconH1PreTrainedModel(PreTrainedModel):
    config: FalconH1Config
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _is_stateful = ...

def compute_mup_vector(config):  # -> Tensor:
    ...

@auto_docstring
class FalconH1Model(FalconH1PreTrainedModel):
    def __init__(self, config: FalconH1Config) -> None: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: FalconHybridMambaAttentionDynamicCache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs,
    ) -> tuple | BaseModelOutputWithPast: ...

@auto_docstring
class FalconH1ForCausalLM(FalconH1PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    _tp_plan = ...
    _pp_plan = ...
    def __init__(self, config) -> None: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: FalconHybridMambaAttentionDynamicCache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs,
    ) -> tuple | CausalLMOutputWithPast: ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=...,
        attention_mask=...,
        inputs_embeds=...,
        cache_position=...,
        position_ids=...,
        use_cache=...,
        **kwargs,
    ):  # -> dict[str, Any]:
        ...

__all__ = ["FalconH1ForCausalLM", "FalconH1Model", "FalconH1PreTrainedModel"]
