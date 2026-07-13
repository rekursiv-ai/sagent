from typing import Any

from torch import nn

import torch

from .configuration_zamba import ZambaConfig
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import auto_docstring
from ...utils.deprecation import deprecate_kwarg
from ...utils.import_utils import is_causal_conv1d_available, is_mamba_ssm_available

"""PyTorch Zamba model."""
if is_mamba_ssm_available(): ...
else: ...
if is_causal_conv1d_available(): ...
else: ...
is_fast_path_available = ...
logger = ...

class ZambaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, hidden_states): ...
    def extra_repr(self):  # -> str:
        ...

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor: ...

class ZambaHybridDynamicCache:
    is_compileable = ...
    def __init__(self, config, batch_size, dtype=..., device=...) -> None: ...
    def __len__(self):  # -> int:
        ...
    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]: ...
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

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = ...,
    **kwargs,
):  # -> tuple[Tensor, Tensor]:
    ...

class ZambaAttention(nn.Module):
    def __init__(self, config: ZambaConfig, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        attention_mask: torch.Tensor | None,
        past_key_values: ZambaHybridDynamicCache | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class ZambaMambaMixer(nn.Module):
    def __init__(self, config: ZambaConfig, layer_idx) -> None: ...
    def cuda_kernels_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: ZambaHybridDynamicCache = ...,
        attention_mask=...,
    ):  # -> Any:
        ...
    def slow_forward(
        self,
        input_states,
        cache_params: ZambaHybridDynamicCache = ...,
        attention_mask=...,
    ):  # -> Any:
        ...
    def forward(
        self,
        hidden_states,
        cache_params: ZambaHybridDynamicCache = ...,
        attention_mask=...,
    ):  # -> Any:
        ...

class ZambaMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class ZambaAttentionDecoderLayer(nn.Module):
    def __init__(self, config: ZambaConfig, layer_idx: int | None = ...) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        original_hidden_states: torch.Tensor,
        layer_idx: int,
        attention_mask: torch.Tensor | None = ...,
        past_key_values: ZambaHybridDynamicCache | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[
        torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None
    ]: ...

class ZambaMambaDecoderLayer(nn.Module):
    def __init__(self, config: ZambaConfig, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        original_hidden_states: torch.Tensor | None = ...,
        layer_idx: int | None = ...,
        attention_mask: torch.Tensor | None = ...,
        causal_mask: torch.Tensor | None = ...,
        past_key_values: ZambaHybridDynamicCache | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        transformer_hidden_states: torch.Tensor | None = ...,
        **kwargs,
    ) -> tuple[
        torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None
    ]: ...

class ZambaHybridLayer(nn.Module):
    def __init__(
        self,
        shared_transf: ZambaAttentionDecoderLayer,
        linear: nn.Linear,
        mamba: ZambaMambaDecoderLayer,
    ) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        original_hidden_states: torch.Tensor | None = ...,
        layer_idx: int | None = ...,
        attention_mask: torch.Tensor | None = ...,
        causal_mask: torch.Tensor | None = ...,
        past_key_values: ZambaHybridDynamicCache | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple[
        torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None
    ]: ...

@auto_docstring
class ZambaPreTrainedModel(PreTrainedModel):
    config: ZambaConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _is_stateful = ...

@auto_docstring
class ZambaModel(ZambaPreTrainedModel):
    def __init__(self, config: ZambaConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: ZambaHybridDynamicCache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple | BaseModelOutputWithPast: ...

class ZambaForCausalLM(ZambaPreTrainedModel, GenerationMixin):
    def __init__(self, config: ZambaConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: ZambaHybridDynamicCache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
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

@auto_docstring(custom_intro=...)
class ZambaForSequenceClassification(ZambaPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | list[torch.FloatTensor] | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | SequenceClassifierOutputWithPast: ...

__all__ = [
    "ZambaForCausalLM",
    "ZambaForSequenceClassification",
    "ZambaModel",
    "ZambaPreTrainedModel",
]
