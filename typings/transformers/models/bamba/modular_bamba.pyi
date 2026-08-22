from typing import Any, TypedDict

from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from torch import nn
from transformers.models.jamba.modeling_jamba import (
    HybridMambaAttentionDynamicCache,
    JambaAttentionDecoderLayer,
)
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from transformers.models.mamba2.modeling_mamba2 import MambaRMSNormGated

import torch

from .configuration_bamba import BambaConfig
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import auto_docstring, can_return_tuple
from ...utils.deprecation import deprecate_kwarg
from ...utils.import_utils import is_causal_conv1d_available, is_mamba_2_ssm_available

"""PyTorch Bamba model."""
if is_mamba_2_ssm_available(): ...
else:
    selective_state_update = ...
if is_causal_conv1d_available(): ...
else: ...
is_fast_path_available = ...
logger = ...

class BambaFlashAttentionKwargs(TypedDict, total=False):
    cu_seq_lens_q: torch.LongTensor
    cu_seq_lens_k: torch.LongTensor
    max_length_q: int
    max_length_k: int
    seq_idx: torch.IntTensor

class HybridMambaAttentionDynamicCache(HybridMambaAttentionDynamicCache):
    def __init__(
        self, config: BambaConfig, batch_size, dtype=..., device=...
    ) -> None: ...

class BambaRotaryEmbedding(LlamaRotaryEmbedding): ...

def apply_rotary_pos_emb(
    q, k, cos, sin, position_ids=..., unsqueeze_dim=...
):  # -> tuple[Tensor, Tensor]:
    ...

class BambaAttention(LlamaAttention): ...
class BambaRMSNormGated(MambaRMSNormGated): ...

def apply_mask_to_padding_states(hidden_states, attention_mask): ...

class BambaMixer(nn.Module):
    def __init__(self, config: BambaConfig, layer_idx: int) -> None: ...
    def cuda_kernels_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: HybridMambaAttentionDynamicCache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        seq_idx: torch.IntTensor | None = ...,
    ):  # -> Any:
        ...
    def torch_forward(
        self,
        input_states,
        cache_params: HybridMambaAttentionDynamicCache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
    ):  # -> Any:
        ...
    def forward(
        self,
        hidden_states,
        cache_params: HybridMambaAttentionDynamicCache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        seq_idx: torch.IntTensor | None = ...,
        **kwargs,
    ):  # -> Any:
        ...

class BambaMLP(LlamaMLP): ...
class BambaRMSNorm(LlamaRMSNorm): ...

class BambaDecoderLayer(JambaAttentionDecoderLayer):
    def __init__(
        self, config: BambaConfig, layer_idx: int, layer_type: str = ...
    ) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: HybridMambaAttentionDynamicCache | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
        **kwargs: Unpack[BambaFlashAttentionKwargs],
    ) -> tuple[
        torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None
    ]: ...

@auto_docstring
class BambaPreTrainedModel(PreTrainedModel):
    config: BambaConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _is_stateful = ...

@auto_docstring
class BambaModel(BambaPreTrainedModel):
    def __init__(self, config: BambaConfig) -> None: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: HybridMambaAttentionDynamicCache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[BambaFlashAttentionKwargs],
    ) -> BaseModelOutputWithPast: ...

class BambaForCausalLM(LlamaForCausalLM):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: HybridMambaAttentionDynamicCache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs,
    ) -> CausalLMOutputWithPast: ...
    def __call__(self, *args: Any, **kwargs: Any) -> CausalLMOutputWithPast: ...
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

__all__ = ["BambaForCausalLM", "BambaModel", "BambaPreTrainedModel"]
