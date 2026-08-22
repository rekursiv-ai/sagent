from typing import Any

from torch import nn

import torch

from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
)
from ..llama4.modeling_llama4 import Llama4TextRotaryEmbedding
from ...cache_utils import Cache
from ...utils.deprecation import deprecate_kwarg

logger = ...

class DeepseekV2Config(LlamaConfig):
    keys_to_ignore_at_inference = ...
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
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        tie_word_embeddings=...,
        rope_theta=...,
        rope_scaling=...,
        attention_bias=...,
        attention_dropout=...,
        mlp_bias=...,
        aux_loss_alpha=...,
        first_k_dense_replace=...,
        kv_lora_rank=...,
        q_lora_rank=...,
        n_group=...,
        n_routed_experts=...,
        n_shared_experts=...,
        qk_nope_head_dim=...,
        qk_rope_head_dim=...,
        routed_scaling_factor=...,
        seq_aux=...,
        topk_group=...,
        topk_method=...,
        v_head_dim=...,
        num_experts_per_tok=...,
        norm_topk_prob=...,
        moe_intermediate_size=...,
        **kwargs,
    ) -> None: ...

def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]: ...

class DeepseekV2MoEGate(nn.Module):
    def __init__(self, config: DeepseekV2Config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class DeepseekV2MoE(nn.Module):
    def __init__(self, config: DeepseekV2Config) -> None: ...
    def moe(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weight: torch.Tensor,
    ) -> torch.Tensor: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class DeepseekV2MLP(LlamaMLP):
    def __init__(
        self, config: DeepseekV2Config, hidden_size=..., intermediate_size=...
    ) -> None: ...

class DeepseekV2RMSNorm(LlamaRMSNorm): ...

class DeepseekV2RotaryEmbedding(Llama4TextRotaryEmbedding):
    def __init__(self, config: DeepseekV2Config, device=...) -> None: ...

class DeepseekV2Attention(nn.Module):
    def __init__(
        self, config: DeepseekV2Config, layer_idx: int | None = ...
    ) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
        position_ids: torch.Tensor | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class DeepseekV2DecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: DeepseekV2Config, layer_idx: int) -> None: ...

class DeepseekV2PreTrainedModel(LlamaPreTrainedModel):
    _can_compile_fullgraph = ...

class DeepseekV2Model(LlamaModel): ...
class DeepseekV2ForCausalLM(LlamaForCausalLM): ...
class DeepseekV2ForSequenceClassification(LlamaForSequenceClassification): ...

__all__ = [
    "DeepseekV2Config",
    "DeepseekV2ForCausalLM",
    "DeepseekV2ForSequenceClassification",
    "DeepseekV2Model",
    "DeepseekV2PreTrainedModel",
]
