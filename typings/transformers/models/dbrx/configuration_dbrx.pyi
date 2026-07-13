from typing import Any

from ...configuration_utils import PretrainedConfig

"""DBRX model configuration"""
logger = ...

class DbrxAttentionConfig(PretrainedConfig):
    base_config_key = ...
    def __init__(
        self,
        attn_pdrop: float = ...,
        clip_qkv: float | None = ...,
        kv_n_heads: int = ...,
        rope_theta: float = ...,
        **kwargs: Any,
    ) -> None: ...

class DbrxFFNConfig(PretrainedConfig):
    base_config_key = ...
    def __init__(
        self,
        ffn_act_fn: dict | None = ...,
        ffn_hidden_size: int = ...,
        moe_num_experts: int = ...,
        moe_top_k: int = ...,
        moe_jitter_eps: float | None = ...,
        moe_loss_weight: float = ...,
        moe_normalize_expert_weights: float | None = ...,
        **kwargs: Any,
    ) -> None: ...

class DbrxConfig(PretrainedConfig):
    model_type = ...
    sub_configs = ...
    attribute_map = ...
    def __init__(
        self,
        d_model: int = ...,
        n_heads: int = ...,
        n_layers: int = ...,
        max_seq_len: int = ...,
        vocab_size: int = ...,
        resid_pdrop: float = ...,
        emb_pdrop: float = ...,
        attn_config: DbrxAttentionConfig | None = ...,
        ffn_config: DbrxFFNConfig | None = ...,
        use_cache: bool = ...,
        initializer_range: float = ...,
        output_router_logits: bool = ...,
        **kwargs: Any,
    ) -> None: ...

__all__ = ["DbrxConfig"]
