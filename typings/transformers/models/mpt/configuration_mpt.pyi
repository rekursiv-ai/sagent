from ...configuration_utils import PretrainedConfig

"""Mpt configuration"""
logger = ...

class MptAttentionConfig(PretrainedConfig):
    base_config_key = ...
    def __init__(
        self,
        attn_type=...,
        attn_pdrop=...,
        attn_impl=...,
        clip_qkv=...,
        softmax_scale=...,
        prefix_lm=...,
        qk_ln=...,
        attn_uses_sequence_id=...,
        alibi=...,
        alibi_bias_max=...,
        **kwargs,
    ) -> None: ...

class MptConfig(PretrainedConfig):
    model_type = ...
    sub_configs = ...
    attribute_map = ...
    def __init__(
        self,
        d_model: int = ...,
        n_heads: int = ...,
        n_layers: int = ...,
        expansion_ratio: int = ...,
        max_seq_len: int = ...,
        vocab_size: int = ...,
        resid_pdrop: float = ...,
        layer_norm_epsilon: float = ...,
        emb_pdrop: float = ...,
        learned_pos_emb: bool = ...,
        attn_config: MptAttentionConfig = ...,
        init_device: str = ...,
        logit_scale: float | str | None = ...,
        no_bias: bool = ...,
        verbose: int = ...,
        embedding_fraction: float = ...,
        norm_type: str = ...,
        use_cache: bool = ...,
        initializer_range=...,
        **kwargs,
    ) -> None: ...

__all__ = ["MptConfig"]
