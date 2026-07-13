from ...configuration_utils import PretrainedConfig

"""Zamba model configuration"""
logger = ...

class ZambaConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    def __init__(
        self,
        vocab_size=...,
        tie_word_embeddings=...,
        hidden_size=...,
        attention_hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        attention_head_dim=...,
        num_key_value_heads=...,
        n_mamba_heads=...,
        hidden_act=...,
        hidden_mamba_act=...,
        initializer_range=...,
        rms_norm_eps=...,
        use_cache=...,
        num_logits_to_keep=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        max_position_embeddings=...,
        attention_dropout=...,
        attn_layer_period=...,
        attn_layer_offset=...,
        use_mamba_kernels=...,
        mamba_d_state=...,
        mamba_d_conv=...,
        mamba_expand=...,
        mamba_dt_rank=...,
        time_step_min=...,
        time_step_max=...,
        time_step_floor=...,
        mamba_conv_bias=...,
        mamba_proj_bias=...,
        **kwargs,
    ) -> None: ...

__all__ = ["ZambaConfig"]
