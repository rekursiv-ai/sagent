from ...configuration_utils import PretrainedConfig

"""Jamba model configuration"""
logger = ...

class JambaConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    def __init__(
        self,
        vocab_size=...,
        tie_word_embeddings=...,
        hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_key_value_heads=...,
        hidden_act=...,
        initializer_range=...,
        rms_norm_eps=...,
        use_cache=...,
        num_logits_to_keep=...,
        output_router_logits=...,
        router_aux_loss_coef=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        sliding_window=...,
        max_position_embeddings=...,
        attention_dropout=...,
        num_experts_per_tok=...,
        num_experts=...,
        expert_layer_period=...,
        expert_layer_offset=...,
        attn_layer_period=...,
        attn_layer_offset=...,
        use_mamba_kernels=...,
        mamba_d_state=...,
        mamba_d_conv=...,
        mamba_expand=...,
        mamba_dt_rank=...,
        mamba_conv_bias=...,
        mamba_proj_bias=...,
        **kwargs,
    ) -> None: ...
    @property
    def layers_block_type(self):  # -> list[str]:
        ...
    @property
    def layers_num_experts(self):  # -> list[int]:
        ...

__all__ = ["JambaConfig"]
