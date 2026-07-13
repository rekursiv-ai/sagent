from ...configuration_utils import PretrainedConfig

"""FalconH1 model configuration"""
logger = ...

class FalconH1Config(PretrainedConfig):
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
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        max_position_embeddings=...,
        attention_dropout=...,
        mamba_d_ssm=...,
        mamba_n_heads=...,
        mamba_d_head=...,
        mamba_n_groups=...,
        mamba_d_state=...,
        mamba_d_conv=...,
        mamba_expand=...,
        mamba_chunk_size=...,
        mamba_conv_bias=...,
        mamba_proj_bias=...,
        mamba_norm_before_gate=...,
        mamba_rms_norm=...,
        projectors_bias=...,
        rope_theta=...,
        rope_scaling=...,
        lm_head_multiplier=...,
        embedding_multiplier=...,
        mlp_multipliers=...,
        key_multiplier=...,
        attention_out_multiplier=...,
        attention_in_multiplier=...,
        ssm_multipliers=...,
        ssm_in_multiplier=...,
        ssm_out_multiplier=...,
        **kwargs,
    ) -> None: ...
    @property
    def layers_block_type(self):  # -> list[str]:
        ...

__all__ = ["FalconH1Config"]
