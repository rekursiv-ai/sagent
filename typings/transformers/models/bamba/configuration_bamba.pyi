from ...configuration_utils import PretrainedConfig

"""Bamba model configuration"""
logger = ...

class BambaConfig(PretrainedConfig):
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
        attn_layer_indices=...,
        mamba_n_heads=...,
        mamba_d_head=...,
        mamba_n_groups=...,
        mamba_d_state=...,
        mamba_d_conv=...,
        mamba_expand=...,
        mamba_chunk_size=...,
        mamba_conv_bias=...,
        mamba_proj_bias=...,
        z_loss_coefficient=...,
        **kwargs,
    ) -> None: ...
    @property
    def layers_block_type(self):  # -> list[str]:
        ...

__all__ = ["BambaConfig"]
