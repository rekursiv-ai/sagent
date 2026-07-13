from ...configuration_utils import PretrainedConfig

"""OPT model configuration"""
logger = ...

class OPTConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        num_hidden_layers=...,
        ffn_dim=...,
        max_position_embeddings=...,
        do_layer_norm_before=...,
        _remove_final_layer_norm=...,
        word_embed_proj_dim=...,
        dropout=...,
        attention_dropout=...,
        num_attention_heads=...,
        activation_function=...,
        layerdrop=...,
        init_std=...,
        use_cache=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        enable_bias=...,
        layer_norm_elementwise_affine=...,
        **kwargs,
    ) -> None: ...

__all__ = ["OPTConfig"]
