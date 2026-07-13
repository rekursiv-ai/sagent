from ...configuration_utils import PretrainedConfig

"""MGP-STR model configuration"""
logger = ...

class MgpstrConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        image_size=...,
        patch_size=...,
        num_channels=...,
        max_token_length=...,
        num_character_labels=...,
        num_bpe_labels=...,
        num_wordpiece_labels=...,
        hidden_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        mlp_ratio=...,
        qkv_bias=...,
        distilled=...,
        layer_norm_eps=...,
        drop_rate=...,
        attn_drop_rate=...,
        drop_path_rate=...,
        output_a3_attentions=...,
        initializer_range=...,
        **kwargs,
    ) -> None: ...

__all__ = ["MgpstrConfig"]
