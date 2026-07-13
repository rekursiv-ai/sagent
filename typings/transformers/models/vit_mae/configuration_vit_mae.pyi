from ...configuration_utils import PretrainedConfig

"""ViT MAE model configuration"""
logger = ...

class ViTMAEConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        hidden_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        intermediate_size=...,
        hidden_act=...,
        hidden_dropout_prob=...,
        attention_probs_dropout_prob=...,
        initializer_range=...,
        layer_norm_eps=...,
        image_size=...,
        patch_size=...,
        num_channels=...,
        qkv_bias=...,
        decoder_num_attention_heads=...,
        decoder_hidden_size=...,
        decoder_num_hidden_layers=...,
        decoder_intermediate_size=...,
        mask_ratio=...,
        norm_pix_loss=...,
        **kwargs,
    ) -> None: ...

__all__ = ["ViTMAEConfig"]
