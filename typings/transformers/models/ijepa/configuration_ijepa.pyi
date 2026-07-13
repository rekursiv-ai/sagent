from ...configuration_utils import PretrainedConfig

"""I-JEPA model configuration"""

class IJepaConfig(PretrainedConfig):
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
        pooler_output_size=...,
        pooler_act=...,
        **kwargs,
    ) -> None: ...

__all__ = ["IJepaConfig"]
