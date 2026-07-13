from ...configuration_utils import PretrainedConfig

"""VilT model configuration"""
logger = ...

class ViltConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        vocab_size=...,
        type_vocab_size=...,
        modality_type_vocab_size=...,
        max_position_embeddings=...,
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
        max_image_length=...,
        tie_word_embeddings=...,
        num_images=...,
        **kwargs,
    ) -> None: ...

__all__ = ["ViltConfig"]
