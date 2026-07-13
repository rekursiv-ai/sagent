from ...configuration_utils import PretrainedConfig

"""SegGpt model configuration"""
logger = ...

class SegGptConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        hidden_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        hidden_act=...,
        hidden_dropout_prob=...,
        initializer_range=...,
        layer_norm_eps=...,
        image_size=...,
        patch_size=...,
        num_channels=...,
        qkv_bias=...,
        mlp_dim=...,
        drop_path_rate=...,
        pretrain_image_size=...,
        decoder_hidden_size=...,
        use_relative_position_embeddings=...,
        merge_index=...,
        intermediate_hidden_state_indices=...,
        beta=...,
        **kwargs,
    ) -> None: ...

__all__ = ["SegGptConfig"]
