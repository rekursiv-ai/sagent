from ...configuration_utils import PretrainedConfig

"""LayoutLMv2 model configuration"""
logger = ...

class LayoutLMv2Config(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        intermediate_size=...,
        hidden_act=...,
        hidden_dropout_prob=...,
        attention_probs_dropout_prob=...,
        max_position_embeddings=...,
        type_vocab_size=...,
        initializer_range=...,
        layer_norm_eps=...,
        pad_token_id=...,
        max_2d_position_embeddings=...,
        max_rel_pos=...,
        rel_pos_bins=...,
        fast_qkv=...,
        max_rel_2d_pos=...,
        rel_2d_pos_bins=...,
        convert_sync_batchnorm=...,
        image_feature_pool_shape=...,
        coordinate_size=...,
        shape_size=...,
        has_relative_attention_bias=...,
        has_spatial_attention_bias=...,
        has_visual_segment_embedding=...,
        detectron2_config_args=...,
        **kwargs,
    ) -> None: ...
    @classmethod
    def get_default_detectron2_config(
        cls,
    ):  # -> dict[str, bool | list[float] | str | list[str] | list[list[int]] | int | list[list[float]]]:
        ...
    def get_detectron2_config(self): ...

__all__ = ["LayoutLMv2Config"]
