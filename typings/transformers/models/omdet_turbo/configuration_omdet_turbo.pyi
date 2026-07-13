from ...configuration_utils import PretrainedConfig

"""OmDet-Turbo model configuration"""
logger = ...

class OmDetTurboConfig(PretrainedConfig):
    model_type = ...
    attribute_map = ...
    def __init__(
        self,
        text_config=...,
        backbone_config=...,
        use_timm_backbone=...,
        backbone=...,
        backbone_kwargs=...,
        use_pretrained_backbone=...,
        apply_layernorm_after_vision_backbone=...,
        image_size=...,
        disable_custom_kernels=...,
        layer_norm_eps=...,
        batch_norm_eps=...,
        init_std=...,
        text_projection_in_dim=...,
        text_projection_out_dim=...,
        task_encoder_hidden_dim=...,
        class_embed_dim=...,
        class_distance_type=...,
        num_queries=...,
        csp_activation=...,
        conv_norm_activation=...,
        encoder_feedforward_activation=...,
        encoder_feedforward_dropout=...,
        encoder_dropout=...,
        hidden_expansion=...,
        vision_features_channels=...,
        encoder_hidden_dim=...,
        encoder_in_channels=...,
        encoder_projection_indices=...,
        encoder_attention_heads=...,
        encoder_dim_feedforward=...,
        encoder_layers=...,
        positional_encoding_temperature=...,
        num_feature_levels=...,
        decoder_hidden_dim=...,
        decoder_num_heads=...,
        decoder_num_layers=...,
        decoder_activation=...,
        decoder_dim_feedforward=...,
        decoder_num_points=...,
        decoder_dropout=...,
        eval_size=...,
        learn_initial_query=...,
        cache_size=...,
        is_encoder_decoder=...,
        **kwargs,
    ) -> None: ...
    @property
    def sub_configs(self):  # -> dict[Any, Any]:
        ...

__all__ = ["OmDetTurboConfig"]
