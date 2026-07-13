from ...configuration_utils import PretrainedConfig

class EdgeTamVisionConfig(PretrainedConfig):
    base_config_key = ...
    model_type = ...
    sub_configs = ...
    def __init__(
        self,
        backbone_config=...,
        backbone_channel_list=...,
        backbone_feature_sizes=...,
        fpn_hidden_size=...,
        fpn_kernel_size=...,
        fpn_stride=...,
        fpn_padding=...,
        fpn_top_down_levels=...,
        num_feature_levels=...,
        hidden_act=...,
        layer_norm_eps=...,
        initializer_range=...,
        **kwargs,
    ) -> None: ...

class EdgeTamPromptEncoderConfig(PretrainedConfig):
    base_config_key = ...
    def __init__(
        self,
        hidden_size=...,
        image_size=...,
        patch_size=...,
        mask_input_channels=...,
        num_point_embeddings=...,
        hidden_act=...,
        layer_norm_eps=...,
        scale=...,
        **kwargs,
    ) -> None: ...

class EdgeTamMaskDecoderConfig(PretrainedConfig):
    base_config_key = ...
    def __init__(
        self,
        hidden_size=...,
        hidden_act=...,
        mlp_dim=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        attention_downsample_rate=...,
        num_multimask_outputs=...,
        iou_head_depth=...,
        iou_head_hidden_dim=...,
        dynamic_multimask_via_stability=...,
        dynamic_multimask_stability_delta=...,
        dynamic_multimask_stability_thresh=...,
        **kwargs,
    ) -> None: ...

class EdgeTamConfig(PretrainedConfig):
    model_type = ...
    sub_configs = ...
    def __init__(
        self,
        vision_config=...,
        prompt_encoder_config=...,
        mask_decoder_config=...,
        initializer_range=...,
        **kwargs,
    ) -> None: ...

__all__ = [
    "EdgeTamConfig",
    "EdgeTamMaskDecoderConfig",
    "EdgeTamPromptEncoderConfig",
    "EdgeTamVisionConfig",
]
