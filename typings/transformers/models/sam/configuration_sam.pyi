from ...configuration_utils import PretrainedConfig

"""SAM model configuration"""
logger = ...

class SamPromptEncoderConfig(PretrainedConfig):
    def __init__(
        self,
        hidden_size=...,
        image_size=...,
        patch_size=...,
        mask_input_channels=...,
        num_point_embeddings=...,
        hidden_act=...,
        layer_norm_eps=...,
        **kwargs,
    ) -> None: ...

class SamMaskDecoderConfig(PretrainedConfig):
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
        layer_norm_eps=...,
        **kwargs,
    ) -> None: ...

class SamVisionConfig(PretrainedConfig):
    def __init__(
        self,
        hidden_size=...,
        output_channels=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_channels=...,
        image_size=...,
        patch_size=...,
        hidden_act=...,
        layer_norm_eps=...,
        attention_dropout=...,
        initializer_range=...,
        qkv_bias=...,
        mlp_ratio=...,
        use_abs_pos=...,
        use_rel_pos=...,
        window_size=...,
        global_attn_indexes=...,
        num_pos_feats=...,
        mlp_dim=...,
        **kwargs,
    ) -> None: ...

class SamConfig(PretrainedConfig):
    def __init__(
        self,
        vision_config=...,
        prompt_encoder_config=...,
        mask_decoder_config=...,
        initializer_range=...,
        **kwargs,
    ) -> None: ...

__all__ = [
    "SamConfig",
    "SamMaskDecoderConfig",
    "SamPromptEncoderConfig",
    "SamVisionConfig",
]
