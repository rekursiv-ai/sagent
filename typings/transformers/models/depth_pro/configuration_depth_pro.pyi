from ...configuration_utils import PretrainedConfig

"""DepthPro model configuration"""
logger = ...

class DepthProConfig(PretrainedConfig):
    model_type = ...
    sub_configs = ...
    def __init__(
        self,
        fusion_hidden_size=...,
        patch_size=...,
        initializer_range=...,
        intermediate_hook_ids=...,
        intermediate_feature_dims=...,
        scaled_images_ratios=...,
        scaled_images_overlap_ratios=...,
        scaled_images_feature_dims=...,
        merge_padding_value=...,
        use_batch_norm_in_fusion_residual=...,
        use_bias_in_fusion_residual=...,
        use_fov_model=...,
        num_fov_head_layers=...,
        image_model_config=...,
        patch_model_config=...,
        fov_model_config=...,
        **kwargs,
    ) -> None: ...

__all__ = ["DepthProConfig"]
