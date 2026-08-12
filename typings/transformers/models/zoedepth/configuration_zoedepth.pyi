from ...configuration_utils import PretrainedConfig

"""ZoeDepth model configuration"""
logger = ...
ZOEDEPTH_PRETRAINED_CONFIG_ARCHIVE_MAP = ...

class ZoeDepthConfig(PretrainedConfig):
    def __init__(
        self,
        backbone_config=...,
        backbone=...,
        use_pretrained_backbone=...,
        backbone_kwargs=...,
        hidden_act=...,
        initializer_range=...,
        batch_norm_eps=...,
        readout_type=...,
        reassemble_factors=...,
        neck_hidden_sizes=...,
        fusion_hidden_size=...,
        head_in_index=...,
        use_batch_norm_in_fusion_residual=...,
        use_bias_in_fusion_residual=...,
        num_relative_features=...,
        add_projection=...,
        bottleneck_features=...,
        num_attractors=...,
        bin_embedding_dim=...,
        attractor_alpha=...,
        attractor_gamma=...,
        attractor_kind=...,
        min_temp=...,
        max_temp=...,
        bin_centers_type=...,
        bin_configurations=...,
        num_patch_transformer_layers=...,
        patch_transformer_hidden_size=...,
        patch_transformer_intermediate_size=...,
        patch_transformer_num_attention_heads=...,
        **kwargs,
    ) -> None: ...
    @property
    def sub_configs(
        self,
    ):  # -> dict[str, type[PretrainedConfig] | type[Any] | type[None]]:
        ...

__all__ = ["ZOEDEPTH_PRETRAINED_CONFIG_ARCHIVE_MAP", "ZoeDepthConfig"]
