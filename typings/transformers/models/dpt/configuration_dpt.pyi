from ...configuration_utils import PretrainedConfig

"""DPT model configuration"""
logger = ...

class DPTConfig(PretrainedConfig):
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
        is_hybrid=...,
        qkv_bias=...,
        backbone_out_indices=...,
        readout_type=...,
        reassemble_factors=...,
        neck_hidden_sizes=...,
        fusion_hidden_size=...,
        head_in_index=...,
        use_batch_norm_in_fusion_residual=...,
        use_bias_in_fusion_residual=...,
        add_projection=...,
        use_auxiliary_head=...,
        auxiliary_loss_weight=...,
        semantic_loss_ignore_index=...,
        semantic_classifier_dropout=...,
        backbone_featmap_shape=...,
        neck_ignore_stages=...,
        backbone_config=...,
        backbone=...,
        use_pretrained_backbone=...,
        use_timm_backbone=...,
        backbone_kwargs=...,
        pooler_output_size=...,
        pooler_act=...,
        **kwargs,
    ) -> None: ...
    def to_dict(self):  # -> dict[str, Any]:
        ...
    @property
    def sub_configs(
        self,
    ):  # -> dict[str, type[BitConfig] | type[PretrainedConfig] | type[None] | type[Any]]:
        ...

__all__ = ["DPTConfig"]
