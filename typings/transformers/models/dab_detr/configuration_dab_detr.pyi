from ...configuration_utils import PretrainedConfig

"""DAB-DETR model configuration"""
logger = ...

class DabDetrConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    attribute_map = ...
    def __init__(
        self,
        use_timm_backbone=...,
        backbone_config=...,
        backbone=...,
        use_pretrained_backbone=...,
        backbone_kwargs=...,
        num_queries=...,
        encoder_layers=...,
        encoder_ffn_dim=...,
        encoder_attention_heads=...,
        decoder_layers=...,
        decoder_ffn_dim=...,
        decoder_attention_heads=...,
        is_encoder_decoder=...,
        activation_function=...,
        hidden_size=...,
        dropout=...,
        attention_dropout=...,
        activation_dropout=...,
        init_std=...,
        init_xavier_std=...,
        auxiliary_loss=...,
        dilation=...,
        class_cost=...,
        bbox_cost=...,
        giou_cost=...,
        cls_loss_coefficient=...,
        bbox_loss_coefficient=...,
        giou_loss_coefficient=...,
        focal_alpha=...,
        temperature_height=...,
        temperature_width=...,
        query_dim=...,
        random_refpoints_xy=...,
        keep_query_pos=...,
        num_patterns=...,
        normalize_before=...,
        sine_position_embedding_scale=...,
        initializer_bias_prior_prob=...,
        **kwargs,
    ) -> None: ...
    @property
    def sub_configs(
        self,
    ):  # -> dict[str, type[None] | type[Any] | type[PretrainedConfig]]:
        ...

__all__ = ["DabDetrConfig"]
