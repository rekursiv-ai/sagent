from ...configuration_utils import PretrainedConfig

"""Deformable DETR model configuration"""
logger = ...

class DeformableDetrConfig(PretrainedConfig):
    model_type = ...
    attribute_map = ...
    def __init__(
        self,
        use_timm_backbone=...,
        backbone_config=...,
        num_channels=...,
        num_queries=...,
        max_position_embeddings=...,
        encoder_layers=...,
        encoder_ffn_dim=...,
        encoder_attention_heads=...,
        decoder_layers=...,
        decoder_ffn_dim=...,
        decoder_attention_heads=...,
        encoder_layerdrop=...,
        is_encoder_decoder=...,
        activation_function=...,
        d_model=...,
        dropout=...,
        attention_dropout=...,
        activation_dropout=...,
        init_std=...,
        init_xavier_std=...,
        return_intermediate=...,
        auxiliary_loss=...,
        position_embedding_type=...,
        backbone=...,
        use_pretrained_backbone=...,
        backbone_kwargs=...,
        dilation=...,
        num_feature_levels=...,
        encoder_n_points=...,
        decoder_n_points=...,
        two_stage=...,
        two_stage_num_proposals=...,
        with_box_refine=...,
        class_cost=...,
        bbox_cost=...,
        giou_cost=...,
        mask_loss_coefficient=...,
        dice_loss_coefficient=...,
        bbox_loss_coefficient=...,
        giou_loss_coefficient=...,
        eos_coefficient=...,
        focal_alpha=...,
        disable_custom_kernels=...,
        **kwargs,
    ) -> None: ...
    @property
    def num_attention_heads(self) -> int: ...
    @property
    def hidden_size(self) -> int: ...
    @property
    def sub_configs(
        self,
    ):  # -> dict[str, type[None] | type[Any] | type[PretrainedConfig]]:
        ...

__all__ = ["DeformableDetrConfig"]
