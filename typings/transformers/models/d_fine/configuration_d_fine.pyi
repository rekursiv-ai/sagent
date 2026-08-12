from ...configuration_utils import PretrainedConfig

logger = ...

class DFineConfig(PretrainedConfig):
    layer_types = ...
    def __init__(
        self,
        initializer_range=...,
        initializer_bias_prior_prob=...,
        layer_norm_eps=...,
        batch_norm_eps=...,
        backbone_config=...,
        backbone=...,
        use_pretrained_backbone=...,
        use_timm_backbone=...,
        freeze_backbone_batch_norms=...,
        backbone_kwargs=...,
        encoder_hidden_dim=...,
        encoder_in_channels=...,
        feat_strides=...,
        encoder_layers=...,
        encoder_ffn_dim=...,
        encoder_attention_heads=...,
        dropout=...,
        activation_dropout=...,
        encode_proj_layers=...,
        positional_encoding_temperature=...,
        encoder_activation_function=...,
        activation_function=...,
        eval_size=...,
        normalize_before=...,
        hidden_expansion=...,
        d_model=...,
        num_queries=...,
        decoder_in_channels=...,
        decoder_ffn_dim=...,
        num_feature_levels=...,
        decoder_n_points=...,
        decoder_layers=...,
        decoder_attention_heads=...,
        decoder_activation_function=...,
        attention_dropout=...,
        num_denoising=...,
        label_noise_ratio=...,
        box_noise_scale=...,
        learn_initial_query=...,
        anchor_image_size=...,
        with_box_refine=...,
        is_encoder_decoder=...,
        matcher_alpha=...,
        matcher_gamma=...,
        matcher_class_cost=...,
        matcher_bbox_cost=...,
        matcher_giou_cost=...,
        use_focal_loss=...,
        auxiliary_loss=...,
        focal_loss_alpha=...,
        focal_loss_gamma=...,
        weight_loss_vfl=...,
        weight_loss_bbox=...,
        weight_loss_giou=...,
        weight_loss_fgl=...,
        weight_loss_ddf=...,
        eos_coefficient=...,
        eval_idx=...,
        layer_scale=...,
        max_num_bins=...,
        reg_scale=...,
        depth_mult=...,
        top_prob_values=...,
        lqe_hidden_dim=...,
        lqe_layers=...,
        decoder_offset_scale=...,
        decoder_method=...,
        up=...,
        **kwargs,
    ) -> None: ...
    @property
    def num_attention_heads(self) -> int: ...
    @property
    def hidden_size(self) -> int: ...
    @property
    def sub_configs(
        self,
    ):  # -> dict[str, type[PretrainedConfig] | type[Any] | type[None]]:
        ...
    @classmethod
    def from_backbone_configs(
        cls, backbone_config: PretrainedConfig, **kwargs
    ):  # -> Self:
        ...

__all__ = ["DFineConfig"]
