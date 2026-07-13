from ...configuration_utils import PretrainedConfig

logger = ...

class MMGroundingDinoConfig(PretrainedConfig):
    model_type = ...
    attribute_map = ...
    def __init__(
        self,
        backbone_config=...,
        backbone=...,
        use_pretrained_backbone=...,
        use_timm_backbone=...,
        backbone_kwargs=...,
        text_config=...,
        num_queries=...,
        encoder_layers=...,
        encoder_ffn_dim=...,
        encoder_attention_heads=...,
        decoder_layers=...,
        decoder_ffn_dim=...,
        decoder_attention_heads=...,
        is_encoder_decoder=...,
        activation_function=...,
        d_model=...,
        dropout=...,
        attention_dropout=...,
        activation_dropout=...,
        auxiliary_loss=...,
        position_embedding_type=...,
        num_feature_levels=...,
        encoder_n_points=...,
        decoder_n_points=...,
        two_stage=...,
        class_cost=...,
        bbox_cost=...,
        giou_cost=...,
        bbox_loss_coefficient=...,
        giou_loss_coefficient=...,
        focal_alpha=...,
        disable_custom_kernels=...,
        max_text_len=...,
        text_enhancer_dropout=...,
        fusion_droppath=...,
        fusion_dropout=...,
        embedding_init_target=...,
        query_dim=...,
        positional_embedding_temperature=...,
        init_std=...,
        layer_norm_eps=...,
        **kwargs,
    ) -> None: ...
    @property
    def num_attention_heads(self) -> int: ...
    @property
    def hidden_size(self) -> int: ...
    @property
    def sub_configs(self):  # -> dict[Any, Any]:
        ...

__all__ = ["MMGroundingDinoConfig"]
