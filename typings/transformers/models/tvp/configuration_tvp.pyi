from ...configuration_utils import PretrainedConfig

"""TVP model configuration"""
logger = ...

class TvpConfig(PretrainedConfig):
    def __init__(
        self,
        backbone_config=...,
        backbone=...,
        use_pretrained_backbone=...,
        use_timm_backbone=...,
        backbone_kwargs=...,
        distance_loss_weight=...,
        duration_loss_weight=...,
        visual_prompter_type=...,
        visual_prompter_apply=...,
        visual_prompt_size=...,
        max_img_size=...,
        num_frames=...,
        vocab_size=...,
        hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        max_position_embeddings=...,
        max_grid_col_position_embeddings=...,
        max_grid_row_position_embeddings=...,
        hidden_dropout_prob=...,
        hidden_act=...,
        layer_norm_eps=...,
        initializer_range=...,
        attention_probs_dropout_prob=...,
        **kwargs,
    ) -> None: ...
    @property
    def sub_configs(
        self,
    ):  # -> dict[str, type[PretrainedConfig] | type[Any] | type[None]]:
        ...
    @classmethod
    def from_backbone_config(
        cls, backbone_config: PretrainedConfig, **kwargs
    ):  # -> Self:
        ...
    def to_dict(self):  # -> dict[str, Any]:
        ...

__all__ = ["TvpConfig"]
