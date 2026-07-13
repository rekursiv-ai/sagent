from ...configuration_utils import PretrainedConfig

"""OneFormer model configuration"""
logger = ...

class OneFormerConfig(PretrainedConfig):
    model_type = ...
    attribute_map = ...
    def __init__(
        self,
        backbone_config: dict | None = ...,
        backbone: str | None = ...,
        use_pretrained_backbone: bool = ...,
        use_timm_backbone: bool = ...,
        backbone_kwargs: dict | None = ...,
        ignore_value: int = ...,
        num_queries: int = ...,
        no_object_weight: int = ...,
        class_weight: float = ...,
        mask_weight: float = ...,
        dice_weight: float = ...,
        contrastive_weight: float = ...,
        contrastive_temperature: float = ...,
        train_num_points: int = ...,
        oversample_ratio: float = ...,
        importance_sample_ratio: float = ...,
        init_std: float = ...,
        init_xavier_std: float = ...,
        layer_norm_eps: float = ...,
        is_training: bool = ...,
        use_auxiliary_loss: bool = ...,
        output_auxiliary_logits: bool = ...,
        strides: list | None = ...,
        task_seq_len: int = ...,
        text_encoder_width: int = ...,
        text_encoder_context_length: int = ...,
        text_encoder_num_layers: int = ...,
        text_encoder_vocab_size: int = ...,
        text_encoder_proj_layers: int = ...,
        text_encoder_n_ctx: int = ...,
        conv_dim: int = ...,
        mask_dim: int = ...,
        hidden_dim: int = ...,
        encoder_feedforward_dim: int = ...,
        norm: str = ...,
        encoder_layers: int = ...,
        decoder_layers: int = ...,
        use_task_norm: bool = ...,
        num_attention_heads: int = ...,
        dropout: float = ...,
        dim_feedforward: int = ...,
        pre_norm: bool = ...,
        enforce_input_proj: bool = ...,
        query_dec_layers: int = ...,
        common_stride: int = ...,
        **kwargs,
    ) -> None: ...
    @property
    def sub_configs(self):  # -> dict[str, type[dict[Any, Any]] | type[None]]:
        ...

__all__ = ["OneFormerConfig"]
