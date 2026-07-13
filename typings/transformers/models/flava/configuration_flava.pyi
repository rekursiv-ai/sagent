from typing import Any

from ...configuration_utils import PretrainedConfig

"""FLAVA model configurations"""
logger = ...

class FlavaImageConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
    def __init__(
        self,
        hidden_size: int = ...,
        num_hidden_layers: int = ...,
        num_attention_heads: int = ...,
        intermediate_size: int = ...,
        hidden_act: int = ...,
        hidden_dropout_prob: float = ...,
        attention_probs_dropout_prob: float = ...,
        initializer_range: float = ...,
        layer_norm_eps: float = ...,
        image_size: int = ...,
        patch_size: int = ...,
        num_channels: int = ...,
        qkv_bias: bool = ...,
        mask_token: bool = ...,
        vocab_size: int = ...,
        **kwargs,
    ) -> None: ...

class FlavaTextConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
    def __init__(
        self,
        vocab_size: int = ...,
        type_vocab_size: int = ...,
        max_position_embeddings: int = ...,
        position_embedding_type: str = ...,
        hidden_size: int = ...,
        num_hidden_layers: int = ...,
        num_attention_heads: int = ...,
        intermediate_size: int = ...,
        hidden_act: str = ...,
        hidden_dropout_prob: float = ...,
        attention_probs_dropout_prob: float = ...,
        initializer_range: float = ...,
        layer_norm_eps: float = ...,
        pad_token_id: int = ...,
        qkv_bias: bool = ...,
        **kwargs,
    ) -> None: ...

class FlavaMultimodalConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
    def __init__(
        self,
        hidden_size: int = ...,
        num_hidden_layers: int = ...,
        num_attention_heads: int = ...,
        intermediate_size: int = ...,
        hidden_act: int = ...,
        hidden_dropout_prob: int = ...,
        attention_probs_dropout_prob: int = ...,
        initializer_range: float = ...,
        layer_norm_eps: float = ...,
        qkv_bias: bool = ...,
        use_cls_token: bool = ...,
        **kwargs,
    ) -> None: ...

class FlavaImageCodebookConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
    def __init__(
        self,
        num_groups: int = ...,
        input_channels: int = ...,
        num_blocks_per_group: int = ...,
        hidden_size: int = ...,
        vocab_size: int = ...,
        freeze: int = ...,
        initializer_range: float = ...,
        **kwargs,
    ) -> None: ...

class FlavaConfig(PretrainedConfig):
    model_type = ...
    sub_configs = ...
    def __init__(
        self,
        image_config: dict[str, Any] | None = ...,
        text_config: dict[str, Any] | None = ...,
        multimodal_config: dict[str, Any] | None = ...,
        image_codebook_config: dict[str, Any] | None = ...,
        hidden_size: int = ...,
        layer_norm_eps: float = ...,
        projection_dim: int = ...,
        init_codebook: bool = ...,
        logit_scale_init_value: float = ...,
        initializer_range: float = ...,
        ce_ignore_index: int = ...,
        mim_weight: float = ...,
        mlm_weight: float = ...,
        global_contrastive_weight: float = ...,
        itm_weight: float = ...,
        mmm_image_weight: float = ...,
        mmm_text_weight: float = ...,
        global_backprop_contrastive: bool = ...,
        skip_unmasked_multimodal_encoder: bool = ...,
        return_loss: bool = ...,
        **kwargs,
    ) -> None: ...
    @classmethod
    def from_configs(
        cls,
        image_config: FlavaImageConfig,
        text_config: FlavaTextConfig,
        multimodal_config: FlavaMultimodalConfig,
        image_codebook_config: FlavaImageCodebookConfig,
        **kwargs,
    ):  # -> Self:
        ...

__all__ = [
    "FlavaConfig",
    "FlavaImageCodebookConfig",
    "FlavaImageConfig",
    "FlavaMultimodalConfig",
    "FlavaTextConfig",
]
