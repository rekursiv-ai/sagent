from ...configuration_utils import PretrainedConfig

"""Mllama model configuration"""
logger = ...

class MllamaVisionConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
    def __init__(
        self,
        hidden_size: int = ...,
        hidden_act: str = ...,
        num_hidden_layers: int = ...,
        num_global_layers: int = ...,
        num_attention_heads: int = ...,
        num_channels: int = ...,
        intermediate_size: int = ...,
        vision_output_dim: int = ...,
        image_size: int = ...,
        patch_size: int = ...,
        norm_eps: float = ...,
        max_num_tiles: int = ...,
        intermediate_layers_indices: list[int] | None = ...,
        supported_aspect_ratios: list[list[int]] | None = ...,
        initializer_range: float = ...,
        **kwargs,
    ) -> None: ...
    @property
    def max_aspect_ratio_id(self) -> int: ...

class MllamaTextConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
    def __init__(
        self,
        vocab_size: int = ...,
        hidden_size: int = ...,
        hidden_act: str = ...,
        num_hidden_layers: int = ...,
        num_attention_heads: int = ...,
        num_key_value_heads: int = ...,
        intermediate_size: int = ...,
        rope_theta: float = ...,
        rope_scaling: dict | None = ...,
        rms_norm_eps: float = ...,
        max_position_embeddings: int = ...,
        initializer_range: float = ...,
        use_cache: bool = ...,
        tie_word_embeddings: bool = ...,
        cross_attention_layers: list[int] | None = ...,
        dropout: float = ...,
        bos_token_id: int = ...,
        eos_token_id: int = ...,
        pad_token_id: int | None = ...,
        **kwargs,
    ) -> None: ...

class MllamaConfig(PretrainedConfig):
    model_type = ...
    attribute_map = ...
    sub_configs = ...
    def __init__(
        self, vision_config=..., text_config=..., image_token_index=..., **kwargs
    ) -> None: ...

__all__ = ["MllamaConfig"]
