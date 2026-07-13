from ...configuration_utils import PretrainedConfig

"""Dia model configuration"""
logger = ...

class DiaEncoderConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        max_position_embeddings: int = ...,
        num_hidden_layers: int = ...,
        hidden_size: int = ...,
        num_attention_heads: int = ...,
        num_key_value_heads: int = ...,
        head_dim: int = ...,
        intermediate_size: int = ...,
        norm_eps: float = ...,
        vocab_size: int = ...,
        hidden_act: str = ...,
        rope_theta: float = ...,
        rope_scaling: dict | None = ...,
        initializer_range: float = ...,
        **kwargs,
    ) -> None: ...

class DiaDecoderConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        max_position_embeddings: int = ...,
        num_hidden_layers: int = ...,
        hidden_size: int = ...,
        intermediate_size: int = ...,
        num_attention_heads: int = ...,
        num_key_value_heads: int = ...,
        head_dim: int = ...,
        cross_num_attention_heads: int = ...,
        cross_head_dim: int = ...,
        cross_num_key_value_heads: int = ...,
        cross_hidden_size: int = ...,
        norm_eps: float = ...,
        vocab_size: int = ...,
        hidden_act: str = ...,
        num_channels: int = ...,
        rope_theta: float = ...,
        rope_scaling: dict | None = ...,
        initializer_range: float = ...,
        use_cache: bool = ...,
        is_encoder_decoder: bool = ...,
        **kwargs,
    ) -> None: ...

class DiaConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    sub_configs = ...
    def __init__(
        self,
        encoder_config: DiaEncoderConfig | None = ...,
        decoder_config: DiaDecoderConfig | None = ...,
        norm_eps: float = ...,
        is_encoder_decoder: bool = ...,
        pad_token_id: int = ...,
        eos_token_id: int = ...,
        bos_token_id: int = ...,
        delay_pattern: list[int] | None = ...,
        initializer_range: float = ...,
        use_cache: bool = ...,
        **kwargs,
    ) -> None: ...
    def get_text_config(self, *args, **kwargs):  # -> DiaDecoderConfig:
        ...

__all__ = ["DiaConfig", "DiaDecoderConfig", "DiaEncoderConfig"]
