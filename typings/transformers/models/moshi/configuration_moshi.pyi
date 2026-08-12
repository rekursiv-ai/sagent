from ...configuration_utils import PretrainedConfig

"""Moshi model configuration"""
logger = ...

class MoshiDepthConfig(PretrainedConfig):
    keys_to_ignore_at_inference = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        input_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_key_value_heads=...,
        audio_vocab_size=...,
        max_position_embeddings=...,
        hidden_act=...,
        head_dim=...,
        initializer_range=...,
        use_cache=...,
        sliding_window=...,
        attention_dropout=...,
        ffn_dim=...,
        rms_norm_eps=...,
        num_codebooks=...,
        tie_word_embeddings=...,
        **kwargs,
    ) -> None: ...

class MoshiConfig(PretrainedConfig):
    keys_to_ignore_at_inference = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_key_value_heads=...,
        audio_vocab_size=...,
        max_position_embeddings=...,
        rope_theta=...,
        hidden_act=...,
        head_dim=...,
        initializer_range=...,
        use_cache=...,
        sliding_window=...,
        attention_dropout=...,
        ffn_dim=...,
        rms_norm_eps=...,
        num_codebooks=...,
        tie_word_embeddings=...,
        **kwargs,
    ) -> None: ...
    @property
    def sampling_rate(self):  # -> Any:
        ...
    @classmethod
    def from_audio_encoder_config(
        cls, audio_encoder_config: PretrainedConfig, **kwargs
    ):  # -> Self:
        ...

__all__ = ["MoshiConfig", "MoshiDepthConfig"]
