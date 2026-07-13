import os

from ...configuration_utils import PretrainedConfig

"""CLVP model configuration"""
logger = ...

class ClvpEncoderConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        intermediate_size=...,
        projection_dim=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        hidden_act=...,
        layer_norm_eps=...,
        attention_dropout=...,
        dropout=...,
        use_rotary_embedding=...,
        use_attention_bias=...,
        summary_type=...,
        initializer_factor=...,
        bos_token_id=...,
        eos_token_id=...,
        **kwargs,
    ) -> None: ...
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        config_type: str = ...,
        **kwargs,
    ):  # -> Self:
        ...

class ClvpDecoderConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
    def __init__(
        self,
        vocab_size=...,
        max_position_embeddings=...,
        max_text_tokens=...,
        hidden_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        n_inner=...,
        num_mel_attn_blocks=...,
        activation_function=...,
        resid_pdrop=...,
        embd_pdrop=...,
        attention_dropout=...,
        layer_norm_epsilon=...,
        initializer_range=...,
        summary_type=...,
        summary_use_proj=...,
        summary_activation=...,
        summary_proj_to_labels=...,
        summary_first_dropout=...,
        use_cache=...,
        bos_token_id=...,
        eos_token_id=...,
        feature_size=...,
        use_attention_bias=...,
        initializer_factor=...,
        decoder_fixing_codes=...,
        **kwargs,
    ) -> None: ...

class ClvpConfig(PretrainedConfig):
    model_type = ...
    sub_configs = ...
    def __init__(
        self,
        text_config=...,
        speech_config=...,
        decoder_config=...,
        projection_dim=...,
        logit_scale_init_value=...,
        initializer_factor=...,
        **kwargs,
    ) -> None: ...
    @classmethod
    def from_sub_model_configs(
        cls,
        text_config: ClvpEncoderConfig,
        speech_config: ClvpEncoderConfig,
        decoder_config: ClvpDecoderConfig,
        **kwargs,
    ):  # -> Self:
        ...

__all__ = ["ClvpConfig", "ClvpDecoderConfig", "ClvpEncoderConfig"]
