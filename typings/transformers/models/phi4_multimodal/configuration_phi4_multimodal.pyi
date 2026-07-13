from ...configuration_utils import PretrainedConfig

class Phi4MultimodalVisionConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
    def __init__(
        self,
        hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_channels=...,
        image_size=...,
        patch_size=...,
        hidden_act=...,
        layer_norm_eps=...,
        attention_dropout=...,
        crop_size: int = ...,
        image_token_id: int = ...,
        feature_layer: int = ...,
        **kwargs,
    ) -> None: ...

class Phi4MultimodalAudioConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        hidden_size: int = ...,
        intermediate_size: int = ...,
        num_blocks: int = ...,
        num_attention_heads: int = ...,
        activation: str = ...,
        chunk_size: int = ...,
        left_chunk: int = ...,
        dropout_rate: float = ...,
        ext_pw_out_channel: int = ...,
        depthwise_separable_out_channel: int = ...,
        depthwise_multiplier: int = ...,
        kernel_size: int = ...,
        conv_activation: str = ...,
        input_size: int = ...,
        conv_glu_type: str = ...,
        time_reduction: int = ...,
        bias_max_distance: int = ...,
        bias_symmetric: bool = ...,
        nemo_activation: str = ...,
        nemo_conv_channels: int = ...,
        downsample_rate: int = ...,
        initializer_range: float = ...,
        audio_token_id: int = ...,
        feature_layer: int = ...,
        **kwargs,
    ) -> None: ...

class Phi4MultimodalConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    base_model_tp_plan = ...
    base_model_pp_plan = ...
    sub_configs = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_key_value_heads=...,
        resid_pdrop=...,
        embd_pdrop=...,
        attention_dropout=...,
        hidden_act=...,
        max_position_embeddings=...,
        initializer_range=...,
        rms_norm_eps=...,
        use_cache=...,
        tie_word_embeddings=...,
        rope_theta=...,
        rope_scaling=...,
        partial_rotary_factor=...,
        bos_token_id=...,
        eos_token_id=...,
        pad_token_id=...,
        original_max_position_embeddings=...,
        sliding_window=...,
        vision_config=...,
        audio_config=...,
        **kwargs,
    ) -> None: ...

__all__ = [
    "Phi4MultimodalAudioConfig",
    "Phi4MultimodalConfig",
    "Phi4MultimodalVisionConfig",
]
