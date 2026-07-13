from ...configuration_utils import PretrainedConfig

"""Mimi model configuration"""
logger = ...

class MimiConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        sampling_rate=...,
        frame_rate=...,
        audio_channels=...,
        hidden_size=...,
        num_filters=...,
        num_residual_layers=...,
        upsampling_ratios=...,
        kernel_size=...,
        last_kernel_size=...,
        residual_kernel_size=...,
        dilation_growth_rate=...,
        use_causal_conv=...,
        pad_mode=...,
        compress=...,
        trim_right_ratio=...,
        codebook_size=...,
        codebook_dim=...,
        num_quantizers=...,
        use_conv_shortcut=...,
        vector_quantization_hidden_dimension=...,
        num_semantic_quantizers=...,
        upsample_groups=...,
        num_hidden_layers=...,
        intermediate_size=...,
        num_attention_heads=...,
        num_key_value_heads=...,
        head_dim=...,
        hidden_act=...,
        max_position_embeddings=...,
        initializer_range=...,
        norm_eps=...,
        use_cache=...,
        use_streaming=...,
        rope_theta=...,
        sliding_window=...,
        attention_dropout=...,
        layer_scale_initial_scale=...,
        attention_bias=...,
        **kwargs,
    ) -> None: ...
    @property
    def encodec_frame_rate(self) -> int: ...
    @property
    def num_codebooks(self) -> int: ...
    @property
    def frame_size(self) -> int: ...
    @property
    def frame_rate(self) -> float: ...

__all__ = ["MimiConfig"]
