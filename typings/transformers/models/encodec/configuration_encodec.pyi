from ...configuration_utils import PretrainedConfig

"""EnCodec model configuration"""
logger = ...

class EncodecConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        target_bandwidths=...,
        sampling_rate=...,
        audio_channels=...,
        normalize=...,
        chunk_length_s=...,
        overlap=...,
        hidden_size=...,
        num_filters=...,
        num_residual_layers=...,
        upsampling_ratios=...,
        norm_type=...,
        kernel_size=...,
        last_kernel_size=...,
        residual_kernel_size=...,
        dilation_growth_rate=...,
        use_causal_conv=...,
        pad_mode=...,
        compress=...,
        num_lstm_layers=...,
        trim_right_ratio=...,
        codebook_size=...,
        codebook_dim=...,
        use_conv_shortcut=...,
        **kwargs,
    ) -> None: ...
    @property
    def chunk_length(self) -> int | None: ...
    @property
    def chunk_stride(self) -> int | None: ...
    @property
    def hop_length(self) -> int: ...
    @property
    def codebook_nbits(self) -> int: ...
    @property
    def frame_rate(self) -> int: ...
    @property
    def num_quantizers(self) -> int: ...

__all__ = ["EncodecConfig"]
