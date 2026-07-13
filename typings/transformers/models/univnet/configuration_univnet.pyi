from ...configuration_utils import PretrainedConfig

"""UnivNetModel model configuration"""
logger = ...

class UnivNetConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        model_in_channels=...,
        model_hidden_channels=...,
        num_mel_bins=...,
        resblock_kernel_sizes=...,
        resblock_stride_sizes=...,
        resblock_dilation_sizes=...,
        kernel_predictor_num_blocks=...,
        kernel_predictor_hidden_channels=...,
        kernel_predictor_conv_size=...,
        kernel_predictor_dropout=...,
        initializer_range=...,
        leaky_relu_slope=...,
        **kwargs,
    ) -> None: ...

__all__ = ["UnivNetConfig"]
