from ...configuration_utils import PretrainedConfig

"""Dac model configuration"""
logger = ...

class DacConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        encoder_hidden_size=...,
        downsampling_ratios=...,
        decoder_hidden_size=...,
        n_codebooks=...,
        codebook_size=...,
        codebook_dim=...,
        quantizer_dropout=...,
        commitment_loss_weight=...,
        codebook_loss_weight=...,
        sampling_rate=...,
        **kwargs,
    ) -> None: ...
    @property
    def frame_rate(self) -> int: ...

__all__ = ["DacConfig"]
