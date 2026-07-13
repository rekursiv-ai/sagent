from transformers import DacConfig, HubertConfig

from ...configuration_utils import PretrainedConfig

"""Xcodec model configuration"""
logger = ...

class XcodecConfig(PretrainedConfig):
    model_type = ...
    sub_configs = ...
    def __init__(
        self,
        target_bandwidths: list[float] | None = ...,
        sample_rate: int = ...,
        kernel_size: int = ...,
        channel_ratios: list[float] = ...,
        strides: list[int] = ...,
        block_dilations: list[int] = ...,
        unit_kernel_size: int = ...,
        codebook_size: int = ...,
        codebook_dim: int | None = ...,
        initializer_range: float = ...,
        acoustic_model_config: dict | DacConfig = ...,
        semantic_model_config: dict | HubertConfig = ...,
        **kwargs,
    ) -> None: ...
    @property
    def frame_rate(self) -> int: ...
    @property
    def semantic_hidden_size(self) -> int: ...
    @property
    def hop_length(self) -> int: ...
    @property
    def codebook_nbits(self) -> int: ...
    @property
    def hidden_size(self) -> int: ...
    @property
    def num_quantizers(self) -> int: ...

__all__ = ["XcodecConfig"]
