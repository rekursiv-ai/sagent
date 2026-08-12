from ...configuration_utils import PretrainedConfig

logger = ...

class SuperPointConfig(PretrainedConfig):
    def __init__(
        self,
        encoder_hidden_sizes: list[int] = ...,
        decoder_hidden_size: int = ...,
        keypoint_decoder_dim: int = ...,
        descriptor_decoder_dim: int = ...,
        keypoint_threshold: float = ...,
        max_keypoints: int = ...,
        nms_radius: int = ...,
        border_removal_distance: int = ...,
        initializer_range=...,
        **kwargs,
    ) -> None: ...

__all__ = ["SuperPointConfig"]
