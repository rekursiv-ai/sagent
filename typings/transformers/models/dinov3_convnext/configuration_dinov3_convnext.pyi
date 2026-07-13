from ...configuration_utils import PretrainedConfig

"""ConvNeXT model configuration"""
logger = ...

class DINOv3ConvNextConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        num_channels: int = ...,
        hidden_sizes: list[int] | None = ...,
        depths: list[int] | None = ...,
        hidden_act: str = ...,
        initializer_range: float = ...,
        layer_norm_eps: float = ...,
        layer_scale_init_value: float = ...,
        drop_path_rate: float = ...,
        image_size: int = ...,
        **kwargs,
    ) -> None: ...
    @property
    def num_stages(self) -> int: ...

__all__ = ["DINOv3ConvNextConfig"]
