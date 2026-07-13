import numpy as np

from ...image_processing_utils import BaseImageProcessor
from ...image_utils import ChannelDimension, ImageInput
from ...utils import TensorType, filter_out_non_signature_kwargs
from ...utils.deprecation import deprecate_kwarg

"""Image processor class for ViTMatte."""
logger = ...

class VitMatteImageProcessor(BaseImageProcessor):
    model_input_names = ...
    def __init__(
        self,
        do_rescale: bool = ...,
        rescale_factor: float = ...,
        do_normalize: bool = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        do_pad: bool = ...,
        size_divisor: int = ...,
        **kwargs,
    ) -> None: ...
    @property
    def size_divisibility(self):  # -> int:
        ...
    @size_divisibility.setter
    def size_divisibility(self, value):  # -> None:
        ...
    def pad_image(
        self,
        image: np.ndarray,
        size_divisibility: int = ...,
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ) -> np.ndarray: ...
    @filter_out_non_signature_kwargs()
    @deprecate_kwarg("size_divisibility", version="v5", new_name="size_divisor")
    def preprocess(
        self,
        images: ImageInput,
        trimaps: ImageInput,
        do_rescale: bool | None = ...,
        rescale_factor: float | None = ...,
        do_normalize: bool | None = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        do_pad: bool | None = ...,
        size_divisor: int | None = ...,
        return_tensors: str | TensorType | None = ...,
        data_format: str | ChannelDimension = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ):  # -> BatchFeature:
        ...

__all__ = ["VitMatteImageProcessor"]
