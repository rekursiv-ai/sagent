from collections.abc import Iterable

import numpy as np

from ...image_processing_utils import BaseImageProcessor
from ...image_utils import ChannelDimension, ImageInput, PILImageResampling
from ...utils import TensorType

logger = ...

def smart_resize(
    height: int,
    width: int,
    factor: int = ...,
    min_pixels: int = ...,
    max_pixels: int = ...,
):  # -> tuple[int, int]:
    ...

class Emu3ImageProcessor(BaseImageProcessor):
    model_input_names = ...
    def __init__(
        self,
        do_resize: bool = ...,
        resample: PILImageResampling = ...,
        do_rescale: bool = ...,
        rescale_factor: float = ...,
        do_normalize: bool = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        do_convert_rgb: bool = ...,
        do_pad: bool = ...,
        min_pixels: int = ...,
        max_pixels: int = ...,
        spatial_factor: int = ...,
        **kwargs,
    ) -> None: ...
    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool | None = ...,
        size: dict[str, int] | None = ...,
        resample: PILImageResampling | None = ...,
        do_rescale: bool | None = ...,
        rescale_factor: float | None = ...,
        do_normalize: bool | None = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        do_convert_rgb: bool | None = ...,
        do_pad: bool = ...,
        return_tensors: str | TensorType | None = ...,
        data_format: ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ):  # -> BatchFeature:
        ...
    def postprocess(
        self,
        images: ImageInput,
        do_rescale: bool | None = ...,
        rescale_factor: float | None = ...,
        do_normalize: bool | None = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        return_tensors: str | TensorType = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ):  # -> ImageInput | Any | BatchFeature:
        ...
    def unnormalize(
        self,
        image: np.ndarray,
        image_mean: float | Iterable[float],
        image_std: float | Iterable[float],
        input_data_format: str | ChannelDimension | None = ...,
    ) -> np.ndarray: ...

__all__ = ["Emu3ImageProcessor"]
