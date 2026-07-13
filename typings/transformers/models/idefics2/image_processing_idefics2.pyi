from collections.abc import Iterable
from typing import Any

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_utils import ChannelDimension, ImageInput, PILImageResampling
from ...utils import TensorType

logger = ...

def get_resize_output_image_size(image, size, input_data_format) -> tuple[int, int]: ...
def max_across_indices(values: Iterable[Any]) -> list[Any]: ...
def get_max_height_width(
    images_list: list[list[np.ndarray]],
    input_data_format: str | ChannelDimension | None = ...,
) -> list[int]: ...
def make_pixel_mask(
    image: np.ndarray,
    output_size: tuple[int, int],
    input_data_format: str | ChannelDimension | None = ...,
) -> np.ndarray: ...
def convert_to_rgb(image: ImageInput) -> ImageInput: ...

class Idefics2ImageProcessor(BaseImageProcessor):
    model_input_names = ...
    def __init__(
        self,
        do_convert_rgb: bool = ...,
        do_resize: bool = ...,
        size: dict[str, int] | None = ...,
        resample: PILImageResampling = ...,
        do_rescale: bool = ...,
        rescale_factor: float = ...,
        do_normalize: bool = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        do_pad: bool = ...,
        do_image_splitting: bool = ...,
        **kwargs,
    ) -> None: ...
    def resize(
        self,
        image: np.ndarray,
        size: dict[str, int],
        resample: PILImageResampling = ...,
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
        **kwargs,
    ) -> np.ndarray: ...
    def pad(
        self,
        images: list[np.ndarray],
        constant_values: float | Iterable[float] = ...,
        return_pixel_mask: bool = ...,
        return_tensors: str | TensorType | None = ...,
        data_format: ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ) -> BatchFeature: ...
    def split_image(
        self,
        image: np.ndarray,
        input_data_format: str | ChannelDimension | None = ...,
    ):  # -> list[ndarray[Any, Any]]:
        ...
    def preprocess(
        self,
        images: ImageInput,
        do_convert_rgb: bool | None = ...,
        do_resize: bool | None = ...,
        size: dict[str, int] | None = ...,
        resample: PILImageResampling | None = ...,
        do_rescale: bool | None = ...,
        rescale_factor: float | None = ...,
        do_normalize: bool | None = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        do_pad: bool | None = ...,
        do_image_splitting: bool | None = ...,
        return_tensors: str | TensorType | None = ...,
        input_data_format: ChannelDimension | None = ...,
        data_format: ChannelDimension | None = ...,
    ):  # -> BatchFeature:
        ...

__all__ = ["Idefics2ImageProcessor"]
