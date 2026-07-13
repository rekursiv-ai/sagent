from collections.abc import Iterable
from typing import Any

import numpy as np
import PIL

from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_utils import ChannelDimension, ImageInput, PILImageResampling
from ...utils import TensorType

logger = ...
MAX_IMAGE_SIZE = ...

def get_resize_output_image_size(
    image,
    resolution_max_side: int,
    input_data_format: str | ChannelDimension | None = ...,
) -> tuple[int, int]: ...
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
def convert_to_rgb(
    image: np.ndarray,
    palette: PIL.ImagePalette.ImagePalette | None = ...,
    data_format: str | ChannelDimension | None = ...,
    input_data_format: str | ChannelDimension | None = ...,
) -> ImageInput: ...

class Idefics3ImageProcessor(BaseImageProcessor):
    model_input_names = ...
    def __init__(
        self,
        do_convert_rgb: bool = ...,
        do_resize: bool = ...,
        size: dict[str, int] | None = ...,
        resample: PILImageResampling = ...,
        do_image_splitting: bool = ...,
        max_image_size: dict[str, int] | None = ...,
        do_rescale: bool = ...,
        rescale_factor: float = ...,
        do_normalize: bool = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        do_pad: bool = ...,
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
    def split_image(
        self,
        image,
        max_image_size: dict[str, int],
        resample: PILImageResampling = ...,
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ):  # -> tuple[list[Any], int, int]:
        ...
    def resize_for_vision_encoder(
        self,
        image: np.ndarray,
        vision_encoder_max_size: int,
        resample: PILImageResampling = ...,
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ):  # -> ndarray[Any, Any]:
        ...
    def pad(
        self,
        images: list[list[np.ndarray]],
        constant_values: float | Iterable[float] = ...,
        return_pixel_mask: bool = ...,
        return_tensors: str | TensorType | None = ...,
        data_format: ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ) -> BatchFeature: ...
    def preprocess(
        self,
        images: ImageInput,
        do_convert_rgb: bool | None = ...,
        do_resize: bool | None = ...,
        size: dict[str, int] | None = ...,
        resample: PILImageResampling | None = ...,
        do_image_splitting: bool | None = ...,
        do_rescale: bool | None = ...,
        max_image_size: dict[str, int] | None = ...,
        rescale_factor: float | None = ...,
        do_normalize: bool | None = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        do_pad: bool | None = ...,
        return_tensors: str | TensorType | None = ...,
        return_row_col_info: bool = ...,
        data_format: ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ):  # -> BatchFeature:
        ...
    def get_number_of_image_patches(
        self, height: int, width: int, images_kwargs=...
    ):  # -> tuple[Any | Literal[1], Any | Literal[1], Any | Literal[1]]:
        ...

__all__ = ["Idefics3ImageProcessor"]
