from collections.abc import Iterable

import numpy as np

from ...image_processing_utils import BaseImageProcessor
from ...image_transforms import PaddingMode
from ...image_utils import ChannelDimension, ImageInput, PILImageResampling
from ...utils import TensorType

"""Image processor class for LLaVa-NeXT."""
logger = ...

def divide_to_patches(
    image: np.ndarray, patch_size: int, input_data_format
) -> list[np.ndarray]: ...
def expand_to_square(
    image: np.ndarray, background_color, input_data_format
) -> np.ndarray: ...

class LlavaNextImageProcessor(BaseImageProcessor):
    model_input_names = ...
    def __init__(
        self,
        do_resize: bool = ...,
        size: dict[str, int] | None = ...,
        image_grid_pinpoints: list | None = ...,
        resample: PILImageResampling = ...,
        do_center_crop: bool = ...,
        crop_size: dict[str, int] | None = ...,
        do_rescale: bool = ...,
        rescale_factor: float = ...,
        do_normalize: bool = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        do_pad: bool | None = ...,
        do_convert_rgb: bool = ...,
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
        image: np.ndarray,
        padding: int | tuple[int, int] | Iterable[tuple[int, int]],
        mode: PaddingMode = ...,
        constant_values: float | Iterable[float] = ...,
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ) -> np.ndarray: ...
    def get_image_patches(
        self,
        image: np.ndarray,
        grid_pinpoints,
        size: tuple,
        patch_size: int,
        resample: PILImageResampling,
        data_format: ChannelDimension,
        input_data_format: ChannelDimension,
    ) -> list[np.ndarray]: ...
    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool | None = ...,
        size: dict[str, int] | None = ...,
        image_grid_pinpoints: list | None = ...,
        resample: PILImageResampling | None = ...,
        do_center_crop: bool | None = ...,
        crop_size: int | None = ...,
        do_rescale: bool | None = ...,
        rescale_factor: float | None = ...,
        do_normalize: bool | None = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        do_pad: bool | None = ...,
        do_convert_rgb: bool | None = ...,
        return_tensors: str | TensorType | None = ...,
        data_format: ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ):  # -> BatchFeature:
        ...

__all__ = ["LlavaNextImageProcessor"]
