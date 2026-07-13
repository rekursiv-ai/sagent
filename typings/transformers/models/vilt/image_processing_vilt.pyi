from collections.abc import Iterable
from typing import Any

import numpy as np
import PIL

from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_utils import ChannelDimension, ImageInput, PILImageResampling
from ...utils import TensorType, filter_out_non_signature_kwargs
from ...utils.import_utils import requires

"""Image processor class for Vilt."""
logger = ...

def max_across_indices(values: Iterable[Any]) -> list[Any]: ...
def make_pixel_mask(
    image: np.ndarray,
    output_size: tuple[int, int],
    input_data_format: str | ChannelDimension | None = ...,
) -> np.ndarray: ...
def get_max_height_width(
    images: list[np.ndarray],
    input_data_format: str | ChannelDimension | None = ...,
) -> list[int]: ...
def get_resize_output_image_size(
    input_image: np.ndarray,
    shorter: int = ...,
    longer: int = ...,
    size_divisor: int = ...,
    input_data_format: str | ChannelDimension | None = ...,
) -> tuple[int, int]: ...

@requires(backends=("vision",))
class ViltImageProcessor(BaseImageProcessor):
    model_input_names = ...
    def __init__(
        self,
        do_resize: bool = ...,
        size: dict[str, int] | None = ...,
        size_divisor: int = ...,
        resample: PILImageResampling = ...,
        do_rescale: bool = ...,
        rescale_factor: float = ...,
        do_normalize: bool = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        do_pad: bool = ...,
        **kwargs,
    ) -> None: ...
    @classmethod
    def from_dict(
        cls, image_processor_dict: dict[str, Any], **kwargs
    ):  # -> tuple[Self, dict[str, Any]] | Self:
        ...
    def resize(
        self,
        image: np.ndarray,
        size: dict[str, int],
        size_divisor: int = ...,
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
    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool | None = ...,
        size: dict[str, int] | None = ...,
        size_divisor: int | None = ...,
        resample: PILImageResampling | None = ...,
        do_rescale: bool | None = ...,
        rescale_factor: float | None = ...,
        do_normalize: bool | None = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        do_pad: bool | None = ...,
        return_tensors: str | TensorType | None = ...,
        data_format: ChannelDimension = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ) -> PIL.Image.Image: ...

__all__ = ["ViltImageProcessor"]
