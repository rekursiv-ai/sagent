from collections.abc import Iterable

import numpy as np
import PIL

from ...image_processing_utils import BaseImageProcessor
from ...image_transforms import PaddingMode
from ...image_utils import ChannelDimension, ImageInput, PILImageResampling
from ...utils import TensorType, filter_out_non_signature_kwargs

"""Image processor class for TVP."""
logger = ...

def make_batched(videos) -> list[list[ImageInput]]: ...
def get_resize_output_image_size(
    input_image: np.ndarray,
    max_size: int = ...,
    input_data_format: str | ChannelDimension | None = ...,
) -> tuple[int, int]: ...

class TvpImageProcessor(BaseImageProcessor):
    model_input_names = ...
    def __init__(
        self,
        do_resize: bool = ...,
        size: dict[str, int] | None = ...,
        resample: PILImageResampling = ...,
        do_center_crop: bool = ...,
        crop_size: dict[str, int] | None = ...,
        do_rescale: bool = ...,
        rescale_factor: float = ...,
        do_pad: bool = ...,
        pad_size: dict[str, int] | None = ...,
        constant_values: float | Iterable[float] = ...,
        pad_mode: PaddingMode = ...,
        do_normalize: bool = ...,
        do_flip_channel_order: bool = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
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
    def pad_image(
        self,
        image: np.ndarray,
        pad_size: dict[str, int] | None = ...,
        constant_values: float | Iterable[float] = ...,
        pad_mode: PaddingMode = ...,
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
        **kwargs,
    ):  # -> ndarray[Any, Any]:
        ...
    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        videos: ImageInput | list[ImageInput] | list[list[ImageInput]],
        do_resize: bool | None = ...,
        size: dict[str, int] | None = ...,
        resample: PILImageResampling | None = ...,
        do_center_crop: bool | None = ...,
        crop_size: dict[str, int] | None = ...,
        do_rescale: bool | None = ...,
        rescale_factor: float | None = ...,
        do_pad: bool | None = ...,
        pad_size: dict[str, int] | None = ...,
        constant_values: float | Iterable[float] | None = ...,
        pad_mode: PaddingMode | None = ...,
        do_normalize: bool | None = ...,
        do_flip_channel_order: bool | None = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        return_tensors: str | TensorType | None = ...,
        data_format: ChannelDimension = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ) -> PIL.Image.Image: ...

__all__ = ["TvpImageProcessor"]
