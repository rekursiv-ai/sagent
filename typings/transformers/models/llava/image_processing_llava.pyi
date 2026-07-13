import numpy as np
import PIL

from ...image_processing_utils import BaseImageProcessor
from ...image_utils import ChannelDimension, ImageInput, PILImageResampling
from ...utils import TensorType

"""Image processor class for LLaVa."""
logger = ...

class LlavaImageProcessor(BaseImageProcessor):
    model_input_names = ...
    def __init__(
        self,
        do_pad: bool = ...,
        do_resize: bool = ...,
        size: dict[str, int] | None = ...,
        resample: PILImageResampling = ...,
        do_center_crop: bool = ...,
        crop_size: dict[str, int] | None = ...,
        do_rescale: bool = ...,
        rescale_factor: float = ...,
        do_normalize: bool = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        do_convert_rgb: bool = ...,
        **kwargs,
    ) -> None: ...
    def pad_to_square(
        self,
        image: np.ndarray,
        background_color: int | tuple[int, int, int] = ...,
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ) -> np.ndarray: ...
    def resize(
        self,
        image: np.ndarray,
        size: dict[str, int],
        resample: PILImageResampling = ...,
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
        **kwargs,
    ) -> np.ndarray: ...
    def preprocess(
        self,
        images: ImageInput,
        do_pad: bool | None = ...,
        do_resize: bool | None = ...,
        size: dict[str, int] | None = ...,
        resample: PILImageResampling | None = ...,
        do_center_crop: bool | None = ...,
        crop_size: int | None = ...,
        do_rescale: bool | None = ...,
        rescale_factor: float | None = ...,
        do_normalize: bool | None = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        do_convert_rgb: bool | None = ...,
        return_tensors: str | TensorType | None = ...,
        data_format: ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
        **kwargs,
    ) -> PIL.Image.Image: ...

__all__ = ["LlavaImageProcessor"]
