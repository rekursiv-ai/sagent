import numpy as np
import PIL

from ...image_processing_utils import BaseImageProcessor
from ...image_utils import ChannelDimension, ImageInput, PILImageResampling
from ...utils import TensorType, filter_out_non_signature_kwargs

logger = ...

class DeepseekVLHybridImageProcessor(BaseImageProcessor):
    model_input_names = ...
    def __init__(
        self,
        do_resize: bool = ...,
        size: dict[str, int] | None = ...,
        high_res_size: dict[str, int] | None = ...,
        min_size: int = ...,
        resample: PILImageResampling = ...,
        high_res_resample: PILImageResampling = ...,
        do_rescale: bool = ...,
        rescale_factor: float = ...,
        do_normalize: bool = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        high_res_image_mean: float | list[float] | None = ...,
        high_res_image_std: float | list[float] | None = ...,
        do_convert_rgb: bool | None = ...,
        do_pad: bool = ...,
        **kwargs,
    ) -> None: ...
    def resize(
        self,
        image: np.ndarray,
        size: dict[str, int] | int,
        resample: PILImageResampling = ...,
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
        **kwargs,
    ) -> np.ndarray: ...
    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool | None = ...,
        size: dict[str, int] | None = ...,
        high_res_size: dict[str, int] | None = ...,
        resample: PILImageResampling | None = ...,
        high_res_resample: PILImageResampling | None = ...,
        do_rescale: bool | None = ...,
        rescale_factor: float | None = ...,
        do_normalize: bool | None = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        high_res_image_mean: float | list[float] | None = ...,
        high_res_image_std: float | list[float] | None = ...,
        return_tensors: str | TensorType | None = ...,
        data_format: str | ChannelDimension = ...,
        input_data_format: str | ChannelDimension | None = ...,
        do_convert_rgb: bool | None = ...,
        do_pad: bool | None = ...,
        background_color: tuple[int, int, int] | None = ...,
    ) -> PIL.Image.Image: ...
    def pad_to_square(
        self,
        image: np.ndarray,
        background_color: int | tuple[int, int, int] = ...,
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ) -> np.ndarray: ...

__all__ = ["DeepseekVLHybridImageProcessor"]
