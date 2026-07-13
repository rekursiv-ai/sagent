import numpy as np
import PIL

from ...image_processing_utils import BaseImageProcessor
from ...image_utils import ChannelDimension, ImageInput, PILImageResampling
from ...utils import TensorType, filter_out_non_signature_kwargs

"""Image processor class for Gemma3."""
logger = ...

class Gemma3ImageProcessor(BaseImageProcessor):
    model_input_names = ...
    def __init__(
        self,
        do_resize: bool = ...,
        size: dict[str, int] | None = ...,
        resample: PILImageResampling = ...,
        do_rescale: bool = ...,
        rescale_factor: float = ...,
        do_normalize: bool = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        do_convert_rgb: bool | None = ...,
        do_pan_and_scan: bool | None = ...,
        pan_and_scan_min_crop_size: int | None = ...,
        pan_and_scan_max_num_crops: int | None = ...,
        pan_and_scan_min_ratio_to_activate: float | None = ...,
        **kwargs,
    ) -> None: ...
    def pan_and_scan(
        self,
        image: np.ndarray,
        pan_and_scan_min_crop_size: int,
        pan_and_scan_max_num_crops: int,
        pan_and_scan_min_ratio_to_activate: float,
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ):  # -> list[Any] | list[ndarray[Any, Any]]:
        ...
    @filter_out_non_signature_kwargs()
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
        return_tensors: str | TensorType | None = ...,
        data_format: ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
        do_convert_rgb: bool | None = ...,
        do_pan_and_scan: bool | None = ...,
        pan_and_scan_min_crop_size: int | None = ...,
        pan_and_scan_max_num_crops: int | None = ...,
        pan_and_scan_min_ratio_to_activate: float | None = ...,
    ) -> PIL.Image.Image: ...

__all__ = ["Gemma3ImageProcessor"]
