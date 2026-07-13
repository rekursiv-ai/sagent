from collections.abc import Iterable

import numpy as np
import PIL

from .modeling_zoedepth import ZoeDepthDepthEstimatorOutput
from ...image_processing_utils import BaseImageProcessor
from ...image_transforms import PaddingMode
from ...image_utils import ChannelDimension, ImageInput, PILImageResampling
from ...utils import (
    TensorType,
    filter_out_non_signature_kwargs,
)

"""Image processor class for ZoeDepth."""
logger = ...

def get_resize_output_image_size(
    input_image: np.ndarray,
    output_size: int | Iterable[int],
    keep_aspect_ratio: bool,
    multiple: int,
    input_data_format: str | ChannelDimension | None = ...,
) -> tuple[int, int]: ...

class ZoeDepthImageProcessor(BaseImageProcessor):
    model_input_names = ...
    def __init__(
        self,
        do_pad: bool = ...,
        do_rescale: bool = ...,
        rescale_factor: float = ...,
        do_normalize: bool = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        do_resize: bool = ...,
        size: dict[str, int] | None = ...,
        resample: PILImageResampling = ...,
        keep_aspect_ratio: bool = ...,
        ensure_multiple_of: int = ...,
        **kwargs,
    ) -> None: ...
    def resize(
        self,
        image: np.ndarray,
        size: dict[str, int],
        keep_aspect_ratio: bool = ...,
        ensure_multiple_of: int = ...,
        resample: PILImageResampling = ...,
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ) -> np.ndarray: ...
    def pad_image(
        self,
        image: np.ndarray,
        mode: PaddingMode = ...,
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ):  # -> ndarray[Any, Any]:
        ...
    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        do_pad: bool | None = ...,
        do_rescale: bool | None = ...,
        rescale_factor: float | None = ...,
        do_normalize: bool | None = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        do_resize: bool | None = ...,
        size: int | None = ...,
        keep_aspect_ratio: bool | None = ...,
        ensure_multiple_of: int | None = ...,
        resample: PILImageResampling | None = ...,
        return_tensors: str | TensorType | None = ...,
        data_format: ChannelDimension = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ) -> PIL.Image.Image: ...
    def post_process_depth_estimation(
        self,
        outputs: ZoeDepthDepthEstimatorOutput,
        source_sizes: TensorType | list[tuple[int, int]] | None = ...,
        target_sizes: TensorType | list[tuple[int, int]] | None = ...,
        outputs_flipped: ZoeDepthDepthEstimatorOutput | None = ...,
        do_remove_padding: bool | None = ...,
    ) -> list[dict[str, TensorType]]: ...

__all__ = ["ZoeDepthImageProcessor"]
