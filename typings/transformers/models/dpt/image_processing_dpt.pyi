from collections.abc import Iterable

import numpy as np
import PIL

from ...image_processing_utils import BaseImageProcessor
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
)
from ...modeling_outputs import DepthEstimatorOutput
from ...utils import TensorType, filter_out_non_signature_kwargs
from ...utils.import_utils import requires

"""Image processor class for DPT."""
logger = ...

def get_resize_output_image_size(
    input_image: np.ndarray,
    output_size: int | Iterable[int],
    keep_aspect_ratio: bool,
    multiple: int,
    input_data_format: str | ChannelDimension | None = ...,
) -> tuple[int, int]: ...

@requires(backends=("vision",))
class DPTImageProcessor(BaseImageProcessor):
    model_input_names = ...
    def __init__(
        self,
        do_resize: bool = ...,
        size: dict[str, int] | None = ...,
        resample: PILImageResampling = ...,
        keep_aspect_ratio: bool = ...,
        ensure_multiple_of: int = ...,
        do_rescale: bool = ...,
        rescale_factor: float = ...,
        do_normalize: bool = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        do_pad: bool = ...,
        size_divisor: int | None = ...,
        do_reduce_labels: bool = ...,
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
        **kwargs,
    ) -> np.ndarray: ...
    def pad_image(
        self,
        image: np.ndarray,
        size_divisor: int,
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ):  # -> ndarray[Any, Any]:
        ...
    def reduce_label(self, label: ImageInput) -> np.ndarray: ...
    def __call__(self, images, segmentation_maps=..., **kwargs):  # -> BatchFeature:
        ...
    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        segmentation_maps: ImageInput | None = ...,
        do_resize: bool | None = ...,
        size: int | None = ...,
        keep_aspect_ratio: bool | None = ...,
        ensure_multiple_of: int | None = ...,
        resample: PILImageResampling | None = ...,
        do_rescale: bool | None = ...,
        rescale_factor: float | None = ...,
        do_normalize: bool | None = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        do_pad: bool | None = ...,
        size_divisor: int | None = ...,
        do_reduce_labels: bool | None = ...,
        return_tensors: str | TensorType | None = ...,
        data_format: ChannelDimension = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ) -> PIL.Image.Image: ...
    def post_process_semantic_segmentation(
        self, outputs, target_sizes: list[tuple] | None = ...
    ):  # -> list[Any]:
        ...
    def post_process_depth_estimation(
        self,
        outputs: DepthEstimatorOutput,
        target_sizes: TensorType | list[tuple[int, int]] | None = ...,
    ) -> list[dict[str, TensorType]]: ...

__all__ = ["DPTImageProcessor"]
