import numpy as np
import PIL

from .modeling_owlv2 import Owlv2ObjectDetectionOutput
from ...image_processing_utils import BaseImageProcessor
from ...image_utils import ChannelDimension, ImageInput, PILImageResampling
from ...utils import (
    TensorType,
    filter_out_non_signature_kwargs,
)

"""Image processor class for OWLv2."""
logger = ...

def box_area(boxes): ...
def box_iou(boxes1, boxes2):  # -> tuple[Any, Any]:
    ...

class Owlv2ImageProcessor(BaseImageProcessor):
    model_input_names = ...
    def __init__(
        self,
        do_rescale: bool = ...,
        rescale_factor: float = ...,
        do_pad: bool = ...,
        do_resize: bool = ...,
        size: dict[str, int] | None = ...,
        resample: PILImageResampling = ...,
        do_normalize: bool = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        **kwargs,
    ) -> None: ...
    def pad(
        self,
        image: np.ndarray,
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ):  # -> ndarray[Any, Any]:
        ...
    def resize(
        self,
        image: np.ndarray,
        size: dict[str, int],
        anti_aliasing: bool = ...,
        anti_aliasing_sigma=...,
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
        **kwargs,
    ) -> np.ndarray: ...
    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        do_pad: bool | None = ...,
        do_resize: bool | None = ...,
        size: dict[str, int] | None = ...,
        do_rescale: bool | None = ...,
        rescale_factor: float | None = ...,
        do_normalize: bool | None = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        return_tensors: str | TensorType | None = ...,
        data_format: ChannelDimension = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ) -> PIL.Image.Image: ...
    def post_process_object_detection(
        self,
        outputs: Owlv2ObjectDetectionOutput,
        threshold: float = ...,
        target_sizes: TensorType | list[tuple] | None = ...,
    ):  # -> list[Any]:
        ...
    def post_process_image_guided_detection(
        self, outputs, threshold=..., nms_threshold=..., target_sizes=...
    ):  # -> list[Any]:
        ...

__all__ = ["Owlv2ImageProcessor"]
