import numpy as np

from .modeling_owlvit import OwlViTObjectDetectionOutput
from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_utils import ChannelDimension, ImageInput, PILImageResampling
from ...utils import TensorType, filter_out_non_signature_kwargs
from ...utils.import_utils import requires

"""Image processor class for OwlViT"""
logger = ...

def box_area(boxes): ...
def box_iou(boxes1, boxes2):  # -> tuple[Any, Any]:
    ...

@requires(backends=("vision",))
class OwlViTImageProcessor(BaseImageProcessor):
    model_input_names = ...
    def __init__(
        self,
        do_resize=...,
        size=...,
        resample=...,
        do_center_crop=...,
        crop_size=...,
        do_rescale=...,
        rescale_factor=...,
        do_normalize=...,
        image_mean=...,
        image_std=...,
        **kwargs,
    ) -> None: ...
    def resize(
        self,
        image: np.ndarray,
        size: dict[str, int],
        resample: PILImageResampling.BICUBIC,
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
        **kwargs,
    ) -> np.ndarray: ...
    def center_crop(
        self,
        image: np.ndarray,
        crop_size: dict[str, int],
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
        **kwargs,
    ) -> np.ndarray: ...
    def rescale(
        self,
        image: np.ndarray,
        rescale_factor: float,
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ) -> np.ndarray: ...
    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool | None = ...,
        size: dict[str, int] | None = ...,
        resample: PILImageResampling | None = ...,
        do_center_crop: bool | None = ...,
        crop_size: dict[str, int] | None = ...,
        do_rescale: bool | None = ...,
        rescale_factor: float | None = ...,
        do_normalize: bool | None = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        return_tensors: TensorType | str | None = ...,
        data_format: str | ChannelDimension = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ) -> BatchFeature: ...
    def post_process(self, outputs, target_sizes):  # -> list[dict[str, Tensor | str]]:
        ...
    def post_process_object_detection(
        self,
        outputs: OwlViTObjectDetectionOutput,
        threshold: float = ...,
        target_sizes: TensorType | list[tuple] | None = ...,
    ):  # -> list[Any]:
        ...
    def post_process_image_guided_detection(
        self, outputs, threshold=..., nms_threshold=..., target_sizes=...
    ):  # -> list[Any]:
        ...

__all__ = ["OwlViTImageProcessor"]
