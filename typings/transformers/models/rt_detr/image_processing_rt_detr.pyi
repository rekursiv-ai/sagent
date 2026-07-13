from collections.abc import Callable, Iterable
from typing import Any

import pathlib

import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...image_processing_utils import BaseImageProcessor
from ...image_utils import (
    AnnotationFormat,
    AnnotationType,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
)
from ...utils import filter_out_non_signature_kwargs
from ...utils.generic import TensorType

"""Image processor class for RT-DETR."""
logger = ...
SUPPORTED_ANNOTATION_FORMATS = ...

def get_size_with_aspect_ratio(image_size, size, max_size=...) -> tuple[int, int]: ...
def get_resize_output_image_size(
    input_image: np.ndarray,
    size: int | tuple[int, int] | list[int],
    max_size: int | None = ...,
    input_data_format: str | ChannelDimension | None = ...,
) -> tuple[int, int]: ...
def get_image_size_for_max_height_width(
    input_image: np.ndarray,
    max_height: int,
    max_width: int,
    input_data_format: str | ChannelDimension | None = ...,
) -> tuple[int, int]: ...
def get_numpy_to_framework_fn(arr) -> Callable: ...
def safe_squeeze(arr: np.ndarray, axis: int | None = ...) -> np.ndarray: ...
def normalize_annotation(annotation: dict, image_size: tuple[int, int]) -> dict: ...
def max_across_indices(values: Iterable[Any]) -> list[Any]: ...
def get_max_height_width(
    images: list[np.ndarray],
    input_data_format: str | ChannelDimension | None = ...,
) -> list[int]: ...
def make_pixel_mask(
    image: np.ndarray,
    output_size: tuple[int, int],
    input_data_format: str | ChannelDimension | None = ...,
) -> np.ndarray: ...
def prepare_coco_detection_annotation(
    image,
    target,
    return_segmentation_masks: bool = ...,
    input_data_format: ChannelDimension | str | None = ...,
):  # -> dict[Any, Any]:
    ...
def resize_annotation(
    annotation: dict[str, Any],
    orig_size: tuple[int, int],
    target_size: tuple[int, int],
    threshold: float = ...,
    resample: PILImageResampling = ...,
):  # -> dict[Any, Any]:
    ...

class RTDetrImageProcessor(BaseImageProcessor):
    model_input_names = ...
    def __init__(
        self,
        format: str | AnnotationFormat = ...,
        do_resize: bool = ...,
        size: dict[str, int] | None = ...,
        resample: PILImageResampling = ...,
        do_rescale: bool = ...,
        rescale_factor: float = ...,
        do_normalize: bool = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        do_convert_annotations: bool = ...,
        do_pad: bool = ...,
        pad_size: dict[str, int] | None = ...,
        **kwargs,
    ) -> None: ...
    def prepare_annotation(
        self,
        image: np.ndarray,
        target: dict,
        format: AnnotationFormat | None = ...,
        return_segmentation_masks: bool | None = ...,
        masks_path: str | pathlib.Path | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ) -> dict: ...
    def resize(
        self,
        image: np.ndarray,
        size: dict[str, int],
        resample: PILImageResampling = ...,
        data_format: ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
        **kwargs,
    ) -> np.ndarray: ...
    def resize_annotation(
        self, annotation, orig_size, size, resample: PILImageResampling = ...
    ) -> dict: ...
    def rescale(
        self,
        image: np.ndarray,
        rescale_factor: float,
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ) -> np.ndarray: ...
    def normalize_annotation(
        self, annotation: dict, image_size: tuple[int, int]
    ) -> dict: ...
    def pad(
        self,
        images: list[np.ndarray],
        annotations: AnnotationType | list[AnnotationType] | None = ...,
        constant_values: float | Iterable[float] = ...,
        return_pixel_mask: bool = ...,
        return_tensors: str | TensorType | None = ...,
        data_format: ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
        update_bboxes: bool = ...,
        pad_size: dict[str, int] | None = ...,
    ) -> BatchFeature: ...
    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        annotations: AnnotationType | list[AnnotationType] | None = ...,
        return_segmentation_masks: bool | None = ...,
        masks_path: str | pathlib.Path | None = ...,
        do_resize: bool | None = ...,
        size: dict[str, int] | None = ...,
        resample=...,
        do_rescale: bool | None = ...,
        rescale_factor: float | None = ...,
        do_normalize: bool | None = ...,
        do_convert_annotations: bool | None = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        do_pad: bool | None = ...,
        format: str | AnnotationFormat | None = ...,
        return_tensors: TensorType | str | None = ...,
        data_format: str | ChannelDimension = ...,
        input_data_format: str | ChannelDimension | None = ...,
        pad_size: dict[str, int] | None = ...,
    ) -> BatchFeature: ...
    def post_process_object_detection(
        self,
        outputs,
        threshold: float = ...,
        target_sizes: TensorType | list[tuple] = ...,
        use_focal_loss: bool = ...,
    ):  # -> list[Any]:
        ...

__all__ = ["RTDetrImageProcessor"]
