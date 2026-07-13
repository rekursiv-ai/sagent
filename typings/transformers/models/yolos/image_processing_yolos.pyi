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
from ...utils import (
    TensorType,
)
from ...utils.import_utils import requires

"""Image processor class for YOLOS."""
logger = ...
SUPPORTED_ANNOTATION_FORMATS = ...

def get_max_height_width(
    images: list[np.ndarray],
    input_data_format: str | ChannelDimension | None = ...,
) -> list[int]: ...
def get_size_with_aspect_ratio(
    image_size: tuple[int, int],
    size: int,
    max_size: int | None = ...,
    mod_size: int = ...,
) -> tuple[int, int]: ...
def get_image_size_for_max_height_width(
    input_image: np.ndarray,
    max_height: int,
    max_width: int,
    input_data_format: str | ChannelDimension | None = ...,
) -> tuple[int, int]: ...
def get_resize_output_image_size(
    input_image: np.ndarray,
    size: int | tuple[int, int] | list[int],
    max_size: int | None = ...,
    input_data_format: str | ChannelDimension | None = ...,
) -> tuple[int, int]: ...
def get_numpy_to_framework_fn(arr) -> Callable: ...
def safe_squeeze(arr: np.ndarray, axis: int | None = ...) -> np.ndarray: ...
def normalize_annotation(annotation: dict, image_size: tuple[int, int]) -> dict: ...
def max_across_indices(values: Iterable[Any]) -> list[Any]: ...
def make_pixel_mask(
    image: np.ndarray,
    output_size: tuple[int, int],
    input_data_format: str | ChannelDimension | None = ...,
) -> np.ndarray: ...
def convert_coco_poly_to_mask(segmentations, height: int, width: int) -> np.ndarray: ...
def prepare_coco_detection_annotation(
    image,
    target,
    return_segmentation_masks: bool = ...,
    input_data_format: ChannelDimension | str | None = ...,
):  # -> dict[Any, Any]:
    ...
def masks_to_boxes(masks: np.ndarray) -> np.ndarray: ...
def prepare_coco_panoptic_annotation(
    image: np.ndarray,
    target: dict,
    masks_path: str | pathlib.Path,
    return_masks: bool = ...,
    input_data_format: ChannelDimension | str = ...,
) -> dict: ...
def get_segmentation_image(
    masks: np.ndarray,
    input_size: tuple,
    target_size: tuple,
    stuff_equiv_classes,
    deduplicate=...,
):  # -> ndarray[Any, Any]:
    ...
def get_mask_area(
    seg_img: np.ndarray, target_size: tuple[int, int], n_classes: int
) -> np.ndarray: ...
def score_labels_from_class_probabilities(
    logits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]: ...
def resize_annotation(
    annotation: dict[str, Any],
    orig_size: tuple[int, int],
    target_size: tuple[int, int],
    threshold: float = ...,
    resample: PILImageResampling = ...,
):  # -> dict[Any, Any]:
    ...
def binary_mask_to_rle(mask):  # -> list[Any]:
    ...
def convert_segmentation_to_rle(segmentation):  # -> list[Any]:
    ...
def remove_low_and_no_objects(
    masks, scores, labels, object_mask_threshold, num_labels
):  # -> tuple[Any, Any, Any]:
    ...
def check_segment_validity(
    mask_labels, mask_probs, k, mask_threshold=..., overlap_mask_area_threshold=...
):  # -> tuple[Any | Literal[False], Any]:
    ...
def compute_segments(
    mask_probs,
    pred_scores,
    pred_labels,
    mask_threshold: float = ...,
    overlap_mask_area_threshold: float = ...,
    label_ids_to_fuse: set[int] | None = ...,
    target_size: tuple[int, int] | None = ...,
):  # -> tuple[Tensor, list[dict[Any, Any]]]:
    ...

@requires(backends=("vision",))
class YolosImageProcessor(BaseImageProcessor):
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
        do_convert_annotations: bool | None = ...,
        do_pad: bool = ...,
        pad_size: dict[str, int] | None = ...,
        **kwargs,
    ) -> None: ...
    @classmethod
    def from_dict(
        cls, image_processor_dict: dict[str, Any], **kwargs
    ):  # -> tuple[Self, dict[str, Any]] | Self:
        ...
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
        annotations: list[dict[str, Any]] | None = ...,
        constant_values: float | Iterable[float] = ...,
        return_pixel_mask: bool = ...,
        return_tensors: str | TensorType | None = ...,
        data_format: ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
        update_bboxes: bool = ...,
        pad_size: dict[str, int] | None = ...,
    ) -> BatchFeature: ...
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
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        do_convert_annotations: bool | None = ...,
        do_pad: bool | None = ...,
        format: str | AnnotationFormat | None = ...,
        return_tensors: TensorType | str | None = ...,
        data_format: str | ChannelDimension = ...,
        input_data_format: str | ChannelDimension | None = ...,
        pad_size: dict[str, int] | None = ...,
        **kwargs,
    ) -> BatchFeature: ...
    def post_process(self, outputs, target_sizes):  # -> list[dict[str, Tensor | str]]:
        ...
    def post_process_object_detection(
        self,
        outputs,
        threshold: float = ...,
        target_sizes: TensorType | list[tuple] = ...,
    ):  # -> list[Any]:
        ...

__all__ = ["YolosImageProcessor"]
