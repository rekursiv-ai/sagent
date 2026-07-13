from collections.abc import Iterable
from typing import Any

import numpy as np
import torch

from ...image_processing_utils import (
    INIT_SERVICE_KWARGS,
    BaseImageProcessor,
    BatchFeature,
)
from ...image_utils import ChannelDimension, ImageInput, PILImageResampling
from ...utils import TensorType, filter_out_non_signature_kwargs

"""Image processor class for OneFormer."""
logger = ...

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
def convert_segmentation_map_to_binary_masks(
    segmentation_map: np.ndarray,
    instance_id_to_semantic_id: dict[int, int] | None = ...,
    ignore_index: int | None = ...,
    do_reduce_labels: bool = ...,
):  # -> tuple[NDArray[floating[_32Bit]], NDArray[signedinteger[_64Bit]]]:
    ...
def get_oneformer_resize_output_image_size(
    image: np.ndarray,
    size: int | tuple[int, int] | list[int] | tuple[int],
    max_size: int | None = ...,
    default_to_square: bool = ...,
    input_data_format: str | ChannelDimension | None = ...,
) -> tuple: ...
def prepare_metadata(class_info):  # -> dict[Any, Any]:
    ...
def load_metadata(repo_id, class_info_file):  # -> Any:
    ...

class OneFormerImageProcessor(BaseImageProcessor):
    model_input_names = ...
    @filter_out_non_signature_kwargs(
        extra=["max_size", "metadata", *INIT_SERVICE_KWARGS]
    )
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
        ignore_index: int | None = ...,
        do_reduce_labels: bool = ...,
        repo_path: str | None = ...,
        class_info_file: str | None = ...,
        num_text: int | None = ...,
        num_labels: int | None = ...,
        **kwargs,
    ) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...
    @filter_out_non_signature_kwargs(extra=["max_size"])
    def resize(
        self,
        image: np.ndarray,
        size: dict[str, int],
        resample: PILImageResampling = ...,
        data_format=...,
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
    def convert_segmentation_map_to_binary_masks(
        self,
        segmentation_map: np.ndarray,
        instance_id_to_semantic_id: dict[int, int] | None = ...,
        ignore_index: int | None = ...,
        do_reduce_labels: bool = ...,
    ):  # -> tuple[NDArray[floating[_32Bit]], NDArray[signedinteger[_64Bit]]]:
        ...
    def __call__(
        self, images, task_inputs=..., segmentation_maps=..., **kwargs
    ) -> BatchFeature: ...
    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        task_inputs: list[str] | None = ...,
        segmentation_maps: ImageInput | None = ...,
        instance_id_to_semantic_id: dict[int, int] | None = ...,
        do_resize: bool | None = ...,
        size: dict[str, int] | None = ...,
        resample: PILImageResampling | None = ...,
        do_rescale: bool | None = ...,
        rescale_factor: float | None = ...,
        do_normalize: bool | None = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        ignore_index: int | None = ...,
        do_reduce_labels: bool | None = ...,
        return_tensors: str | TensorType | None = ...,
        data_format: str | ChannelDimension = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ) -> BatchFeature: ...
    def pad(
        self,
        images: list[np.ndarray],
        constant_values: float | Iterable[float] = ...,
        return_pixel_mask: bool = ...,
        return_tensors: str | TensorType | None = ...,
        data_format: ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ) -> BatchFeature: ...
    def get_semantic_annotations(
        self, label, num_class_obj
    ):  # -> tuple[NDArray[Any], NDArray[Any], list[str] | Any]:
        ...
    def get_instance_annotations(
        self, label, num_class_obj
    ):  # -> tuple[NDArray[Any], NDArray[Any], list[str] | Any]:
        ...
    def get_panoptic_annotations(
        self, label, num_class_obj
    ):  # -> tuple[NDArray[Any], NDArray[Any], list[str] | Any]:
        ...
    def encode_inputs(
        self,
        pixel_values_list: list[ImageInput],
        task_inputs: list[str],
        segmentation_maps: ImageInput | None = ...,
        instance_id_to_semantic_id: list[dict[int, int]] | dict[int, int] | None = ...,
        ignore_index: int | None = ...,
        do_reduce_labels: bool = ...,
        return_tensors: str | TensorType | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ):  # -> BatchFeature:
        ...
    def post_process_semantic_segmentation(
        self, outputs, target_sizes: list[tuple[int, int]] | None = ...
    ) -> torch.Tensor: ...
    def post_process_instance_segmentation(
        self,
        outputs,
        task_type: str = ...,
        is_demo: bool = ...,
        threshold: float = ...,
        mask_threshold: float = ...,
        overlap_mask_area_threshold: float = ...,
        target_sizes: list[tuple[int, int]] | None = ...,
        return_coco_annotation: bool | None = ...,
    ):  # -> list[dict[str, Tensor]]:
        ...
    def post_process_panoptic_segmentation(
        self,
        outputs,
        threshold: float = ...,
        mask_threshold: float = ...,
        overlap_mask_area_threshold: float = ...,
        label_ids_to_fuse: set[int] | None = ...,
        target_sizes: list[tuple[int, int]] | None = ...,
    ) -> list[dict]: ...

__all__ = ["OneFormerImageProcessor"]
