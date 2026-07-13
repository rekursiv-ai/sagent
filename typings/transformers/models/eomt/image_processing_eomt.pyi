import numpy as np
import torch

from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_utils import ChannelDimension, ImageInput, PILImageResampling
from ...utils import TensorType, filter_out_non_signature_kwargs

"""Image processor class for EoMT."""
logger = ...

def convert_segmentation_map_to_binary_masks(
    segmentation_map: np.ndarray,
    instance_id_to_semantic_id: dict[int, int] | None = ...,
    ignore_index: int | None = ...,
):  # -> tuple[NDArray[floating[_32Bit]], NDArray[signedinteger[_64Bit]]]:
    ...
def get_size_with_aspect_ratio(image_size, size, max_size=...) -> tuple[int, int]: ...
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
    stuff_classes,
    mask_threshold: float = ...,
    overlap_mask_area_threshold: float = ...,
    target_size: tuple[int, int] | None = ...,
):  # -> tuple[Tensor, list[dict[Any, Any]]]:
    ...
def get_target_size(size_dict: dict[str, int]) -> tuple[int, int]: ...

class EomtImageProcessor(BaseImageProcessor):
    model_input_names = ...
    def __init__(
        self,
        do_resize: bool = ...,
        size: dict[str, int] | None = ...,
        resample: PILImageResampling = ...,
        do_rescale: bool = ...,
        rescale_factor: float = ...,
        do_normalize: bool = ...,
        do_split_image: bool = ...,
        do_pad: bool = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        ignore_index: int | None = ...,
        num_labels: int | None = ...,
        **kwargs,
    ) -> None: ...
    def resize(
        self,
        image: np.ndarray,
        size: dict,
        resample: PILImageResampling = ...,
        data_format=...,
        input_data_format: str | ChannelDimension | None = ...,
        **kwargs,
    ) -> np.ndarray: ...
    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        segmentation_maps: list[dict[int, int]] | dict[int, int] | None = ...,
        instance_id_to_semantic_id: dict[int, int] | None = ...,
        do_split_image: bool | None = ...,
        do_resize: bool | None = ...,
        size: dict[str, int] | None = ...,
        resample: PILImageResampling | None = ...,
        do_rescale: bool | None = ...,
        rescale_factor: float | None = ...,
        do_normalize: bool | None = ...,
        do_pad: bool | None = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        ignore_index: int | None = ...,
        return_tensors: str | TensorType | None = ...,
        data_format: str | ChannelDimension = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ) -> BatchFeature: ...
    def encode_inputs(
        self,
        pixel_values_list: list[ImageInput],
        segmentation_maps: ImageInput | None = ...,
        instance_id_to_semantic_id: list[dict[int, int]] | dict[int, int] | None = ...,
        ignore_index: int | None = ...,
        return_tensors: str | TensorType | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ):  # -> BatchFeature:
        ...
    def merge_image_patches(
        self,
        segmentation_logits: torch.Tensor,
        patch_offsets: list[tuple[int, int, int]],
        target_sizes: list[tuple[int, int]],
        size: dict[str, int],
    ) -> list[torch.Tensor]: ...
    def unpad_image(
        self,
        segmentation_logits: torch.Tensor,
        target_sizes: list[tuple[int, int]],
        size: dict[str, int],
    ) -> list[torch.Tensor]: ...
    def post_process_semantic_segmentation(
        self,
        outputs,
        target_sizes: list[tuple[int, int]],
        size: dict[str, int] | None = ...,
    ) -> np.ndarray: ...
    def post_process_panoptic_segmentation(
        self,
        outputs,
        target_sizes: list[tuple[int, int]],
        threshold: float = ...,
        mask_threshold: float = ...,
        overlap_mask_area_threshold: float = ...,
        stuff_classes: list[int] | None = ...,
        size: dict[str, int] | None = ...,
    ):  # -> list[Any]:
        ...
    @filter_out_non_signature_kwargs()
    def post_process_instance_segmentation(
        self,
        outputs,
        target_sizes: list[tuple[int, int]],
        threshold: float = ...,
        size: dict[str, int] | None = ...,
    ):  # -> list[Any]:
        ...

__all__ = ["EomtImageProcessor"]
