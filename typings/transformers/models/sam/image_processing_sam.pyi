import numpy as np
import torch

from ...image_processing_utils import BaseImageProcessor
from ...image_utils import ChannelDimension, ImageInput, PILImageResampling
from ...utils import (
    TensorType,
    filter_out_non_signature_kwargs,
)

"""Image processor class for SAM."""
logger = ...

class SamImageProcessor(BaseImageProcessor):
    model_input_names = ...
    def __init__(
        self,
        do_resize: bool = ...,
        size: dict[str, int] | None = ...,
        mask_size: dict[str, int] | None = ...,
        resample: PILImageResampling = ...,
        do_rescale: bool = ...,
        rescale_factor: float = ...,
        do_normalize: bool = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        do_pad: bool = ...,
        pad_size: int | None = ...,
        mask_pad_size: int | None = ...,
        do_convert_rgb: bool = ...,
        **kwargs,
    ) -> None: ...
    def pad_image(
        self,
        image: np.ndarray,
        pad_size: dict[str, int],
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
        **kwargs,
    ) -> np.ndarray: ...
    def resize(
        self,
        image: np.ndarray,
        size: dict[str, int],
        resample: PILImageResampling = ...,
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
        **kwargs,
    ) -> np.ndarray: ...
    def __call__(self, images, segmentation_maps=..., **kwargs):  # -> BatchFeature:
        ...
    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        segmentation_maps: ImageInput | None = ...,
        do_resize: bool | None = ...,
        size: dict[str, int] | None = ...,
        mask_size: dict[str, int] | None = ...,
        resample: PILImageResampling | None = ...,
        do_rescale: bool | None = ...,
        rescale_factor: float | None = ...,
        do_normalize: bool | None = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        do_pad: bool | None = ...,
        pad_size: dict[str, int] | None = ...,
        mask_pad_size: dict[str, int] | None = ...,
        do_convert_rgb: bool | None = ...,
        return_tensors: str | TensorType | None = ...,
        data_format: ChannelDimension = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ):  # -> BatchFeature:
        ...
    def post_process_masks(
        self,
        masks,
        original_sizes,
        reshaped_input_sizes,
        mask_threshold=...,
        binarize=...,
        pad_size=...,
        return_tensors=...,
    ):  # -> list[Any]:
        ...
    def post_process_for_mask_generation(
        self, all_masks, all_scores, all_boxes, crops_nms_thresh, return_tensors=...
    ):  # -> tuple[list[ndarray[Any, Any]], Any, list[Any], Any] | None:
        ...
    def generate_crop_boxes(
        self,
        image,
        target_size,
        crop_n_layers: int = ...,
        overlap_ratio: float = ...,
        points_per_crop: int | None = ...,
        crop_n_points_downscale_factor: list[int] | None = ...,
        device: torch.device | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
        return_tensors: str = ...,
    ):  # -> tuple[Tensor | Any, int | Any | None, Any, Tensor | Any]:
        ...
    def filter_masks(
        self,
        masks,
        iou_scores,
        original_size,
        cropped_box_image,
        pred_iou_thresh=...,
        stability_score_thresh=...,
        mask_threshold=...,
        stability_score_offset=...,
        return_tensors=...,
    ):  # -> tuple[list[Any], Any, Tensor] | tuple[list[Any], Any, Any] | None:
        ...

__all__ = ["SamImageProcessor"]
