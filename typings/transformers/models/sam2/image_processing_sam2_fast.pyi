import torch

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
)
from ...image_utils import ImageInput
from ...processing_utils import Unpack
from ...utils import auto_docstring

class Sam2FastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    mask_size: dict[str, int] | None

@auto_docstring
class Sam2ImageProcessorFast(BaseImageProcessorFast):
    resample = ...
    image_mean = ...
    image_std = ...
    size = ...
    mask_size = ...
    do_resize = ...
    do_rescale = ...
    do_normalize = ...
    do_convert_rgb = ...
    valid_kwargs = Sam2FastImageProcessorKwargs
    do_pad = ...
    pad_size = ...
    mask_pad_size = ...
    def __init__(self, **kwargs: Unpack[Sam2FastImageProcessorKwargs]) -> None: ...
    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        segmentation_maps: ImageInput | None = ...,
        **kwargs: Unpack[Sam2FastImageProcessorKwargs],
    ) -> BatchFeature: ...
    def generate_crop_boxes(
        self,
        image: torch.Tensor,
        target_size,
        crop_n_layers: int = ...,
        overlap_ratio: float = ...,
        points_per_crop: int | None = ...,
        crop_n_points_downscale_factor: list[int] | None = ...,
        device: torch.device | None = ...,
    ):  # -> tuple[Any, Any | int | None, Any, Any]:
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
    ):  # -> tuple[list[Any], Any, Tensor]:
        ...
    def post_process_masks(
        self,
        masks,
        original_sizes,
        mask_threshold=...,
        binarize=...,
        max_hole_area=...,
        max_sprinkle_area=...,
        apply_non_overlapping_constraints=...,
        **kwargs,
    ):  # -> list[Any]:
        ...
    def post_process_for_mask_generation(
        self, all_masks, all_scores, all_boxes, crops_nms_thresh
    ):  # -> tuple[list[Tensor], Any, list[Any], Any]:
        ...
    def pad_image(self): ...

__all__ = ["Sam2ImageProcessorFast"]
