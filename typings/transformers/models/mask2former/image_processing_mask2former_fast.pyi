from typing import Any

from torchvision.transforms.v2 import functional as F

import torch

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    SizeDict,
)
from ...image_utils import ImageInput
from ...processing_utils import Unpack
from ...utils import auto_docstring

logger = ...

class Mask2FormerFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    size_divisor: int | None
    ignore_index: int | None
    do_reduce_labels: bool | None
    num_labels: int | None

def convert_segmentation_map_to_binary_masks_fast(
    segmentation_map: torch.Tensor,
    instance_id_to_semantic_id: dict[int, int] | None = ...,
    ignore_index: int | None = ...,
    do_reduce_labels: bool = ...,
):  # -> tuple[Tensor, Tensor | Any]:
    ...

@auto_docstring
class Mask2FormerImageProcessorFast(BaseImageProcessorFast):
    resample = ...
    image_mean = ...
    image_std = ...
    size = ...
    default_to_square = ...
    do_resize = ...
    do_rescale = ...
    rescale_factor = ...
    do_normalize = ...
    do_pad = ...
    model_input_names = ...
    size_divisor = ...
    do_reduce_labels = ...
    valid_kwargs = Mask2FormerFastImageProcessorKwargs
    def __init__(
        self, **kwargs: Unpack[Mask2FormerFastImageProcessorKwargs]
    ) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...
    def reduce_label(self, labels: list[torch.Tensor]):  # -> None:
        ...
    def resize(
        self,
        image: torch.Tensor,
        size: SizeDict,
        size_divisor: int = ...,
        interpolation: F.InterpolationMode | None = ...,
        **kwargs,
    ) -> torch.Tensor: ...
    def pad(
        self,
        images: torch.Tensor,
        padded_size: tuple[int, int],
        segmentation_maps: torch.Tensor | None = ...,
        fill: int = ...,
        ignore_index: int = ...,
    ) -> BatchFeature: ...
    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        segmentation_maps: ImageInput | None = ...,
        instance_id_to_semantic_id: list[dict[int, int]] | dict[int, int] | None = ...,
        **kwargs: Unpack[Mask2FormerFastImageProcessorKwargs],
    ) -> BatchFeature: ...
    def post_process_semantic_segmentation(
        self, outputs, target_sizes: list[tuple[int, int]] | None = ...
    ) -> torch.Tensor: ...
    def post_process_instance_segmentation(
        self,
        outputs,
        threshold: float = ...,
        mask_threshold: float = ...,
        overlap_mask_area_threshold: float = ...,
        target_sizes: list[tuple[int, int]] | None = ...,
        return_coco_annotation: bool | None = ...,
        return_binary_maps: bool | None = ...,
    ) -> list[dict]: ...
    def post_process_panoptic_segmentation(
        self,
        outputs,
        threshold: float = ...,
        mask_threshold: float = ...,
        overlap_mask_area_threshold: float = ...,
        label_ids_to_fuse: set[int] | None = ...,
        target_sizes: list[tuple[int, int]] | None = ...,
    ) -> list[dict]: ...

__all__ = ["Mask2FormerImageProcessorFast"]
