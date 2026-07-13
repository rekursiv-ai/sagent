import numpy as np
import torch

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
)
from ...image_utils import ImageInput
from ...processing_utils import Unpack
from ...utils import auto_docstring, filter_out_non_signature_kwargs

"""Fast Image processor class for EoMT."""

class EomtImageProcessorFastKwargs(DefaultFastImageProcessorKwargs):
    do_split_image: bool
    do_pad: bool
    ignore_index: int | None = ...

def get_target_size(size_dict: dict[str, int]) -> tuple[int, int]: ...
def reorder_patches_and_offsets(
    patches: list[torch.Tensor], offsets: list[list[int]]
) -> tuple[list[torch.Tensor], list[list[int]]]: ...

@auto_docstring
class EomtImageProcessorFast(BaseImageProcessorFast):
    resample = ...
    image_mean = ...
    image_std = ...
    size = ...
    default_to_square = ...
    do_resize = ...
    do_rescale = ...
    do_normalize = ...
    do_split_image = ...
    do_pad = ...
    ignore_index = ...
    valid_kwargs = EomtImageProcessorFastKwargs
    def __init__(self, **kwargs: Unpack[EomtImageProcessorFastKwargs]) -> None: ...
    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        segmentation_maps: list[torch.Tensor] | None = ...,
        instance_id_to_semantic_id: dict[int, int] | None = ...,
        **kwargs: Unpack[EomtImageProcessorFastKwargs],
    ) -> BatchFeature: ...
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

__all__ = ["EomtImageProcessorFast"]
