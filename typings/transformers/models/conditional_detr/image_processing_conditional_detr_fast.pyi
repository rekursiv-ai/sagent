from typing import Any

import pathlib

from torchvision.transforms.v2 import functional as F

import torch

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    SizeDict,
)
from ...image_utils import (
    AnnotationFormat,
    AnnotationType,
    ChannelDimension,
    ImageInput,
)
from ...processing_utils import Unpack
from ...utils import TensorType, auto_docstring
from ...utils.import_utils import requires

logger = ...

class ConditionalDetrFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    format: str | AnnotationFormat | None
    do_convert_annotations: bool | None
    return_segmentation_masks: bool | None

SUPPORTED_ANNOTATION_FORMATS = ...

def convert_coco_poly_to_mask(
    segmentations, height: int, width: int, device: torch.device
) -> torch.Tensor: ...
def prepare_coco_detection_annotation(
    image,
    target,
    return_segmentation_masks: bool = ...,
    input_data_format: ChannelDimension | str | None = ...,
):  # -> dict[str, Tensor]:
    ...
def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor: ...
def rgb_to_id(color):  # -> Tensor | int:
    ...
def prepare_coco_panoptic_annotation(
    image: torch.Tensor,
    target: dict,
    masks_path: str | pathlib.Path,
    return_masks: bool = ...,
    input_data_format: ChannelDimension | str = ...,
) -> dict: ...

@auto_docstring
@requires(backends=("torchvision", "torch"))
class ConditionalDetrImageProcessorFast(BaseImageProcessorFast):
    resample = ...
    image_mean = ...
    image_std = ...
    format = ...
    do_resize = ...
    do_rescale = ...
    do_normalize = ...
    do_pad = ...
    size = ...
    default_to_square = ...
    model_input_names = ...
    valid_kwargs = ConditionalDetrFastImageProcessorKwargs
    def __init__(
        self, **kwargs: Unpack[ConditionalDetrFastImageProcessorKwargs]
    ) -> None: ...
    @classmethod
    def from_dict(
        cls, image_processor_dict: dict[str, Any], **kwargs
    ):  # -> tuple[Self, dict[str, Any]] | Self:
        ...
    def prepare_annotation(
        self,
        image: torch.Tensor,
        target: dict,
        format: AnnotationFormat | None = ...,
        return_segmentation_masks: bool | None = ...,
        masks_path: str | pathlib.Path | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ) -> dict: ...
    def resize(
        self,
        image: torch.Tensor,
        size: SizeDict,
        interpolation: F.InterpolationMode | None = ...,
        **kwargs,
    ) -> torch.Tensor: ...
    def resize_annotation(
        self,
        annotation: dict[str, Any],
        orig_size: tuple[int, int],
        target_size: tuple[int, int],
        threshold: float = ...,
        interpolation: F.InterpolationMode | None = ...,
    ):  # -> dict[Any, Any]:
        ...
    def normalize_annotation(
        self, annotation: dict, image_size: tuple[int, int]
    ) -> dict: ...
    def pad(
        self,
        image: torch.Tensor,
        padded_size: tuple[int, int],
        annotation: dict[str, Any] | None = ...,
        update_bboxes: bool = ...,
        fill: int = ...,
    ):  # -> tuple[Any | Tensor, Tensor, dict[str, Any] | None]:
        ...
    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        annotations: AnnotationType | list[AnnotationType] | None = ...,
        masks_path: str | pathlib.Path | None = ...,
        **kwargs: Unpack[ConditionalDetrFastImageProcessorKwargs],
    ) -> BatchFeature: ...
    def post_process(self, outputs, target_sizes):  # -> list[dict[str, Tensor | Any]]:
        ...
    def post_process_object_detection(
        self,
        outputs,
        threshold: float = ...,
        target_sizes: TensorType | list[tuple] = ...,
        top_k: int = ...,
    ):  # -> list[Any]:
        ...
    def post_process_semantic_segmentation(
        self, outputs, target_sizes: list[tuple[int, int]] | None = ...
    ):  # -> list[Tensor]:
        ...
    def post_process_instance_segmentation(
        self,
        outputs,
        threshold: float = ...,
        mask_threshold: float = ...,
        overlap_mask_area_threshold: float = ...,
        target_sizes: list[tuple[int, int]] | None = ...,
        return_coco_annotation: bool | None = ...,
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

__all__ = ["ConditionalDetrImageProcessorFast"]
