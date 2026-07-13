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

class RTDetrFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    format: str | AnnotationFormat | None
    do_convert_annotations: bool | None
    return_segmentation_masks: bool | None

SUPPORTED_ANNOTATION_FORMATS = ...

def prepare_coco_detection_annotation(
    image,
    target,
    return_segmentation_masks: bool = ...,
    input_data_format: ChannelDimension | str | None = ...,
):  # -> dict[str, Tensor]:
    ...

@auto_docstring
@requires(backends=("torchvision", "torch"))
class RTDetrImageProcessorFast(BaseImageProcessorFast):
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
    valid_kwargs = RTDetrFastImageProcessorKwargs
    do_convert_annotations = ...
    def __init__(self, **kwargs: Unpack[RTDetrFastImageProcessorKwargs]) -> None: ...
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
        **kwargs: Unpack[RTDetrFastImageProcessorKwargs],
    ) -> BatchFeature: ...
    def post_process_object_detection(
        self,
        outputs,
        threshold: float = ...,
        target_sizes: TensorType | list[tuple] = ...,
        use_focal_loss: bool = ...,
    ):  # -> list[Any]:
        ...

__all__ = ["RTDetrImageProcessorFast"]
