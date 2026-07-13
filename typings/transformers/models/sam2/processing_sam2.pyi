import torch

from ...image_utils import ImageInput
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding
from ...utils import TensorType
from ...utils.import_utils import requires

"""
Processor class for SAM2.
"""
logger = ...

@requires(backends=("torch",))
class Sam2Processor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    def __init__(
        self,
        image_processor,
        target_size: int | None = ...,
        point_pad_value: int = ...,
        **kwargs,
    ) -> None: ...
    def __call__(
        self,
        images: ImageInput | None = ...,
        segmentation_maps: ImageInput | None = ...,
        input_points: list[list[list[list[float]]]] | torch.Tensor | None = ...,
        input_labels: list[list[list[int]]] | torch.Tensor | None = ...,
        input_boxes: list[list[list[float]]] | torch.Tensor | None = ...,
        original_sizes: list[list[float]] | torch.Tensor | None = ...,
        return_tensors: str | TensorType | None = ...,
        **kwargs,
    ) -> BatchEncoding: ...
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
    ): ...

__all__ = ["Sam2Processor"]
