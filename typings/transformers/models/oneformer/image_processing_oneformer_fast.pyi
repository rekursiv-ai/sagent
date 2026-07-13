import torch

from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    BatchFeature,
    DefaultFastImageProcessorKwargs,
)
from ...image_utils import ImageInput
from ...processing_utils import Unpack
from ...utils import TensorType, auto_docstring

"""Fast Image processor class for OneFormer."""
logger = ...

def make_pixel_mask(
    image: torch.Tensor, output_size: tuple[int, int]
) -> torch.Tensor: ...
def binary_mask_to_rle(mask):  # -> list[Tensor]:
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
def convert_segmentation_map_to_binary_masks_fast(
    segmentation_map: torch.Tensor,
    instance_id_to_semantic_id: dict[int, int] | None = ...,
    ignore_index: int | None = ...,
    do_reduce_labels: bool = ...,
):  # -> tuple[Tensor, Tensor | Any]:
    ...
def get_oneformer_resize_output_image_size(
    image: torch.Tensor,
    size: int | tuple[int, int] | list[int] | tuple[int],
    max_size: int | None = ...,
    default_to_square: bool = ...,
) -> tuple: ...

class OneFormerFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    repo_path: str | None
    class_info_file: str | None
    num_text: int | None
    num_labels: int | None
    ignore_index: int | None
    do_reduce_labels: bool | None

@auto_docstring
class OneFormerImageProcessorFast(BaseImageProcessorFast):
    resample = ...
    image_mean = ...
    image_std = ...
    size = ...
    crop_size = ...
    do_resize = ...
    do_rescale = ...
    do_normalize = ...
    default_to_square = ...
    do_center_crop = ...
    do_convert_rgb = ...
    rescale_factor = ...
    ignore_index = ...
    do_reduce_labels = ...
    repo_path = ...
    class_info_file = ...
    num_text = ...
    num_labels = ...
    valid_kwargs = OneFormerFastImageProcessorKwargs
    model_input_names = ...
    def __init__(self, **kwargs: Unpack[OneFormerFastImageProcessorKwargs]) -> None: ...
    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        task_inputs: list[str] | None = ...,
        segmentation_maps: ImageInput | None = ...,
        instance_id_to_semantic_id: list[dict[int, int]] | dict[int, int] | None = ...,
        **kwargs: Unpack[OneFormerFastImageProcessorKwargs],
    ) -> BatchFeature: ...
    def pad(
        self,
        images: list[torch.Tensor],
        return_pixel_mask: bool = ...,
        return_tensors: str | TensorType | None = ...,
    ) -> BatchFeature: ...
    def convert_segmentation_map_to_binary_masks(
        self,
        segmentation_map: torch.Tensor,
        instance_id_to_semantic_id: dict[int, int] | None = ...,
        ignore_index: int | None = ...,
        do_reduce_labels: bool = ...,
    ):  # -> tuple[Tensor, Tensor | Any]:
        ...
    def get_semantic_annotations(
        self, label, num_class_obj
    ):  # -> tuple[Tensor, Tensor, Any]:
        ...
    def get_instance_annotations(
        self, label, num_class_obj
    ):  # -> tuple[Tensor, Tensor, Any]:
        ...
    def get_panoptic_annotations(
        self, label, num_class_obj
    ):  # -> tuple[Tensor, Tensor, Any]:
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

__all__ = ["OneFormerImageProcessorFast"]
