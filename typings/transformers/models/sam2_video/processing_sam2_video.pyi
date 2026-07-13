import numpy as np
import torch

from .modeling_sam2_video import Sam2VideoInferenceSession
from ...image_utils import ImageInput
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding
from ...utils import TensorType
from ...utils.import_utils import requires
from ...video_utils import VideoInput

@requires(backends=("torch",))
class Sam2VideoProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    video_processor_class = ...
    def __init__(
        self,
        image_processor,
        video_processor,
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
    def init_video_session(
        self,
        video: VideoInput | None = ...,
        inference_device: str | torch.device = ...,
        inference_state_device: str | torch.device = ...,
        processing_device: str | torch.device = ...,
        video_storage_device: str | torch.device = ...,
        max_vision_features_cache_size: int = ...,
        dtype: torch.dtype = ...,
    ):  # -> Sam2VideoInferenceSession:
        ...
    def add_inputs_to_inference_session(
        self,
        inference_session: Sam2VideoInferenceSession,
        frame_idx: int,
        obj_ids: list[int] | int,
        input_points: list[list[list[list[float]]]] | torch.Tensor | None = ...,
        input_labels: list[list[list[int]]] | torch.Tensor | None = ...,
        input_boxes: list[list[list[float]]] | torch.Tensor | None = ...,
        input_masks: np.ndarray
        | torch.Tensor
        | list[np.ndarray]
        | list[torch.Tensor]
        | None = ...,
        original_size: tuple[int, int] | None = ...,
        clear_old_inputs: bool = ...,
    ) -> Sam2VideoInferenceSession: ...
    def process_new_points_or_boxes_for_video_frame(
        self,
        inference_session: Sam2VideoInferenceSession,
        frame_idx: int,
        obj_ids: list[int],
        input_points: list[list[list[list[float]]]] | torch.Tensor | None = ...,
        input_labels: list[list[list[int]]] | torch.Tensor | None = ...,
        input_boxes: list[list[list[float]]] | torch.Tensor | None = ...,
        original_size: tuple[int, int] | None = ...,
        clear_old_inputs: bool = ...,
    ) -> Sam2VideoInferenceSession: ...
    def process_new_mask_for_video_frame(
        self,
        inference_session: Sam2VideoInferenceSession,
        frame_idx: int,
        obj_ids: list[int],
        input_masks: np.ndarray | torch.Tensor | list[np.ndarray] | list[torch.Tensor],
    ):  # -> None:
        ...

__all__ = ["Sam2VideoProcessor"]
