from transformers.models.maskformer.image_processing_maskformer_fast import (
    MaskFormerImageProcessorFast,
)

import torch

logger = ...

class Mask2FormerImageProcessorFast(MaskFormerImageProcessorFast):
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
    def post_process_segmentation(self): ...

__all__ = ["Mask2FormerImageProcessorFast"]
