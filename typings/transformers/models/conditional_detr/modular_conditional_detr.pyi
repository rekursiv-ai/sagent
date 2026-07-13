from transformers.models.detr.image_processing_detr_fast import DetrImageProcessorFast

from ...utils import TensorType

logger = ...

class ConditionalDetrImageProcessorFast(DetrImageProcessorFast):
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
    def post_process_segmentation(self): ...
    def post_process_instance(self): ...
    def post_process_panoptic(self): ...

__all__ = ["ConditionalDetrImageProcessorFast"]
