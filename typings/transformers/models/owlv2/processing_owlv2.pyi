from .modeling_owlv2 import (
    Owlv2ImageGuidedObjectDetectionOutput,
    Owlv2ObjectDetectionOutput,
)
from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import TensorType

"""
Image/Text processor class for OWLv2
"""

class Owlv2ImagesKwargs(ImagesKwargs, total=False):
    query_images: ImageInput | None

class Owlv2ProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Owlv2ImagesKwargs
    _defaults = ...

class Owlv2Processor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(self, image_processor, tokenizer, **kwargs) -> None: ...
    def __call__(
        self,
        images: ImageInput | None = ...,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput] = ...,
        audio=...,
        videos=...,
        **kwargs: Unpack[Owlv2ProcessorKwargs],
    ) -> BatchFeature: ...
    def post_process_object_detection(self, *args, **kwargs): ...
    def post_process_grounded_object_detection(
        self,
        outputs: Owlv2ObjectDetectionOutput,
        threshold: float = ...,
        target_sizes: TensorType | list[tuple] | None = ...,
        text_labels: list[list[str]] | None = ...,
    ): ...
    def post_process_image_guided_detection(
        self,
        outputs: Owlv2ImageGuidedObjectDetectionOutput,
        threshold: float = ...,
        nms_threshold: float = ...,
        target_sizes: TensorType | list[tuple] | None = ...,
    ): ...

__all__ = ["Owlv2Processor"]
