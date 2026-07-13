from .modeling_owlvit import (
    OwlViTImageGuidedObjectDetectionOutput,
    OwlViTObjectDetectionOutput,
)
from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import TensorType

"""
Image/Text processor class for OWL-ViT
"""

class OwlViTImagesKwargs(ImagesKwargs, total=False):
    query_images: ImageInput | None

class OwlViTProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: OwlViTImagesKwargs
    _defaults = ...

class OwlViTProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(self, image_processor=..., tokenizer=..., **kwargs) -> None: ...
    def __call__(
        self,
        images: ImageInput | None = ...,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput] = ...,
        audio=...,
        videos=...,
        **kwargs: Unpack[OwlViTProcessorKwargs],
    ) -> BatchFeature: ...
    def post_process(self, *args, **kwargs): ...
    def post_process_object_detection(self, *args, **kwargs): ...
    def post_process_grounded_object_detection(
        self,
        outputs: OwlViTObjectDetectionOutput,
        threshold: float = ...,
        target_sizes: TensorType | list[tuple] | None = ...,
        text_labels: list[list[str]] | None = ...,
    ): ...
    def post_process_image_guided_detection(
        self,
        outputs: OwlViTImageGuidedObjectDetectionOutput,
        threshold: float = ...,
        nms_threshold: float = ...,
        target_sizes: TensorType | list[tuple] | None = ...,
    ): ...
    @property
    def feature_extractor_class(self):  # -> str:
        ...
    @property
    def feature_extractor(self): ...

__all__ = ["OwlViTProcessor"]
