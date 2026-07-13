from ...image_utils import ImageInput
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin
from ...tokenization_utils_base import (
    AudioInput,
    BatchEncoding,
    PreTokenizedInput,
    TextInput,
)
from ...video_utils import VideoInput

"""
Processor class for SAM.
"""

class SamImagesKwargs(ImagesKwargs):
    segmentation_maps: ImageInput | None
    input_points: list[list[float]] | None
    input_labels: list[list[int]] | None
    input_boxes: list[list[list[float]]] | None
    point_pad_value: int | None

class SamProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: SamImagesKwargs
    _defaults = ...

class SamProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    def __init__(self, image_processor) -> None: ...
    def __call__(
        self,
        images: ImageInput | None = ...,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput]
        | None = ...,
        audio: AudioInput | None = ...,
        video: VideoInput | None = ...,
        **kwargs,
    ) -> BatchEncoding: ...
    @property
    def model_input_names(self):  # -> list[Any]:
        ...
    def post_process_masks(self, *args, **kwargs): ...

__all__ = ["SamProcessor"]
