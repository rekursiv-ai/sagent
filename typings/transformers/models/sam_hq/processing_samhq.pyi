from ...image_utils import ImageInput
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import (
    AudioInput,
    BatchEncoding,
    PreTokenizedInput,
    TextInput,
)
from ...video_utils import VideoInput

"""
Processor class for SAMHQ.
"""

class SamHQImagesKwargs(ImagesKwargs):
    segmentation_maps: ImageInput | None
    input_points: list[list[float]] | None
    input_labels: list[list[int]] | None
    input_boxes: list[list[list[float]]] | None
    point_pad_value: int | None

class SamHQProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: SamHQImagesKwargs
    _defaults = ...

class SamHQProcessor(ProcessorMixin):
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
        **kwargs: Unpack[SamHQProcessorKwargs],
    ) -> BatchEncoding: ...
    @property
    def model_input_names(self):  # -> list[Any]:
        ...
    def post_process_masks(self, *args, **kwargs): ...

__all__ = ["SamHQProcessor"]
