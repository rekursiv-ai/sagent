from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...video_utils import VideoInput

class InternVLImagesKwargs(ImagesKwargs, total=False):
    crop_to_patches: bool | None
    min_patches: int | None
    max_patches: int | None

class InternVLProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: InternVLImagesKwargs
    _defaults = ...

class InternVLProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    video_processor_class = ...
    tokenizer_class = ...
    def __init__(
        self,
        image_processor=...,
        tokenizer=...,
        video_processor=...,
        image_seq_length: int = ...,
        chat_template=...,
        **kwargs,
    ) -> None: ...
    def __call__(
        self,
        images: ImageInput | None = ...,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput]
        | None = ...,
        audio=...,
        videos: VideoInput | None = ...,
        **kwargs: Unpack[InternVLProcessorKwargs],
    ) -> BatchFeature: ...
    @property
    def model_input_names(self): ...

__all__ = ["InternVLProcessor"]
