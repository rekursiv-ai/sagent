from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput

class Cohere2VisionImagesKwargs(ImagesKwargs, total=False):
    max_patches: int | None

class Cohere2VisionProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Cohere2VisionImagesKwargs
    _defaults = ...

class Cohere2VisionProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(
        self, image_processor=..., tokenizer=..., chat_template=..., **kwargs
    ) -> None: ...
    def __call__(
        self,
        images: ImageInput | None = ...,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput]
        | None = ...,
        **kwargs: Unpack[Cohere2VisionProcessorKwargs],
    ) -> BatchFeature: ...
    def batch_decode(self, *args, **kwargs): ...
    def decode(self, *args, **kwargs): ...
    @property
    def model_input_names(self):  # -> list[Any]:
        ...

__all__ = ["Cohere2VisionProcessor"]
