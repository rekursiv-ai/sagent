from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput

class Gemma3ImagesKwargs(ImagesKwargs):
    do_pan_and_scan: bool | None
    pan_and_scan_min_crop_size: int | None
    pan_and_scan_max_num_crops: int | None
    pan_and_scan_min_ratio_to_activate: float | None
    do_convert_rgb: bool | None

class Gemma3ProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Gemma3ImagesKwargs
    _defaults = ...

class Gemma3Processor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(
        self,
        image_processor,
        tokenizer,
        chat_template=...,
        image_seq_length: int = ...,
        **kwargs,
    ) -> None: ...
    def __call__(
        self,
        images: ImageInput | None = ...,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput] = ...,
        videos=...,
        audio=...,
        **kwargs: Unpack[Gemma3ProcessorKwargs],
    ) -> BatchFeature: ...
    @property
    def model_input_names(self):  # -> list[Any]:
        ...

__all__ = ["Gemma3Processor"]
