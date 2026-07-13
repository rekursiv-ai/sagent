from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput

class AyaVisionImagesKwargs(ImagesKwargs, total=False):
    crop_to_patches: bool | None
    min_patches: int | None
    max_patches: int | None

class AyaVisionProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: AyaVisionImagesKwargs
    _defaults = ...

class AyaVisionProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(
        self,
        image_processor=...,
        tokenizer=...,
        patch_size: int = ...,
        img_size: int = ...,
        image_token=...,
        downsample_factor: int = ...,
        start_of_img_token=...,
        end_of_img_token=...,
        img_patch_token=...,
        img_line_break_token=...,
        tile_token=...,
        tile_global_token=...,
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
        videos=...,
        **kwargs: Unpack[AyaVisionProcessorKwargs],
    ) -> BatchFeature: ...

__all__ = ["AyaVisionProcessor"]
