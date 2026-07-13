from ...image_utils import ImageInput
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import BatchEncoding, TextInput

logger = ...

class Lfm2VlImagesKwargs(ImagesKwargs, total=False):
    downsample_factor: int | None
    do_image_splitting: bool | None
    min_tiles: int | None
    max_tiles: int | None
    use_thumbnail: bool | None
    min_image_tokens: int | None
    max_image_tokens: int | None
    encoder_patch_size: int | None
    tile_size: int | None
    max_pixels_tolerance: float | None
    patch_size: int | None
    do_pad: bool | None
    return_row_col_info: bool | None

class Lfm2VlProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Lfm2VlImagesKwargs
    _defaults = ...

class Lfm2VlProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(
        self,
        image_processor,
        tokenizer,
        chat_template: str | None = ...,
        use_image_special_tokens: bool | None = ...,
        **kwargs,
    ) -> None: ...
    def __call__(
        self,
        images: ImageInput | list[ImageInput] | list[list[ImageInput]] | None = ...,
        text: TextInput | list[TextInput] | None = ...,
        **kwargs: Unpack[Lfm2VlProcessorKwargs],
    ) -> BatchEncoding: ...
    def expand_text_with_placeholders(
        self,
        text: list[str],
        images: list[list[ImageInput]],
        image_rows: list[list[int]],
        image_cols: list[list[int]],
        image_sizes: list[list[int]],
        use_image_special_tokens: bool,
        **images_kwargs,
    ):  # -> list[Any]:
        ...
    def batch_decode(self, *args, **kwargs): ...
    def decode(self, *args, **kwargs): ...
    @property
    def model_input_names(self):  # -> list[Any]:
        ...

__all__ = ["Lfm2VlProcessor"]
