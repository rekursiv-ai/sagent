from ...image_utils import ImageInput
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import BatchEncoding, PreTokenizedInput, TextInput

"""
Processor class for Idefics3.
"""
logger = ...

def is_url(val) -> bool: ...
def is_image_or_image_url(elem):  # -> bool:
    ...
def get_image_prompt_string(
    image_rows,
    image_cols,
    image_seq_len,
    fake_token_around_image,
    image_token,
    global_img_token,
): ...

class Idefics3ImagesKwargs(ImagesKwargs, total=False):
    return_row_col_info: bool | None
    max_image_size: dict[str, int] | None

class Idefics3ProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Idefics3ImagesKwargs
    _defaults = ...

class Idefics3Processor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(
        self,
        image_processor,
        tokenizer=...,
        image_seq_len: int = ...,
        chat_template: str | None = ...,
        **kwargs,
    ) -> None: ...
    def __call__(
        self,
        images: ImageInput | list[ImageInput] | list[list[ImageInput]] = ...,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput] = ...,
        audio=...,
        videos=...,
        image_seq_len: int | None = ...,
        **kwargs: Unpack[Idefics3ProcessorKwargs],
    ) -> BatchEncoding: ...

__all__ = ["Idefics3Processor"]
