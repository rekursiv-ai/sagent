from num2words import num2words

from ...image_utils import ImageInput
from ...processing_utils import (
    AllKwargsForChatTemplate,
    ImagesKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
)
from ...tokenization_utils_base import BatchEncoding, PreTokenizedInput, TextInput
from ...utils import is_num2words_available
from ...video_utils import VideoInput

"""
Processor class for SmolVLM.
"""
logger = ...
if is_num2words_available(): ...
else:
    num2words = ...
DEFAULT_CHAT_TEMPLATE = ...

def get_image_prompt_string(
    image_rows,
    image_cols,
    image_seq_len,
    fake_token_around_image,
    image_token,
    global_image_token,
): ...

class SmolVLMImagesKwargs(ImagesKwargs, total=False):
    return_row_col_info: bool | None
    max_image_size: dict[str, int] | None

class SmolVLMProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: SmolVLMImagesKwargs
    _defaults = ...

class SmolVLMProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    video_processor_class = ...
    tokenizer_class = ...
    def __init__(
        self,
        image_processor,
        tokenizer,
        video_processor,
        image_seq_len: int = ...,
        chat_template: str | None = ...,
        **kwargs,
    ) -> None: ...
    def expand_text_with_image_tokens(
        self, text, image_rows, image_cols
    ):  # -> list[Any]:
        ...
    def expand_text_with_video_tokens(self, text, video_inputs):  # -> list[Any]:
        ...
    def __call__(
        self,
        images: ImageInput | list[ImageInput] | list[list[ImageInput]] = ...,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput] = ...,
        audio=...,
        videos: VideoInput | None = ...,
        **kwargs: Unpack[SmolVLMProcessorKwargs],
    ) -> BatchEncoding: ...
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]] | list[list[dict[str, str]]],
        chat_template: str | None = ...,
        **kwargs: Unpack[AllKwargsForChatTemplate],
    ) -> str: ...

__all__ = ["SmolVLMProcessor"]
