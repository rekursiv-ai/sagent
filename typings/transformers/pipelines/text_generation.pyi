from typing import Any, overload

import enum

from .base import Pipeline, build_pipeline_init_args
from ..utils import add_end_docstrings

type ChatType = list[dict[str, str]]

class ReturnType(enum.Enum):
    TENSORS = ...
    NEW_TEXT = ...
    FULL_TEXT = ...

class Chat:
    def __init__(self, messages: dict) -> None: ...

@add_end_docstrings(build_pipeline_init_args(has_tokenizer=True))
class TextGenerationPipeline(Pipeline):
    XL_PREFIX = ...
    _pipeline_calls_generate = ...
    _load_processor = ...
    _load_image_processor = ...
    _load_feature_extractor = ...
    _load_tokenizer = ...
    _default_generation_config = ...
    def __init__(self, *args, **kwargs) -> None: ...
    @overload
    def __call__(self, text_inputs: str, **kwargs: Any) -> list[dict[str, str]]: ...
    @overload
    def __call__(
        self, text_inputs: list[str], **kwargs: Any
    ) -> list[list[dict[str, str]]]: ...
    @overload
    def __call__(
        self, text_inputs: ChatType, **kwargs: Any
    ) -> list[dict[str, ChatType]]: ...
    @overload
    def __call__(
        self, text_inputs: list[ChatType], **kwargs: Any
    ) -> list[list[dict[str, ChatType]]]: ...
    def __call__(
        self, text_inputs, **kwargs
    ):  # -> list[Any] | PipelineIterator | Generator[Any, Any, None] | Tensor | Any | None:
        ...
    def preprocess(
        self,
        prompt_text,
        prefix=...,
        handle_long_generation=...,
        add_special_tokens=...,
        truncation=...,
        padding=...,
        max_length=...,
        continue_final_message=...,
        tokenizer_encode_kwargs=...,
        **generate_kwargs,
    ):  # -> str | list[int] | list[str] | list[list[int]] | BatchEncoding | Any:
        ...
    def postprocess(
        self,
        model_outputs,
        return_type=...,
        clean_up_tokenization_spaces=...,
        continue_final_message=...,
        skip_special_tokens=...,
    ):  # -> list[Any]:
        ...
