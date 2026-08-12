import numpy as np

from .tokenization_codegen import CodeGenTokenizer
from ...tokenization_utils_fast import PreTrainedTokenizerFast

"""Tokenization classes for OpenAI GPT."""
logger = ...
VOCAB_FILES_NAMES = ...

class CodeGenTokenizerFast(PreTrainedTokenizerFast):
    slow_tokenizer_class = CodeGenTokenizer
    def __init__(
        self,
        vocab_file=...,
        merges_file=...,
        tokenizer_file=...,
        unk_token=...,
        bos_token=...,
        eos_token=...,
        add_prefix_space=...,
        return_token_type_ids=...,
        **kwargs,
    ) -> None: ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str]: ...
    def decode(
        self,
        token_ids: int | list[int] | np.ndarray | torch.Tensor | tf.Tensor,
        skip_special_tokens: bool = ...,
        clean_up_tokenization_spaces: bool | None = ...,
        truncate_before_pattern: list[str] | None = ...,
        **kwargs,
    ) -> str: ...
    def truncate(self, completion, truncate_before_pattern): ...

__all__ = ["CodeGenTokenizerFast"]
