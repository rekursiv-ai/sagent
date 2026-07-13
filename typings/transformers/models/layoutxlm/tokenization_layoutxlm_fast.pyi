from .tokenization_layoutxlm import LayoutXLMTokenizer
from ...tokenization_utils_base import (
    BatchEncoding,
    PreTokenizedInput,
    TextInput,
    TruncationStrategy,
)
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import (
    PaddingStrategy,
    TensorType,
    add_end_docstrings,
    is_sentencepiece_available,
)

"""Tokenization classes for LayoutXLM model."""
if is_sentencepiece_available(): ...
else:
    LayoutXLMTokenizer = ...
logger = ...
LAYOUTXLM_ENCODE_KWARGS_DOCSTRING = ...

class LayoutXLMTokenizerFast(PreTrainedTokenizerFast):
    vocab_files_names = ...
    model_input_names = ...
    slow_tokenizer_class = ...
    def __init__(
        self,
        vocab_file=...,
        tokenizer_file=...,
        bos_token=...,
        eos_token=...,
        sep_token=...,
        cls_token=...,
        unk_token=...,
        pad_token=...,
        mask_token=...,
        cls_token_box=...,
        sep_token_box=...,
        pad_token_box=...,
        pad_token_label=...,
        only_label_first_subword=...,
        **kwargs,
    ) -> None: ...
    @add_end_docstrings(LAYOUTXLM_ENCODE_KWARGS_DOCSTRING)
    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput],
        text_pair: PreTokenizedInput | list[PreTokenizedInput] | None = ...,
        boxes: list[list[int]] | list[list[list[int]]] | None = ...,
        word_labels: list[int] | list[list[int]] | None = ...,
        add_special_tokens: bool = ...,
        padding: bool | str | PaddingStrategy = ...,
        truncation: bool | str | TruncationStrategy = ...,
        max_length: int | None = ...,
        stride: int = ...,
        pad_to_multiple_of: int | None = ...,
        padding_side: str | None = ...,
        return_tensors: str | TensorType | None = ...,
        return_token_type_ids: bool | None = ...,
        return_attention_mask: bool | None = ...,
        return_overflowing_tokens: bool = ...,
        return_special_tokens_mask: bool = ...,
        return_offsets_mapping: bool = ...,
        return_length: bool = ...,
        verbose: bool = ...,
        **kwargs,
    ) -> BatchEncoding: ...
    def tokenize(
        self,
        text: str,
        pair: str | None = ...,
        add_special_tokens: bool = ...,
        **kwargs,
    ) -> list[str]: ...
    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = ...
    ) -> list[int]: ...
    def create_token_type_ids_from_sequences(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = ...
    ) -> list[int]: ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str]: ...

__all__ = ["LayoutXLMTokenizerFast"]
