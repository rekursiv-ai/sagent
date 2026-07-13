from functools import lru_cache

from .tokenization_markuplm import (
    MARKUPLM_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING,
    MarkupLMTokenizer,
)
from ...file_utils import PaddingStrategy, TensorType, add_end_docstrings
from ...tokenization_utils_base import (
    ENCODE_KWARGS_DOCSTRING,
    BatchEncoding,
    PreTokenizedInput,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)
from ...tokenization_utils_fast import PreTrainedTokenizerFast

"""
Fast tokenization class for MarkupLM. It overwrites 2 methods of the slow tokenizer class, namely _batch_encode_plus
and _encode_plus, in which the Rust tokenizer is used.
"""
logger = ...
VOCAB_FILES_NAMES = ...

@lru_cache
def bytes_to_unicode():  # -> dict[int, str]:
    ...
def get_pairs(word):  # -> set[Any]:
    ...

class MarkupLMTokenizerFast(PreTrainedTokenizerFast):
    vocab_files_names = ...
    slow_tokenizer_class = MarkupLMTokenizer
    def __init__(
        self,
        vocab_file,
        merges_file,
        tags_dict,
        tokenizer_file=...,
        errors=...,
        bos_token=...,
        eos_token=...,
        sep_token=...,
        cls_token=...,
        unk_token=...,
        pad_token=...,
        mask_token=...,
        add_prefix_space=...,
        max_depth=...,
        max_width=...,
        pad_width=...,
        pad_token_label=...,
        only_label_first_subword=...,
        trim_offsets=...,
        **kwargs,
    ) -> None: ...
    def get_xpath_seq(self, xpath):  # -> tuple[list[Any], list[Any]]:
        ...
    @add_end_docstrings(
        ENCODE_KWARGS_DOCSTRING, MARKUPLM_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING
    )
    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput],
        text_pair: PreTokenizedInput | list[PreTokenizedInput] | None = ...,
        xpaths: list[list[int]] | list[list[list[int]]] | None = ...,
        node_labels: list[int] | list[list[int]] | None = ...,
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
    @add_end_docstrings(
        ENCODE_KWARGS_DOCSTRING, MARKUPLM_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING
    )
    def batch_encode_plus(
        self,
        batch_text_or_text_pairs: list[TextInput]
        | list[TextInputPair]
        | list[PreTokenizedInput],
        is_pair: bool | None = ...,
        xpaths: list[list[list[int]]] | None = ...,
        node_labels: list[int] | list[list[int]] | None = ...,
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
    @add_end_docstrings(
        ENCODE_KWARGS_DOCSTRING, MARKUPLM_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING
    )
    def encode_plus(
        self,
        text: TextInput | PreTokenizedInput,
        text_pair: PreTokenizedInput | None = ...,
        xpaths: list[list[int]] | None = ...,
        node_labels: list[int] | None = ...,
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
    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = ...
    ) -> list[int]: ...
    def create_token_type_ids_from_sequences(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = ...
    ) -> list[int]: ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str]: ...

__all__ = ["MarkupLMTokenizerFast"]
