from .tokenization_layoutlmv2 import (
    LAYOUTLMV2_ENCODE_KWARGS_DOCSTRING,
    LAYOUTLMV2_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING,
    LayoutLMv2Tokenizer,
)
from ...tokenization_utils_base import (
    BatchEncoding,
    PaddingStrategy,
    PreTokenizedInput,
    TensorType,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import add_end_docstrings

"""
Fast tokenization class for LayoutLMv2. It overwrites 2 methods of the slow tokenizer class, namely _batch_encode_plus
and _encode_plus, in which the Rust tokenizer is used.
"""
logger = ...
VOCAB_FILES_NAMES = ...

class LayoutLMv2TokenizerFast(PreTrainedTokenizerFast):
    vocab_files_names = ...
    slow_tokenizer_class = LayoutLMv2Tokenizer
    def __init__(
        self,
        vocab_file=...,
        tokenizer_file=...,
        do_lower_case=...,
        unk_token=...,
        sep_token=...,
        pad_token=...,
        cls_token=...,
        mask_token=...,
        cls_token_box=...,
        sep_token_box=...,
        pad_token_box=...,
        pad_token_label=...,
        only_label_first_subword=...,
        tokenize_chinese_chars=...,
        strip_accents=...,
        **kwargs,
    ) -> None: ...
    @add_end_docstrings(
        LAYOUTLMV2_ENCODE_KWARGS_DOCSTRING,
        LAYOUTLMV2_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING,
    )
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
    @add_end_docstrings(
        LAYOUTLMV2_ENCODE_KWARGS_DOCSTRING,
        LAYOUTLMV2_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING,
    )
    def batch_encode_plus(
        self,
        batch_text_or_text_pairs: list[TextInput]
        | list[TextInputPair]
        | list[PreTokenizedInput],
        is_pair: bool | None = ...,
        boxes: list[list[list[int]]] | None = ...,
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
    @add_end_docstrings(
        LAYOUTLMV2_ENCODE_KWARGS_DOCSTRING,
        LAYOUTLMV2_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING,
    )
    def encode_plus(
        self,
        text: TextInput | PreTokenizedInput,
        text_pair: PreTokenizedInput | None = ...,
        boxes: list[list[int]] | None = ...,
        word_labels: list[int] | None = ...,
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
        self, token_ids_0, token_ids_1=...
    ):  # -> list[str | list[str] | Any | None]:
        ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str]: ...

__all__ = ["LayoutLMv2TokenizerFast"]
