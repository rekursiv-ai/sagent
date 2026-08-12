from typing import Any

from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import (
    BatchEncoding,
    EncodedInput,
    PreTokenizedInput,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)
from ...utils import PaddingStrategy, TensorType, add_end_docstrings
from ...utils.import_utils import requires

"""Tokenization classes for UDOP model."""
logger = ...
SPIECE_UNDERLINE = ...
UDOP_ENCODE_KWARGS_DOCSTRING = ...
VOCAB_FILES_NAMES = ...

@requires(backends=("sentencepiece",))
class UdopTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        vocab_file,
        eos_token=...,
        unk_token=...,
        sep_token=...,
        pad_token=...,
        sep_token_box=...,
        pad_token_box=...,
        pad_token_label=...,
        only_label_first_subword=...,
        additional_special_tokens=...,
        sp_model_kwargs: dict[str, Any] | None = ...,
        legacy=...,
        add_prefix_space=...,
        **kwargs,
    ) -> None: ...
    @property
    def vocab_size(self):  # -> int:
        ...
    def get_vocab(self):  # -> dict[str, int]:
        ...
    def get_special_tokens_mask(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = ...,
        already_has_special_tokens: bool = ...,
    ) -> list[int]: ...
    def get_sentinel_tokens(self):  # -> list[str]:
        ...
    def get_sentinel_token_ids(self):  # -> list[int | list[int]]:
        ...
    def create_token_type_ids_from_sequences(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = ...
    ) -> list[int]: ...
    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = ...
    ) -> list[int]: ...
    def __getstate__(self):  # -> dict[str, Any]:
        ...
    def __setstate__(self, d):  # -> None:
        ...
    def tokenize(self, text: TextInput, **kwargs) -> list[str]: ...
    def convert_tokens_to_string(self, tokens): ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str]: ...
    @add_end_docstrings(UDOP_ENCODE_KWARGS_DOCSTRING)
    def __call__(
        self,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput] = ...,
        text_pair: PreTokenizedInput | list[PreTokenizedInput] | None = ...,
        boxes: list[list[int]] | list[list[list[int]]] | None = ...,
        word_labels: list[int] | list[list[int]] | None = ...,
        text_target: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput] = ...,
        text_pair_target: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput]
        | None = ...,
        **kwargs,
    ) -> BatchEncoding: ...
    def call_boxes(
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
    def batch_encode_plus_boxes(
        self,
        batch_text_or_text_pairs: list[TextInput]
        | list[TextInputPair]
        | list[PreTokenizedInput],
        is_pair: bool | None = ...,
        boxes: list[list[list[int]]] | None = ...,
        word_labels: list[list[int]] | None = ...,
        add_special_tokens: bool = ...,
        padding: bool | str | PaddingStrategy = ...,
        truncation: bool | str | TruncationStrategy = ...,
        max_length: int | None = ...,
        stride: int = ...,
        is_split_into_words: bool = ...,
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
    def encode_boxes(
        self,
        text: TextInput | PreTokenizedInput | EncodedInput,
        text_pair: TextInput | PreTokenizedInput | EncodedInput | None = ...,
        boxes: list[list[int]] | None = ...,
        word_labels: list[list[int]] | None = ...,
        add_special_tokens: bool = ...,
        padding: bool | str | PaddingStrategy = ...,
        truncation: bool | str | TruncationStrategy = ...,
        max_length: int | None = ...,
        stride: int = ...,
        return_tensors: str | TensorType | None = ...,
        **kwargs,
    ) -> list[int]: ...
    def encode_plus_boxes(
        self,
        text: TextInput | PreTokenizedInput,
        text_pair: PreTokenizedInput | None = ...,
        boxes: list[list[int]] | None = ...,
        word_labels: list[list[int]] | None = ...,
        add_special_tokens: bool = ...,
        padding: bool | str | PaddingStrategy = ...,
        truncation: bool | str | TruncationStrategy = ...,
        max_length: int | None = ...,
        stride: int = ...,
        is_split_into_words: bool = ...,
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
    @add_end_docstrings(UDOP_ENCODE_KWARGS_DOCSTRING)
    def prepare_for_model_boxes(
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
        prepend_batch_axis: bool = ...,
        **kwargs,
    ) -> BatchEncoding: ...
    def truncate_sequences(
        self,
        ids: list[int],
        token_boxes: list[list[int]],
        pair_ids: list[int] | None = ...,
        pair_token_boxes: list[list[int]] | None = ...,
        labels: list[int] | None = ...,
        num_tokens_to_remove: int = ...,
        truncation_strategy: str | TruncationStrategy = ...,
        stride: int = ...,
    ) -> tuple[list[int], list[int], list[int]]: ...

__all__ = ["UdopTokenizer"]
