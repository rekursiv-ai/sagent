from functools import lru_cache

from ...file_utils import PaddingStrategy, TensorType, add_end_docstrings
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import (
    ENCODE_KWARGS_DOCSTRING,
    BatchEncoding,
    PreTokenizedInput,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)

"""Tokenization class for MarkupLM."""
logger = ...
VOCAB_FILES_NAMES = ...
MARKUPLM_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING = ...

@lru_cache
def bytes_to_unicode():  # -> dict[int, str]:
    ...
def get_pairs(word):  # -> set[Any]:
    ...

class MarkupLMTokenizer(PreTrainedTokenizer):
    vocab_files_names = ...
    def __init__(
        self,
        vocab_file,
        merges_file,
        tags_dict,
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
        **kwargs,
    ) -> None: ...
    def get_xpath_seq(self, xpath):  # -> tuple[list[Any], list[Any]]:
        ...
    @property
    def vocab_size(self):  # -> int:
        ...
    def get_vocab(self):  # -> Any:
        ...
    def bpe(self, token):  # -> LiteralString:
        ...
    def convert_tokens_to_string(self, tokens):  # -> str:
        ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str]: ...
    def prepare_for_tokenization(
        self, text, is_split_into_words=..., **kwargs
    ):  # -> tuple[str, dict[str, Any]]:
        ...
    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = ...
    ) -> list[int]: ...
    def build_xpath_tags_with_special_tokens(
        self, xpath_tags_0: list[int], xpath_tags_1: list[int] | None = ...
    ) -> list[int]: ...
    def build_xpath_subs_with_special_tokens(
        self, xpath_subs_0: list[int], xpath_subs_1: list[int] | None = ...
    ) -> list[int]: ...
    def get_special_tokens_mask(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = ...,
        already_has_special_tokens: bool = ...,
    ) -> list[int]: ...
    def create_token_type_ids_from_sequences(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = ...
    ) -> list[int]: ...
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
    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING)
    def encode(
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
    ) -> list[int]: ...
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
    @add_end_docstrings(
        ENCODE_KWARGS_DOCSTRING, MARKUPLM_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING
    )
    def prepare_for_model(
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
        prepend_batch_axis: bool = ...,
        **kwargs,
    ) -> BatchEncoding: ...
    def truncate_sequences(
        self,
        ids: list[int],
        xpath_tags_seq: list[list[int]],
        xpath_subs_seq: list[list[int]],
        pair_ids: list[int] | None = ...,
        pair_xpath_tags_seq: list[list[int]] | None = ...,
        pair_xpath_subs_seq: list[list[int]] | None = ...,
        labels: list[int] | None = ...,
        num_tokens_to_remove: int = ...,
        truncation_strategy: str | TruncationStrategy = ...,
        stride: int = ...,
    ) -> tuple[list[int], list[int], list[int]]: ...

__all__ = ["MarkupLMTokenizer"]
