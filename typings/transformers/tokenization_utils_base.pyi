from collections import UserDict
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, NamedTuple

import os

from tokenizers import (
    AddedToken,
    Encoding as EncodingFast,
)

import numpy as np

from .utils import (
    ExplicitEnum,
    PaddingStrategy,
    PushToHubMixin,
    TensorType,
    add_end_docstrings,
    is_tokenizers_available,
)

"""
Base classes common to both the slow and the fast tokenization classes: PreTrainedTokenizerBase (host all the user
fronting encoding methods) Special token mixing (host the special tokens logic) and BatchEncoding (wrap the dictionary
of output with special method for the Fast tokenizers)
"""

def import_protobuf_decode_error(error_message=...):  # -> type[DecodeError]:
    ...
def flatten(arr: list):  # -> list[Any]:
    ...

if is_tokenizers_available(): ...
else:
    @dataclass(frozen=False, eq=True)
    class AddedToken:
        def __init__(
            self,
            content: str,
            single_word=...,
            lstrip=...,
            rstrip=...,
            special=...,
            normalized=...,
        ) -> None: ...
        def __getstate__(self):  # -> dict[str, Any]:
            ...

logger = ...
VERY_LARGE_INTEGER = ...
LARGE_INTEGER = ...
TextInput = str
type PreTokenizedInput = list[str]
type EncodedInput = list[int]
type TextInputPair = tuple[str, str]
type PreTokenizedInputPair = tuple[list[str], list[str]]
type EncodedInputPair = tuple[list[int], list[int]]
type AudioInput = np.ndarray | torch.Tensor | list[np.ndarray] | list[torch.Tensor]
SPECIAL_TOKENS_MAP_FILE = ...
ADDED_TOKENS_FILE = ...
TOKENIZER_CONFIG_FILE = ...
FULL_TOKENIZER_FILE = ...
_re_tokenizer_file = ...

class TruncationStrategy(ExplicitEnum):
    ONLY_FIRST = ...
    ONLY_SECOND = ...
    LONGEST_FIRST = ...
    DO_NOT_TRUNCATE = ...

class CharSpan(NamedTuple):
    start: int
    end: int

class TokenSpan(NamedTuple):
    start: int
    end: int

class BatchEncoding(UserDict):
    def __init__(
        self,
        data: dict[str, Any] | None = ...,
        encoding: EncodingFast | Sequence[EncodingFast] | None = ...,
        tensor_type: None | str | TensorType = ...,
        prepend_batch_axis: bool = ...,
        n_sequences: int | None = ...,
    ) -> None: ...
    @property
    def n_sequences(self) -> int | None: ...
    @property
    def is_fast(self) -> bool: ...
    def __getitem__(self, item: int | str) -> EncodingFast: ...
    def __getattr__(self, item: str): ...
    def __getstate__(
        self,
    ):  # -> dict[str, dict[Any, Any] | list[Encoding] | Sequence[Encoding] | Any | None]:
        ...
    def __setstate__(self, state):  # -> None:
        ...
    @property
    def encodings(self) -> list[EncodingFast] | None: ...
    def tokens(self, batch_index: int = ...) -> list[str]: ...
    def sequence_ids(self, batch_index: int = ...) -> list[int | None]: ...
    def words(self, batch_index: int = ...) -> list[int | None]: ...
    def word_ids(self, batch_index: int = ...) -> list[int | None]: ...
    def token_to_sequence(
        self, batch_or_token_index: int, token_index: int | None = ...
    ) -> int: ...
    def token_to_word(
        self, batch_or_token_index: int, token_index: int | None = ...
    ) -> int: ...
    def word_to_tokens(
        self,
        batch_or_word_index: int,
        word_index: int | None = ...,
        sequence_index: int = ...,
    ) -> TokenSpan | None: ...
    def token_to_chars(
        self, batch_or_token_index: int, token_index: int | None = ...
    ) -> CharSpan | None: ...
    def char_to_token(
        self,
        batch_or_char_index: int,
        char_index: int | None = ...,
        sequence_index: int = ...,
    ) -> int: ...
    def word_to_chars(
        self,
        batch_or_word_index: int,
        word_index: int | None = ...,
        sequence_index: int = ...,
    ) -> CharSpan: ...
    def char_to_word(
        self,
        batch_or_char_index: int,
        char_index: int | None = ...,
        sequence_index: int = ...,
    ) -> int: ...
    def convert_to_tensors(
        self,
        tensor_type: str | TensorType | None = ...,
        prepend_batch_axis: bool = ...,
    ):  # -> Self:
        ...
    def to(
        self, device: str | torch.device, *, non_blocking: bool = ...
    ) -> BatchEncoding: ...

class SpecialTokensMixin:
    SPECIAL_TOKENS_ATTRIBUTES = ...
    def __init__(self, verbose=..., **kwargs) -> None: ...
    def sanitize_special_tokens(self) -> int: ...
    def add_special_tokens(
        self,
        special_tokens_dict: dict[str, str | AddedToken | Sequence[str | AddedToken]],
        replace_additional_special_tokens=...,
    ) -> int: ...
    def add_tokens(
        self,
        new_tokens: str | AddedToken | Sequence[str | AddedToken],
        special_tokens: bool = ...,
    ) -> int: ...
    @property
    def pad_token_type_id(self) -> int: ...
    def __setattr__(self, key, value):  # -> None:
        ...
    def __getattr__(self, key):  # -> str | list[str] | None:
        ...
    @property
    def special_tokens_map(self) -> dict[str, str | list[str]]: ...
    @property
    def special_tokens_map_extended(
        self,
    ) -> dict[str, str | AddedToken | list[str | AddedToken]]: ...
    @property
    def all_special_tokens_extended(self) -> list[str | AddedToken]: ...
    @property
    def all_special_tokens(self) -> list[str]: ...
    @property
    def all_special_ids(self) -> list[int]: ...

ENCODE_KWARGS_DOCSTRING = ...
ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING = ...
INIT_TOKENIZER_DOCSTRING = ...

@add_end_docstrings(INIT_TOKENIZER_DOCSTRING)
class PreTrainedTokenizerBase(SpecialTokensMixin, PushToHubMixin):
    vocab_files_names: dict[str, str] = ...
    pretrained_vocab_files_map: dict[str, dict[str, str]] = ...
    _auto_class: str | None = ...
    model_input_names: list[str] = ...
    padding_side: str = ...
    truncation_side: str = ...
    slow_tokenizer_class = ...
    def __init__(self, **kwargs) -> None: ...
    @property
    def max_len_single_sentence(self) -> int: ...
    @property
    def max_len_sentences_pair(self) -> int: ...
    @max_len_single_sentence.setter
    def max_len_single_sentence(self, value) -> int: ...
    @max_len_sentences_pair.setter
    def max_len_sentences_pair(self, value) -> int: ...
    @property
    def added_tokens_decoder(self) -> dict[int, AddedToken]: ...
    def __len__(self) -> int: ...
    def get_vocab(self) -> dict[str, int]: ...
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]] | list[list[dict[str, str]]],
        tools: list[dict | Callable] | None = ...,
        documents: list[dict[str, str]] | None = ...,
        chat_template: str | None = ...,
        add_generation_prompt: bool = ...,
        continue_final_message: bool = ...,
        tokenize: bool = ...,
        padding: bool | str | PaddingStrategy = ...,
        truncation: bool = ...,
        max_length: int | None = ...,
        return_tensors: str | TensorType | None = ...,
        return_dict: bool = ...,
        return_assistant_tokens_mask: bool = ...,
        tokenizer_kwargs: dict[str, Any] | None = ...,
        **kwargs,
    ) -> str | list[int] | list[str] | list[list[int]] | BatchEncoding: ...
    def encode_message_with_chat_template(
        self,
        message: dict[str, str],
        conversation_history: list[dict[str, str]] | None = ...,
        **kwargs,
    ) -> list[int]: ...
    def get_chat_template(
        self, chat_template: str | None = ..., tools: list[dict] | None = ...
    ) -> str: ...
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        *init_inputs,
        cache_dir: str | os.PathLike | None = ...,
        force_download: bool = ...,
        local_files_only: bool = ...,
        token: str | bool | None = ...,
        revision: str = ...,
        trust_remote_code=...,
        **kwargs,
    ): ...
    @classmethod
    def convert_added_tokens(
        cls, obj: AddedToken | Any, save=..., add_type_field=...
    ): ...
    def save_chat_templates(
        self,
        save_directory: str | os.PathLike,
        tokenizer_config: dict,
        filename_prefix: str | None,
        save_jinja_files: bool,
    ):  # -> tuple[dict[Any, Any], list[Any]]:
        ...
    def save_pretrained(
        self,
        save_directory: str | os.PathLike,
        legacy_format: bool | None = ...,
        filename_prefix: str | None = ...,
        push_to_hub: bool = ...,
        **kwargs,
    ) -> tuple[str, ...]: ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str, ...]: ...
    def tokenize(
        self,
        text: str,
        pair: str | None = ...,
        add_special_tokens: bool = ...,
        **kwargs,
    ) -> list[str]: ...
    @add_end_docstrings(
        ENCODE_KWARGS_DOCSTRING,
        ...,
        ...,
    )
    def encode(
        self,
        text: TextInput | PreTokenizedInput | EncodedInput,
        text_pair: TextInput | PreTokenizedInput | EncodedInput | None = ...,
        add_special_tokens: bool = ...,
        padding: bool | str | PaddingStrategy = ...,
        truncation: bool | str | TruncationStrategy | None = ...,
        max_length: int | None = ...,
        stride: int = ...,
        padding_side: str | None = ...,
        return_tensors: str | TensorType | None = ...,
        **kwargs,
    ) -> list[int]: ...
    def num_special_tokens_to_add(self, pair: bool = ...) -> int: ...
    @add_end_docstrings(
        ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING
    )
    def __call__(
        self,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput]
        | None = ...,
        text_pair: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput]
        | None = ...,
        text_target: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput]
        | None = ...,
        text_pair_target: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput]
        | None = ...,
        add_special_tokens: bool = ...,
        padding: bool | str | PaddingStrategy = ...,
        truncation: bool | str | TruncationStrategy | None = ...,
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
    @add_end_docstrings(
        ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING
    )
    def encode_plus(
        self,
        text: TextInput | PreTokenizedInput | EncodedInput,
        text_pair: TextInput | PreTokenizedInput | EncodedInput | None = ...,
        add_special_tokens: bool = ...,
        padding: bool | str | PaddingStrategy = ...,
        truncation: bool | str | TruncationStrategy | None = ...,
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
    @add_end_docstrings(
        ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING
    )
    def batch_encode_plus(
        self,
        batch_text_or_text_pairs: list[TextInput]
        | list[TextInputPair]
        | list[PreTokenizedInput]
        | list[PreTokenizedInputPair]
        | list[EncodedInput]
        | list[EncodedInputPair],
        add_special_tokens: bool = ...,
        padding: bool | str | PaddingStrategy = ...,
        truncation: bool | str | TruncationStrategy | None = ...,
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
        split_special_tokens: bool = ...,
        **kwargs,
    ) -> BatchEncoding: ...
    def pad(
        self,
        encoded_inputs: BatchEncoding
        | list[BatchEncoding]
        | dict[str, EncodedInput]
        | dict[str, list[EncodedInput]]
        | list[dict[str, EncodedInput]],
        padding: bool | str | PaddingStrategy = ...,
        max_length: int | None = ...,
        pad_to_multiple_of: int | None = ...,
        padding_side: str | None = ...,
        return_attention_mask: bool | None = ...,
        return_tensors: str | TensorType | None = ...,
        verbose: bool = ...,
    ) -> BatchEncoding: ...
    def create_token_type_ids_from_sequences(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = ...
    ) -> list[int]: ...
    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = ...
    ) -> list[int]: ...
    @add_end_docstrings(
        ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING
    )
    def prepare_for_model(
        self,
        ids: list[int],
        pair_ids: list[int] | None = ...,
        add_special_tokens: bool = ...,
        padding: bool | str | PaddingStrategy = ...,
        truncation: bool | str | TruncationStrategy | None = ...,
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
        pair_ids: list[int] | None = ...,
        num_tokens_to_remove: int = ...,
        truncation_strategy: str | TruncationStrategy = ...,
        stride: int = ...,
    ) -> tuple[list[int], list[int], list[int]]: ...
    def convert_tokens_to_string(self, tokens: list[str]) -> str: ...
    def batch_decode(
        self,
        sequences: list[int] | list[list[int]] | np.ndarray | torch.Tensor | tf.Tensor,
        skip_special_tokens: bool = ...,
        clean_up_tokenization_spaces: bool | None = ...,
        **kwargs,
    ) -> list[str]: ...
    def decode(
        self,
        token_ids: int | list[int] | np.ndarray | torch.Tensor,
        skip_special_tokens: bool = ...,
        clean_up_tokenization_spaces: bool | None = ...,
        **kwargs,
    ) -> str: ...
    def get_special_tokens_mask(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = ...,
        already_has_special_tokens: bool = ...,
    ) -> list[int]: ...
    @staticmethod
    def clean_up_tokenization(out_string: str) -> str: ...
    @contextmanager
    def as_target_tokenizer(self):  # -> Generator[None, Any, None]:
        ...
    @classmethod
    def register_for_auto_class(cls, auto_class=...):  # -> None:
        ...
    def prepare_seq2seq_batch(
        self,
        src_texts: list[str],
        tgt_texts: list[str] | None = ...,
        max_length: int | None = ...,
        max_target_length: int | None = ...,
        padding: str = ...,
        return_tensors: str | None = ...,
        truncation: bool = ...,
        **kwargs,
    ) -> BatchEncoding: ...

def get_fast_tokenizer_file(tokenization_files: list[str]) -> str: ...
