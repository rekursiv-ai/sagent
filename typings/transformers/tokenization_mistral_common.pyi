from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import overload

import os

from mistral_common.protocol.instruct.validator import ValidationMode
from transformers.tokenization_utils_base import (
    BatchEncoding,
    EncodedInput,
    TextInput,
    TruncationStrategy,
)
from transformers.utils import PaddingStrategy, TensorType, add_end_docstrings
from transformers.utils.hub import PushToHubMixin
from transformers.utils.import_utils import (
    requires,
)

import numpy as np
import torch

logger = ...
ENCODE_KWARGS_DOCSTRING = ...
ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING = ...

class MistralTokenizerType(StrEnum):
    spm = ...
    tekken = ...

@requires(backends=("mistral-common",))
class MistralCommonTokenizer(PushToHubMixin):
    model_input_names: list[str] = ...
    padding_side: str = ...
    truncation_side: str = ...
    def __init__(
        self,
        tokenizer_path: str | os.PathLike | Path,
        mode: ValidationMode = ...,
        model_max_length: int = ...,
        padding_side: str = ...,
        truncation_side: str = ...,
        model_input_names: list[str] | None = ...,
        clean_up_tokenization_spaces: bool = ...,
        **kwargs,
    ) -> None: ...
    @property
    def bos_token_id(self) -> int: ...
    @property
    def eos_token_id(self) -> int: ...
    @property
    def unk_token_id(self) -> int: ...
    @property
    def pad_token_id(self) -> int: ...
    @property
    def bos_token(self) -> str: ...
    @property
    def eos_token(self) -> str: ...
    @property
    def unk_token(self) -> str: ...
    @property
    def pad_token(self) -> str: ...
    @property
    def vocab_size(self) -> int: ...
    def get_vocab(self) -> dict[str, int]: ...
    def __len__(self):  # -> int:
        ...
    @add_end_docstrings(
        ENCODE_KWARGS_DOCSTRING,
        ...,
        ...,
    )
    def encode(
        self,
        text: TextInput | EncodedInput,
        text_pair: None = ...,
        add_special_tokens: bool = ...,
        padding: bool | str | PaddingStrategy = ...,
        truncation: bool | str | TruncationStrategy | None = ...,
        max_length: int | None = ...,
        stride: int = ...,
        pad_to_multiple_of: int | None = ...,
        padding_side: str | None = ...,
        return_tensors: str | TensorType | None = ...,
        verbose: bool = ...,
        **kwargs,
    ) -> list[int]: ...
    def decode(
        self,
        token_ids: int | list[int] | np.ndarray | torch.Tensor,
        skip_special_tokens: bool = ...,
        clean_up_tokenization_spaces: bool | None = ...,
        **kwargs,
    ) -> str: ...
    def batch_decode(
        self,
        sequences: list[int] | list[list[int]] | np.ndarray | torch.Tensor,
        skip_special_tokens: bool = ...,
        clean_up_tokenization_spaces: bool | None = ...,
        **kwargs,
    ) -> list[str]: ...
    @overload
    def convert_ids_to_tokens(
        self, ids: int, skip_special_tokens: bool = ...
    ) -> str: ...
    @overload
    def convert_ids_to_tokens(
        self, ids: list[int], skip_special_tokens: bool = ...
    ) -> list[str]: ...
    def convert_ids_to_tokens(
        self, ids: int | list[int], skip_special_tokens: bool = ...
    ) -> str | list[str]: ...
    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]: ...
    def tokenize(self, text: TextInput, **kwargs) -> list[str]: ...
    def get_special_tokens_mask(
        self,
        token_ids_0: list,
        token_ids_1: None = ...,
        already_has_special_tokens: bool = ...,
    ) -> list[int]: ...
    @add_end_docstrings(
        ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING
    )
    def prepare_for_model(
        self,
        ids: list[int],
        pair_ids: None = ...,
        add_special_tokens: bool = ...,
        padding: bool | str | PaddingStrategy = ...,
        truncation: bool | str | TruncationStrategy | None = ...,
        max_length: int | None = ...,
        stride: int = ...,
        pad_to_multiple_of: int | None = ...,
        padding_side: str | None = ...,
        return_tensors: str | TensorType | None = ...,
        return_attention_mask: bool | None = ...,
        return_overflowing_tokens: bool = ...,
        return_special_tokens_mask: bool = ...,
        return_length: bool = ...,
        verbose: bool = ...,
        prepend_batch_axis: bool = ...,
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
    def truncate_sequences(
        self,
        ids: list[int],
        pair_ids: None = ...,
        num_tokens_to_remove: int = ...,
        truncation_strategy: str | TruncationStrategy = ...,
        stride: int = ...,
        **kwargs,
    ) -> tuple[list[int], None, list[int]]: ...
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]] | list[list[dict[str, str]]],
        tools: list[dict | Callable] | None = ...,
        continue_final_message: bool = ...,
        tokenize: bool = ...,
        padding: bool | str | PaddingStrategy = ...,
        truncation: bool = ...,
        max_length: int | None = ...,
        return_tensors: str | TensorType | None = ...,
        return_dict: bool = ...,
        **kwargs,
    ) -> str | list[int] | list[str] | list[list[int]] | BatchEncoding: ...
    @add_end_docstrings(
        ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING
    )
    def __call__(
        self,
        text: TextInput
        | EncodedInput
        | list[TextInput]
        | list[EncodedInput]
        | None = ...,
        text_pair: None = ...,
        text_target: None = ...,
        text_pair_target: None = ...,
        add_special_tokens: bool = ...,
        padding: bool | str | PaddingStrategy = ...,
        truncation: bool | str | TruncationStrategy | None = ...,
        max_length: int | None = ...,
        stride: int = ...,
        pad_to_multiple_of: int | None = ...,
        padding_side: str | None = ...,
        return_tensors: str | TensorType | None = ...,
        return_attention_mask: bool | None = ...,
        return_overflowing_tokens: bool = ...,
        return_special_tokens_mask: bool = ...,
        return_length: bool = ...,
        verbose: bool = ...,
        **kwargs,
    ) -> BatchEncoding: ...
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        *init_inputs,
        mode: ValidationMode = ...,
        cache_dir: str | os.PathLike | None = ...,
        force_download: bool = ...,
        local_files_only: bool = ...,
        token: str | bool | None = ...,
        revision: str = ...,
        model_max_length: int = ...,
        padding_side: str = ...,
        truncation_side: str = ...,
        model_input_names: list[str] | None = ...,
        clean_up_tokenization_spaces: bool = ...,
        **kwargs,
    ):  # -> Self:
        ...
    def save_pretrained(
        self,
        save_directory: str | os.PathLike | Path,
        push_to_hub: bool = ...,
        token: str | bool | None = ...,
        commit_message: str | None = ...,
        repo_id: str | None = ...,
        private: bool | None = ...,
        repo_url: str | None = ...,
        organization: str | None = ...,
        **kwargs,
    ) -> tuple[str, ...]: ...
