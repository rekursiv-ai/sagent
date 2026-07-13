from typing import Any

from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import (
    ENCODE_KWARGS_DOCSTRING,
    BatchEncoding,
    EncodedInput,
    PaddingStrategy,
    TensorType,
    TextInput,
    TruncationStrategy,
)
from ...utils import add_end_docstrings
from ...utils.import_utils import requires

"""Tokenization classes for mLUKE."""
logger = ...
type EntitySpan = tuple[int, int]
type EntitySpanInput = list[EntitySpan]
Entity = str
type EntityInput = list[Entity]
SPIECE_UNDERLINE = ...
VOCAB_FILES_NAMES = ...
ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING = ...

@requires(backends=("sentencepiece",))
class MLukeTokenizer(PreTrainedTokenizer):
    vocab_files_names = ...
    model_input_names = ...
    def __init__(
        self,
        vocab_file,
        entity_vocab_file,
        bos_token=...,
        eos_token=...,
        sep_token=...,
        cls_token=...,
        unk_token=...,
        pad_token=...,
        mask_token=...,
        task=...,
        max_entity_length=...,
        max_mention_length=...,
        entity_token_1=...,
        entity_token_2=...,
        entity_unk_token=...,
        entity_pad_token=...,
        entity_mask_token=...,
        entity_mask2_token=...,
        sp_model_kwargs: dict[str, Any] | None = ...,
        **kwargs,
    ) -> None: ...
    @property
    def vocab_size(self):  # -> int:
        ...
    def get_vocab(self):  # -> dict[str, int]:
        ...
    def convert_tokens_to_string(self, tokens):  # -> str:
        ...
    def __getstate__(self):  # -> dict[str, Any]:
        ...
    def __setstate__(self, d):  # -> None:
        ...
    @add_end_docstrings(
        ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING
    )
    def __call__(
        self,
        text: TextInput | list[TextInput],
        text_pair: TextInput | list[TextInput] | None = ...,
        entity_spans: EntitySpanInput | list[EntitySpanInput] | None = ...,
        entity_spans_pair: EntitySpanInput | list[EntitySpanInput] | None = ...,
        entities: EntityInput | list[EntityInput] | None = ...,
        entities_pair: EntityInput | list[EntityInput] | None = ...,
        add_special_tokens: bool = ...,
        padding: bool | str | PaddingStrategy = ...,
        truncation: bool | str | TruncationStrategy = ...,
        max_length: int | None = ...,
        max_entity_length: int | None = ...,
        stride: int = ...,
        is_split_into_words: bool | None = ...,
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
    def prepare_for_model(
        self,
        ids: list[int],
        pair_ids: list[int] | None = ...,
        entity_ids: list[int] | None = ...,
        pair_entity_ids: list[int] | None = ...,
        entity_token_spans: list[tuple[int, int]] | None = ...,
        pair_entity_token_spans: list[tuple[int, int]] | None = ...,
        add_special_tokens: bool = ...,
        padding: bool | str | PaddingStrategy = ...,
        truncation: bool | str | TruncationStrategy = ...,
        max_length: int | None = ...,
        max_entity_length: int | None = ...,
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
    def pad(
        self,
        encoded_inputs: BatchEncoding
        | list[BatchEncoding]
        | dict[str, EncodedInput]
        | dict[str, list[EncodedInput]]
        | list[dict[str, EncodedInput]],
        padding: bool | str | PaddingStrategy = ...,
        max_length: int | None = ...,
        max_entity_length: int | None = ...,
        pad_to_multiple_of: int | None = ...,
        padding_side: str | None = ...,
        return_attention_mask: bool | None = ...,
        return_tensors: str | TensorType | None = ...,
        verbose: bool = ...,
    ) -> BatchEncoding: ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str, str]: ...
    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = ...
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

__all__ = ["MLukeTokenizer"]
