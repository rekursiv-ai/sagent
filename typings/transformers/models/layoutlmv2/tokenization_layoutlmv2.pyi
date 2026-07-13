from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import (
    BatchEncoding,
    PreTokenizedInput,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)
from ...utils import PaddingStrategy, TensorType, add_end_docstrings

"""Tokenization class for LayoutLMv2."""
logger = ...
VOCAB_FILES_NAMES = ...
LAYOUTLMV2_ENCODE_KWARGS_DOCSTRING = ...
LAYOUTLMV2_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING = ...

def load_vocab(vocab_file):  # -> OrderedDict[Any, Any]:
    ...
def whitespace_tokenize(text):  # -> list[Any]:
    ...

table = ...

def subfinder(mylist, pattern):  # -> tuple[Any, Any] | tuple[None, Literal[0]]:
    ...

class LayoutLMv2Tokenizer(PreTrainedTokenizer):
    vocab_files_names = ...
    def __init__(
        self,
        vocab_file,
        do_lower_case=...,
        do_basic_tokenize=...,
        never_split=...,
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
        model_max_length: int = ...,
        additional_special_tokens: list[str] | None = ...,
        **kwargs,
    ) -> None: ...
    @property
    def do_lower_case(self):  # -> bool:
        ...
    @property
    def vocab_size(self):  # -> int:
        ...
    def get_vocab(self):  # -> dict[str, int]:
        ...
    def convert_tokens_to_string(self, tokens):  # -> str:
        ...
    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = ...
    ) -> list[int]: ...
    def get_special_tokens_mask(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = ...,
        already_has_special_tokens: bool = ...,
    ) -> list[int]: ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str]: ...
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
    @add_end_docstrings(LAYOUTLMV2_ENCODE_KWARGS_DOCSTRING)
    def encode(
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
    ) -> list[int]: ...
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
    @add_end_docstrings(
        LAYOUTLMV2_ENCODE_KWARGS_DOCSTRING,
        LAYOUTLMV2_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING,
    )
    def prepare_for_model(
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

class BasicTokenizer:
    def __init__(
        self,
        do_lower_case=...,
        never_split=...,
        tokenize_chinese_chars=...,
        strip_accents=...,
        do_split_on_punc=...,
    ) -> None: ...
    def tokenize(self, text, never_split=...):  # -> list[Any] | list[LiteralString]:
        ...

class WordpieceTokenizer:
    def __init__(self, vocab, unk_token, max_input_chars_per_word=...) -> None: ...
    def tokenize(self, text):  # -> list[Any]:
        ...

__all__ = ["LayoutLMv2Tokenizer"]
