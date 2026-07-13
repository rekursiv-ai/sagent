from typing import Any

from ...tokenization_utils import (
    BatchEncoding,
    PreTokenizedInput,
    PreTrainedTokenizer,
    TextInput,
)
from ...utils import PaddingStrategy
from ...utils.import_utils import requires

"""Tokenization classes for SeamlessM4T."""
logger = ...
SPIECE_UNDERLINE = ...
VOCAB_FILES_NAMES = ...

@requires(backends=("sentencepiece",))
class SeamlessM4TTokenizer(PreTrainedTokenizer):
    vocab_files_names = ...
    model_input_names = ...
    prefix_tokens: list[int] = ...
    suffix_tokens: list[int] = ...
    def __init__(
        self,
        vocab_file,
        bos_token=...,
        eos_token=...,
        sep_token=...,
        cls_token=...,
        unk_token=...,
        pad_token=...,
        tokenizer_file=...,
        src_lang=...,
        tgt_lang=...,
        sp_model_kwargs: dict[str, Any] | None = ...,
        additional_special_tokens=...,
        add_prefix_space=...,
        **kwargs,
    ) -> None: ...
    def __getstate__(self):  # -> dict[str, Any]:
        ...
    def __setstate__(self, d):  # -> None:
        ...
    @property
    def vocab_size(self):  # -> int:
        ...
    def __call__(
        self,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput] = ...,
        text_pair: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput]
        | None = ...,
        text_target: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput] = ...,
        text_pair_target: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput]
        | None = ...,
        padding: bool | str | PaddingStrategy = ...,
        pad_to_multiple_of: int | None = ...,
        src_lang: str | None = ...,
        tgt_lang: str | None = ...,
        **kwargs,
    ):  # -> BatchEncoding:
        ...
    @property
    def src_lang(self) -> str: ...
    @src_lang.setter
    def src_lang(self, new_src_lang: str) -> None: ...
    @property
    def tgt_lang(self) -> str: ...
    @tgt_lang.setter
    def tgt_lang(self, new_tgt_lang: str) -> None: ...
    def get_special_tokens_mask(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = ...,
        already_has_special_tokens: bool = ...,
    ) -> list[int]: ...
    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = ...
    ) -> list[int]: ...
    def create_token_type_ids_from_sequences(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = ...
    ) -> list[int]: ...
    def get_vocab(self):  # -> dict[str, int]:
        ...
    @property
    def unk_token_length(self):  # -> int:
        ...
    def get_spm_processor(self, from_slow=...): ...
    def tokenize(self, text: TextInput, **kwargs) -> list[str]: ...
    def convert_tokens_to_string(self, tokens):  # -> str:
        ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str]: ...
    def prepare_seq2seq_batch(
        self,
        src_texts: list[str],
        src_lang: str = ...,
        tgt_texts: list[str] | None = ...,
        tgt_lang: str = ...,
        **kwargs,
    ) -> BatchEncoding: ...
    def set_src_lang_special_tokens(self, src_lang) -> None: ...
    def set_tgt_lang_special_tokens(self, lang: str) -> None: ...

__all__ = ["SeamlessM4TTokenizer"]
