from typing import Any

import sentencepiece as spm

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import is_sentencepiece_available

"""Tokenization classes."""
if is_sentencepiece_available(): ...
else:
    spm = ...
logger = ...
VOCAB_FILES_NAMES = ...
SPIECE_UNDERLINE = ...

def load_vocab(vocab_file):  # -> OrderedDict[Any, Any]:
    ...
def whitespace_tokenize(text):  # -> list[Any]:
    ...

class BertJapaneseTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        vocab_file,
        spm_file=...,
        do_lower_case=...,
        do_word_tokenize=...,
        do_subword_tokenize=...,
        word_tokenizer_type=...,
        subword_tokenizer_type=...,
        never_split=...,
        unk_token=...,
        sep_token=...,
        pad_token=...,
        cls_token=...,
        mask_token=...,
        mecab_kwargs=...,
        sudachi_kwargs=...,
        jumanpp_kwargs=...,
        **kwargs,
    ) -> None: ...
    @property
    def do_lower_case(self):  # -> bool:
        ...
    def __getstate__(self):  # -> dict[str, Any]:
        ...
    def __setstate__(self, state):  # -> None:
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

class MecabTokenizer:
    def __init__(
        self,
        do_lower_case=...,
        never_split=...,
        normalize_text=...,
        mecab_dic: str | None = ...,
        mecab_option: str | None = ...,
    ) -> None: ...
    def tokenize(self, text, never_split=..., **kwargs):  # -> list[Any]:
        ...

class SudachiTokenizer:
    def __init__(
        self,
        do_lower_case=...,
        never_split=...,
        normalize_text=...,
        trim_whitespace=...,
        sudachi_split_mode=...,
        sudachi_config_path=...,
        sudachi_resource_dir=...,
        sudachi_dict_type=...,
        sudachi_projection=...,
    ) -> None: ...
    def tokenize(self, text, never_split=..., **kwargs):  # -> list[Any]:
        ...

class JumanppTokenizer:
    def __init__(
        self,
        do_lower_case=...,
        never_split=...,
        normalize_text=...,
        trim_whitespace=...,
    ) -> None: ...
    def tokenize(self, text, never_split=..., **kwargs):  # -> list[Any]:
        ...

class CharacterTokenizer:
    def __init__(self, vocab, unk_token, normalize_text=...) -> None: ...
    def tokenize(self, text):  # -> list[Any]:
        ...

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

class SentencepieceTokenizer:
    def __init__(
        self,
        vocab,
        unk_token,
        do_lower_case=...,
        remove_space=...,
        keep_accents=...,
        sp_model_kwargs: dict[str, Any] | None = ...,
    ) -> None: ...
    def preprocess_text(self, inputs):  # -> str:
        ...
    def tokenize(self, text):  # -> list[Any]:
        ...

__all__ = ["BertJapaneseTokenizer", "CharacterTokenizer", "MecabTokenizer"]
