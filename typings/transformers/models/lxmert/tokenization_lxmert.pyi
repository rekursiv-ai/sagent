from ...tokenization_utils import PreTrainedTokenizer

logger = ...
VOCAB_FILES_NAMES = ...

def load_vocab(vocab_file):  # -> OrderedDict[Any, Any]:
    ...
def whitespace_tokenize(text):  # -> list[Any]:
    ...

class LxmertTokenizer(PreTrainedTokenizer):
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
        tokenize_chinese_chars=...,
        strip_accents=...,
        clean_up_tokenization_spaces=...,
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

__all__ = ["LxmertTokenizer"]
