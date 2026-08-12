from ...tokenization_utils import PreTrainedTokenizer

"""Tokenization class for model DeBERTa."""
logger = ...
VOCAB_FILES_NAMES = ...

def bytes_to_unicode():  # -> dict[int, str]:
    ...
def get_pairs(word):  # -> set[Any]:
    ...

class DebertaTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        vocab_file,
        merges_file,
        errors=...,
        bos_token=...,
        eos_token=...,
        sep_token=...,
        cls_token=...,
        unk_token=...,
        pad_token=...,
        mask_token=...,
        add_prefix_space=...,
        add_bos_token=...,
        **kwargs,
    ) -> None: ...
    @property
    def vocab_size(self):  # -> int:
        ...
    def get_vocab(self):  # -> dict[str, Any]:
        ...
    def bpe(self, token):  # -> LiteralString:
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
    def convert_tokens_to_string(self, tokens):  # -> str:
        ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str]: ...
    def prepare_for_tokenization(
        self, text, is_split_into_words=..., **kwargs
    ):  # -> tuple[str, dict[str, Any]]:
        ...

__all__ = ["DebertaTokenizer"]
