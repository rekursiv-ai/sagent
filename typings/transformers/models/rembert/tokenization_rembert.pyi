from ...tokenization_utils import PreTrainedTokenizer
from ...utils.import_utils import requires

"""Tokenization classes for RemBERT."""
logger = ...
VOCAB_FILES_NAMES = ...

@requires(backends=("sentencepiece",))
class RemBertTokenizer(PreTrainedTokenizer):
    vocab_files_names = ...
    def __init__(
        self,
        vocab_file,
        do_lower_case=...,
        remove_space=...,
        keep_accents=...,
        bos_token=...,
        eos_token=...,
        unk_token=...,
        sep_token=...,
        pad_token=...,
        cls_token=...,
        mask_token=...,
        **kwargs,
    ) -> None: ...
    @property
    def vocab_size(self):  # -> int:
        ...
    def get_vocab(self):  # -> dict[str, int]:
        ...
    def __getstate__(self):  # -> dict[str, Any]:
        ...
    def __setstate__(self, d):  # -> None:
        ...
    def convert_tokens_to_string(self, tokens): ...
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

__all__ = ["RemBertTokenizer"]
