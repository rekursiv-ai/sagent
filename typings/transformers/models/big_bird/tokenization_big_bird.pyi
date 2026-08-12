from typing import Any

from ...tokenization_utils import PreTrainedTokenizer
from ...utils.import_utils import requires

"""Tokenization classes for BigBird."""
logger = ...
VOCAB_FILES_NAMES = ...

@requires(backends=("sentencepiece",))
class BigBirdTokenizer(PreTrainedTokenizer):
    prefix_tokens: list[int] = ...
    def __init__(
        self,
        vocab_file,
        unk_token=...,
        bos_token=...,
        eos_token=...,
        pad_token=...,
        sep_token=...,
        mask_token=...,
        cls_token=...,
        sp_model_kwargs: dict[str, Any] | None = ...,
        **kwargs,
    ) -> None: ...
    @property
    def vocab_size(self): ...
    def get_vocab(self):  # -> dict[str, int]:
        ...
    def __getstate__(self):  # -> dict[str, Any]:
        ...
    def __setstate__(self, d):  # -> None:
        ...
    def convert_tokens_to_string(self, tokens): ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str]: ...
    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = ...
    ) -> list[int]: ...
    def get_special_tokens_mask(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = ...,
        already_has_special_tokens: bool = ...,
    ) -> list[int]: ...

__all__ = ["BigBirdTokenizer"]
