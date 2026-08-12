from ...tokenization_utils import PreTrainedTokenizer

"""Tokenization classes for MGT-STR CHAR."""
logger = ...
VOCAB_FILES_NAMES = ...

class MgpstrTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        vocab_file,
        unk_token=...,
        bos_token=...,
        eos_token=...,
        pad_token=...,
        **kwargs,
    ) -> None: ...
    @property
    def vocab_size(self):  # -> int:
        ...
    def get_vocab(self):  # -> dict[Any, Any]:
        ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str]: ...

__all__ = ["MgpstrTokenizer"]
