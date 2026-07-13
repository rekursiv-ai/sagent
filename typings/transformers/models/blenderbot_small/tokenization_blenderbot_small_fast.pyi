from .tokenization_blenderbot_small import BlenderbotSmallTokenizer
from ...tokenization_utils_fast import PreTrainedTokenizerFast

"""Fast tokenization class for BlenderbotSmall."""
logger = ...
VOCAB_FILES_NAMES = ...

class BlenderbotSmallTokenizerFast(PreTrainedTokenizerFast):
    vocab_files_names = ...
    slow_tokenizer_class = BlenderbotSmallTokenizer
    def __init__(
        self,
        vocab_file=...,
        merges_file=...,
        unk_token=...,
        bos_token=...,
        eos_token=...,
        add_prefix_space=...,
        trim_offsets=...,
        **kwargs,
    ) -> None: ...
    def build_inputs_with_special_tokens(
        self, token_ids_0, token_ids_1=...
    ):  # -> list[str | list[str] | Any | None]:
        ...
    def create_token_type_ids_from_sequences(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = ...
    ) -> list[int]: ...

__all__ = ["BlenderbotSmallTokenizerFast"]
