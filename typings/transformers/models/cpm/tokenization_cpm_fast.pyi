from ...tokenization_utils_fast import PreTrainedTokenizerFast

"""Tokenization classes."""
logger = ...
VOCAB_FILES_NAMES = ...

class CpmTokenizerFast(PreTrainedTokenizerFast):
    def __init__(
        self,
        vocab_file=...,
        tokenizer_file=...,
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
        additional_special_tokens=...,
        **kwargs,
    ) -> None: ...
    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = ...
    ) -> list[int]: ...
    def create_token_type_ids_from_sequences(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = ...
    ) -> list[int]: ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str]: ...

__all__ = ["CpmTokenizerFast"]
