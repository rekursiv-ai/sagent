from ...tokenization_utils_fast import PreTrainedTokenizerFast

"""Tokenization classes for Bloom."""
logger = ...
VOCAB_FILES_NAMES = ...

class BloomTokenizerFast(PreTrainedTokenizerFast):
    vocab_files_names = ...
    model_input_names = ...
    slow_tokenizer_class = ...
    def __init__(
        self,
        vocab_file=...,
        merges_file=...,
        tokenizer_file=...,
        unk_token=...,
        bos_token=...,
        eos_token=...,
        pad_token=...,
        add_prefix_space=...,
        clean_up_tokenization_spaces=...,
        **kwargs,
    ) -> None: ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str]: ...

__all__ = ["BloomTokenizerFast"]
