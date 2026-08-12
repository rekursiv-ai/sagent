from .tokenization_openai import OpenAIGPTTokenizer
from ...tokenization_utils_fast import PreTrainedTokenizerFast

"""Fast Tokenization classes for OpenAI GPT."""
logger = ...
VOCAB_FILES_NAMES = ...

class OpenAIGPTTokenizerFast(PreTrainedTokenizerFast):
    slow_tokenizer_class = OpenAIGPTTokenizer
    def __init__(
        self,
        vocab_file=...,
        merges_file=...,
        tokenizer_file=...,
        unk_token=...,
        **kwargs,
    ) -> None: ...
    @property
    def do_lower_case(self):  # -> Literal[True]:
        ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str]: ...

__all__ = ["OpenAIGPTTokenizerFast"]
