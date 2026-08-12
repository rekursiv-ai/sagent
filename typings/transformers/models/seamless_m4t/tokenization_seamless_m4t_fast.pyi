from .tokenization_seamless_m4t import SeamlessM4TTokenizer
from ...tokenization_utils import BatchEncoding, PreTokenizedInput, TextInput
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import PaddingStrategy, is_sentencepiece_available

"""Fast Tokenization class for SeamlessM4T."""
if is_sentencepiece_available(): ...
else:
    SeamlessM4TTokenizer = ...
logger = ...
VOCAB_FILES_NAMES = ...

class SeamlessM4TTokenizerFast(PreTrainedTokenizerFast):
    prefix_tokens: list[int] = ...
    suffix_tokens: list[int] = ...
    def __init__(
        self,
        vocab_file=...,
        tokenizer_file=...,
        bos_token=...,
        eos_token=...,
        sep_token=...,
        cls_token=...,
        unk_token=...,
        pad_token=...,
        src_lang=...,
        tgt_lang=...,
        additional_special_tokens=...,
        **kwargs,
    ) -> None: ...
    @property
    def src_lang(self) -> str: ...
    @src_lang.setter
    def src_lang(self, new_src_lang: str) -> None: ...
    @property
    def tgt_lang(self) -> str: ...
    @tgt_lang.setter
    def tgt_lang(self, new_tgt_lang: str) -> None: ...
    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = ...
    ) -> list[int]: ...
    def create_token_type_ids_from_sequences(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = ...
    ) -> list[int]: ...
    def prepare_seq2seq_batch(
        self,
        src_texts: list[str],
        src_lang: str = ...,
        tgt_texts: list[str] | None = ...,
        tgt_lang: str = ...,
        **kwargs,
    ) -> BatchEncoding: ...
    def set_src_lang_special_tokens(self, src_lang) -> None: ...
    def set_tgt_lang_special_tokens(self, lang: str) -> None: ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str]: ...
    def __call__(
        self,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput] = ...,
        text_pair: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput]
        | None = ...,
        text_target: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput] = ...,
        text_pair_target: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput]
        | None = ...,
        padding: bool | str | PaddingStrategy = ...,
        pad_to_multiple_of: int | None = ...,
        src_lang: str | None = ...,
        tgt_lang: str | None = ...,
        **kwargs,
    ):  # -> BatchEncoding:
        ...

__all__ = ["SeamlessM4TTokenizerFast"]
