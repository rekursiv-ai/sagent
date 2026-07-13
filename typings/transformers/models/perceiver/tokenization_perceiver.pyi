from ...tokenization_utils import PreTrainedTokenizer

"""Tokenization class for Perceiver."""
logger = ...

class PerceiverTokenizer(PreTrainedTokenizer):
    model_input_names = ...
    def __init__(
        self,
        pad_token=...,
        bos_token=...,
        eos_token=...,
        mask_token=...,
        cls_token=...,
        sep_token=...,
        model_max_length=...,
        **kwargs,
    ) -> None: ...
    def get_vocab(self) -> dict[str, int]: ...
    @property
    def vocab_size(self):  # -> int:
        ...
    def get_special_tokens_mask(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = ...,
        already_has_special_tokens: bool = ...,
    ) -> list[int]: ...
    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = ...
    ) -> list[int]: ...
    def convert_tokens_to_string(self, tokens):  # -> str:
        ...
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = ...
    ) -> tuple[str]: ...

__all__ = ["PerceiverTokenizer"]
