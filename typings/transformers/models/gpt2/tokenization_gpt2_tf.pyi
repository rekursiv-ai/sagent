import os

from .tokenization_gpt2 import GPT2Tokenizer
from ...modeling_tf_utils import keras
from ...utils.import_utils import requires

@requires(backends=("keras_nlp",))
class TFGPT2Tokenizer(keras.layers.Layer):
    def __init__(
        self,
        vocab: dict[str, int],
        merges: list[str],
        max_length: int | None = ...,
        pad_token_id: int | None = ...,
    ) -> None: ...
    @classmethod
    def from_tokenizer(cls, tokenizer: GPT2Tokenizer, *args, **kwargs):  # -> Self:
        ...
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        *init_inputs,
        **kwargs,
    ):  # -> Self:
        ...
    @classmethod
    def from_config(cls, config):  # -> Self:
        ...
    def get_config(self):  # -> dict[str, dict[str, int] | list[str] | int | None]:
        ...
    def call(self, x, max_length: int | None = ...):  # -> dict[str, Any]:
        ...

__all__ = ["TFGPT2Tokenizer"]
