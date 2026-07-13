import os

from ...modeling_tf_utils import keras
from ...utils.import_utils import requires

@requires(backends=("tf", "tensorflow_text"))
class TFBertTokenizer(keras.layers.Layer):
    def __init__(
        self,
        vocab_list: list,
        do_lower_case: bool,
        cls_token_id: int | None = ...,
        sep_token_id: int | None = ...,
        pad_token_id: int | None = ...,
        padding: str = ...,
        truncation: bool = ...,
        max_length: int = ...,
        pad_to_multiple_of: int | None = ...,
        return_token_type_ids: bool = ...,
        return_attention_mask: bool = ...,
        use_fast_bert_tokenizer: bool = ...,
        **tokenizer_kwargs,
    ) -> None: ...
    @classmethod
    def from_tokenizer(cls, tokenizer: PreTrainedTokenizerBase, **kwargs):  # -> Self:
        ...
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        *init_inputs,
        **kwargs,
    ):  # -> Self:
        ...
    def unpaired_tokenize(self, texts): ...
    def call(
        self,
        text,
        text_pair=...,
        padding=...,
        truncation=...,
        max_length=...,
        pad_to_multiple_of=...,
        return_token_type_ids=...,
        return_attention_mask=...,
    ):  # -> dict[str, Any]:
        ...
    def get_config(self):  # -> dict[str, list[Any] | bool | int]:
        ...

__all__ = ["TFBertTokenizer"]
