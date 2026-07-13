from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding

"""
Processor class for Bark
"""
logger = ...

class BarkProcessor(ProcessorMixin):
    tokenizer_class = ...
    attributes = ...
    preset_shape = ...
    def __init__(self, tokenizer, speaker_embeddings=...) -> None: ...
    @classmethod
    def from_pretrained(
        cls,
        pretrained_processor_name_or_path,
        speaker_embeddings_dict_path=...,
        **kwargs,
    ):  # -> Self:
        ...
    def save_pretrained(
        self,
        save_directory,
        speaker_embeddings_dict_path=...,
        speaker_embeddings_directory=...,
        push_to_hub: bool = ...,
        **kwargs,
    ):  # -> None:
        ...
    @property
    def available_voice_presets(self) -> list: ...
    def __call__(
        self,
        text=...,
        voice_preset=...,
        return_tensors=...,
        max_length=...,
        add_special_tokens=...,
        return_attention_mask=...,
        return_token_type_ids=...,
        **kwargs,
    ) -> BatchEncoding: ...

__all__ = ["BarkProcessor"]
