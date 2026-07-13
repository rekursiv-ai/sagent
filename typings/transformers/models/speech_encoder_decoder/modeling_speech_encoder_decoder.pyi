import torch

from .configuration_speech_encoder_decoder import SpeechEncoderDecoderConfig
from ...cache_utils import Cache
from ...configuration_utils import PretrainedConfig
from ...generation import GenerationMixin
from ...modeling_outputs import Seq2SeqLMOutput
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

"""Classes to support Speech-Encoder-Text-Decoder architectures"""
logger = ...

def shift_tokens_right(
    input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int
):  # -> Tensor:
    ...

@auto_docstring
class SpeechEncoderDecoderModel(PreTrainedModel, GenerationMixin):
    config: SpeechEncoderDecoderConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...
    _supports_param_buffer_assignment = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    def __init__(
        self,
        config: PretrainedConfig | None = ...,
        encoder: PreTrainedModel | None = ...,
        decoder: PreTrainedModel | None = ...,
    ) -> None: ...
    def get_encoder(self):  # -> PreTrainedModel | None:
        ...
    def get_input_embeddings(self):  # -> Module:
        ...
    def get_output_embeddings(self):  # -> None:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    def freeze_feature_encoder(self):  # -> None:
        ...
    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: str | None = ...,
        decoder_pretrained_model_name_or_path: str | None = ...,
        *model_args,
        **kwargs,
    ) -> PreTrainedModel: ...
    @auto_docstring
    def forward(
        self,
        inputs: torch.FloatTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.BoolTensor | None = ...,
        encoder_outputs: tuple[torch.FloatTensor] | None = ...,
        past_key_values: Cache | None = ...,
        decoder_inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        input_values: torch.FloatTensor | None = ...,
        input_features: torch.FloatTensor | None = ...,
        return_dict: bool | None = ...,
        **kwargs,
    ) -> tuple[torch.FloatTensor] | Seq2SeqLMOutput: ...
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):  # -> Tensor:
        ...
    def resize_token_embeddings(self, *args, **kwargs): ...

__all__ = ["SpeechEncoderDecoderModel"]
