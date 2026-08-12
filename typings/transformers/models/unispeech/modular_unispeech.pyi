from typing import Any
from dataclasses import dataclass

import torch

from .configuration_unispeech import UniSpeechConfig
from ..wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Encoder,
    Wav2Vec2EncoderStableLayerNorm,
    Wav2Vec2FeatureEncoder,
    Wav2Vec2FeatureProjection,
    Wav2Vec2ForCTC,
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2GumbelVectorQuantizer,
    Wav2Vec2Model,
    Wav2Vec2PositionalConvEmbedding,
)
from ...modeling_outputs import ModelOutput, Wav2Vec2BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

"""PyTorch UniSpeech model."""
logger = ...

@dataclass
@auto_docstring(custom_intro=...)
class UniSpeechForPreTrainingOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    projected_states: torch.FloatTensor | None = ...
    projected_quantized_states: torch.FloatTensor | None = ...
    codevector_perplexity: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

class UniSpeechPositionalConvEmbedding(Wav2Vec2PositionalConvEmbedding): ...
class UniSpeechFeatureEncoder(Wav2Vec2FeatureEncoder): ...
class UniSpeechFeatureProjection(Wav2Vec2FeatureProjection): ...
class UniSpeechEncoder(Wav2Vec2Encoder): ...
class UniSpeechEncoderStableLayerNorm(Wav2Vec2EncoderStableLayerNorm): ...

class UniSpeechGumbelVectorQuantizer(Wav2Vec2GumbelVectorQuantizer):
    def forward(self, hidden_states):  # -> tuple[Tensor | Any, Tensor]:
        ...

@auto_docstring
class UniSpeechPreTrainedModel(PreTrainedModel):
    config: UniSpeechConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_flex_attn = ...

UniSpeechBaseModelOutput = Wav2Vec2BaseModelOutput

class UniSpeechModel(UniSpeechPreTrainedModel, Wav2Vec2Model):
    def __init__(self, config: UniSpeechConfig) -> None: ...
    def freeze_feature_extractor(self): ...
    def freeze_feature_encoder(self): ...
    def forward(
        self,
        input_values: torch.Tensor | None,
        attention_mask: torch.Tensor | None = ...,
        mask_time_indices: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | UniSpeechBaseModelOutput: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple | UniSpeechBaseModelOutput: ...

@auto_docstring(custom_intro=...)
class UniSpeechForPreTraining(UniSpeechPreTrainedModel):
    def __init__(self, config: UniSpeechConfig) -> None: ...
    def set_gumbel_temperature(self, temperature: int):  # -> None:
        ...
    def freeze_feature_extractor(self):  # -> None:
        ...
    def freeze_feature_encoder(self):  # -> None:
        ...
    @staticmethod
    def compute_contrastive_logits(
        target_features: torch.FloatTensor,
        negative_features: torch.FloatTensor,
        predicted_features: torch.FloatTensor,
        temperature: int = ...,
    ):  # -> Tensor:
        ...
    @auto_docstring
    def forward(
        self,
        input_values: torch.Tensor | None,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | UniSpeechForPreTrainingOutput: ...

class UniSpeechForCTC(Wav2Vec2ForCTC): ...
class UniSpeechForSequenceClassification(Wav2Vec2ForSequenceClassification): ...

__all__ = [
    "UniSpeechForCTC",
    "UniSpeechForPreTraining",
    "UniSpeechForSequenceClassification",
    "UniSpeechModel",
    "UniSpeechPreTrainedModel",
]
