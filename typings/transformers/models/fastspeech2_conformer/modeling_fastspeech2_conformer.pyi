from dataclasses import dataclass

from torch import nn

import torch

from .configuration_fastspeech2_conformer import (
    FastSpeech2ConformerConfig,
    FastSpeech2ConformerHifiGanConfig,
    FastSpeech2ConformerWithHifiGanConfig,
)
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring

"""PyTorch FastSpeech2Conformer model."""
logger = ...

@dataclass
@auto_docstring(custom_intro=...)
class FastSpeech2ConformerModelOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    spectrogram: torch.FloatTensor | None = ...
    encoder_last_hidden_state: torch.FloatTensor | None = ...
    encoder_hidden_states: tuple[torch.FloatTensor] | None = ...
    encoder_attentions: tuple[torch.FloatTensor] | None = ...
    decoder_hidden_states: tuple[torch.FloatTensor] | None = ...
    decoder_attentions: tuple[torch.FloatTensor] | None = ...
    duration_outputs: torch.LongTensor | None = ...
    pitch_outputs: torch.FloatTensor | None = ...
    energy_outputs: torch.FloatTensor | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class FastSpeech2ConformerWithHifiGanOutput(FastSpeech2ConformerModelOutput):
    waveform: torch.FloatTensor | None = ...

def length_regulator(encoded_embeddings, duration_labels, speaking_speed=...): ...

class FastSpeech2ConformerDurationPredictor(nn.Module):
    def __init__(self, config: FastSpeech2ConformerConfig) -> None: ...
    def forward(self, encoder_hidden_states):  # -> Tensor | Any:
        ...

class FastSpeech2ConformerBatchNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=...) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class FastSpeech2ConformerSpeechDecoderPostnet(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor):  # -> tuple[Any, Any]:
        ...

class FastSpeech2ConformerPredictorLayer(nn.Module):
    def __init__(
        self, input_channels, num_chans, kernel_size, dropout_rate
    ) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class FastSpeech2ConformerVariancePredictor(nn.Module):
    def __init__(
        self,
        config: FastSpeech2ConformerConfig,
        num_layers=...,
        num_chans=...,
        kernel_size=...,
        dropout_rate=...,
    ) -> None: ...
    def forward(self, encoder_hidden_states, padding_masks=...):  # -> Any:
        ...

class FastSpeech2ConformerVarianceEmbedding(nn.Module):
    def __init__(
        self,
        in_channels=...,
        out_channels=...,
        kernel_size=...,
        padding=...,
        dropout_rate=...,
    ) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class FastSpeech2ConformerAttention(nn.Module):
    def __init__(self, config: FastSpeech2ConformerConfig, module_config) -> None: ...
    def shift_relative_position_tensor(self, pos_tensor):  # -> Tensor:
        ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        pos_emb: torch.Tensor | None = ...,
        output_attentions: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class FastSpeech2ConformerConvolutionModule(nn.Module):
    def __init__(
        self, config: FastSpeech2ConformerConfig, module_config=...
    ) -> None: ...
    def forward(self, hidden_states, attention_mask=...):  # -> Any:
        ...

class FastSpeech2ConformerEncoderLayer(nn.Module):
    def __init__(self, config: FastSpeech2ConformerConfig, module_config) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        pos_emb: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: torch.Tensor | None = ...,
    ):  # -> tuple[Tensor, Any] | tuple[Tensor]:
        ...

class FastSpeech2ConformerMultiLayeredConv1d(nn.Module):
    def __init__(self, config: FastSpeech2ConformerConfig, module_config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class FastSpeech2ConformerRelPositionalEncoding(nn.Module):
    def __init__(self, config: FastSpeech2ConformerConfig, module_config) -> None: ...
    def extend_pos_enc(self, x):  # -> None:
        ...
    def forward(self, feature_representation):  # -> tuple[Any, Any]:
        ...

class FastSpeech2ConformerEncoder(nn.Module):
    def __init__(
        self,
        config: FastSpeech2ConformerConfig,
        module_config,
        use_encoder_input_layer=...,
    ) -> None: ...
    def forward(
        self,
        input_tensor: torch.LongTensor,
        attention_mask: bool | None = ...,
        output_hidden_states: bool | None = ...,
        output_attentions: bool | None = ...,
        return_dict: bool | None = ...,
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()], ...] | BaseModelOutput:
        ...

class FastSpeech2ConformerLoss(nn.Module):
    def __init__(self, config: FastSpeech2ConformerConfig) -> None: ...
    def forward(
        self,
        outputs_after_postnet,
        outputs_before_postnet,
        duration_outputs,
        pitch_outputs,
        energy_outputs,
        spectrogram_labels,
        duration_labels,
        pitch_labels,
        energy_labels,
        duration_mask,
        spectrogram_mask,
    ):  # -> Any:
        ...

@auto_docstring
class FastSpeech2ConformerPreTrainedModel(PreTrainedModel):
    config: FastSpeech2ConformerConfig
    base_model_prefix = ...
    main_input_name = ...

@auto_docstring(
    custom_intro="""
    FastSpeech2Conformer Model.
    """
)
class FastSpeech2ConformerModel(FastSpeech2ConformerPreTrainedModel):
    def __init__(self, config: FastSpeech2ConformerConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor | None = ...,
        spectrogram_labels: torch.FloatTensor | None = ...,
        duration_labels: torch.LongTensor | None = ...,
        pitch_labels: torch.FloatTensor | None = ...,
        energy_labels: torch.FloatTensor | None = ...,
        speaker_ids: torch.LongTensor | None = ...,
        lang_ids: torch.LongTensor | None = ...,
        speaker_embedding: torch.FloatTensor | None = ...,
        return_dict: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
    ) -> tuple | FastSpeech2ConformerModelOutput: ...

class HifiGanResidualBlock(nn.Module):
    def __init__(
        self, channels, kernel_size=..., dilation=..., leaky_relu_slope=...
    ) -> None: ...
    def get_padding(self, kernel_size, dilation=...): ...
    def apply_weight_norm(self):  # -> None:
        ...
    def remove_weight_norm(self):  # -> None:
        ...
    def forward(self, hidden_states): ...

@auto_docstring(
    custom_intro="""
    HiFi-GAN vocoder.
    """
)
class FastSpeech2ConformerHifiGan(PreTrainedModel):
    config: FastSpeech2ConformerHifiGanConfig
    main_input_name = ...
    def __init__(self, config: FastSpeech2ConformerHifiGanConfig) -> None: ...
    def apply_weight_norm(self):  # -> None:
        ...
    def remove_weight_norm(self):  # -> None:
        ...
    @auto_docstring(custom_intro=...)
    def forward(self, spectrogram: torch.FloatTensor) -> torch.FloatTensor: ...

@auto_docstring(custom_intro=...)
class FastSpeech2ConformerWithHifiGan(PreTrainedModel):
    config: FastSpeech2ConformerWithHifiGanConfig
    def __init__(self, config: FastSpeech2ConformerWithHifiGanConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor | None = ...,
        spectrogram_labels: torch.FloatTensor | None = ...,
        duration_labels: torch.LongTensor | None = ...,
        pitch_labels: torch.FloatTensor | None = ...,
        energy_labels: torch.FloatTensor | None = ...,
        speaker_ids: torch.LongTensor | None = ...,
        lang_ids: torch.LongTensor | None = ...,
        speaker_embedding: torch.FloatTensor | None = ...,
        return_dict: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
    ) -> tuple | FastSpeech2ConformerModelOutput: ...

__all__ = [
    "FastSpeech2ConformerHifiGan",
    "FastSpeech2ConformerModel",
    "FastSpeech2ConformerPreTrainedModel",
    "FastSpeech2ConformerWithHifiGan",
]
