from dataclasses import dataclass

from torch import nn

import torch

from .configuration_encodec import EncodecConfig
from ...modeling_utils import PreTrainedAudioTokenizerBase
from ...utils import ModelOutput, auto_docstring

"""PyTorch EnCodec model."""
logger = ...

@dataclass
@auto_docstring
class EncodecOutput(ModelOutput):
    audio_codes: torch.LongTensor | None = ...
    audio_values: torch.FloatTensor | None = ...

@dataclass
@auto_docstring
class EncodecEncoderOutput(ModelOutput):
    audio_codes: torch.LongTensor | None = ...
    audio_scales: torch.FloatTensor | None = ...
    last_frame_pad_length: int | None = ...

@dataclass
@auto_docstring
class EncodecDecoderOutput(ModelOutput):
    audio_values: torch.FloatTensor | None = ...

class EncodecConv1d(nn.Module):
    def __init__(
        self,
        config,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = ...,
        dilation: int = ...,
    ) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class EncodecConvTranspose1d(nn.Module):
    def __init__(
        self,
        config,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = ...,
    ) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class EncodecLSTM(nn.Module):
    def __init__(self, config: EncodecConfig, dimension: int) -> None: ...
    def forward(self, hidden_states): ...

class EncodecResnetBlock(nn.Module):
    def __init__(
        self, config: EncodecConfig, dim: int, dilations: list[int]
    ) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class EncodecEncoder(nn.Module):
    def __init__(self, config: EncodecConfig) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class EncodecDecoder(nn.Module):
    def __init__(self, config: EncodecConfig) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class EncodecEuclideanCodebook(nn.Module):
    def __init__(self, config: EncodecConfig) -> None: ...
    def quantize(self, hidden_states): ...
    def encode(self, hidden_states): ...
    def decode(self, embed_ind):  # -> Tensor:
        ...

class EncodecVectorQuantization(nn.Module):
    def __init__(self, config: EncodecConfig) -> None: ...
    def encode(self, hidden_states): ...
    def decode(self, embed_ind):  # -> Tensor:
        ...

class EncodecResidualVectorQuantizer(nn.Module):
    def __init__(self, config: EncodecConfig) -> None: ...
    def get_num_quantizers_for_bandwidth(
        self, bandwidth: float | None = ...
    ) -> int: ...
    def encode(
        self, embeddings: torch.Tensor, bandwidth: float | None = ...
    ) -> torch.Tensor: ...
    def decode(self, codes: torch.Tensor) -> torch.Tensor: ...

@auto_docstring
class EncodecPreTrainedModel(PreTrainedAudioTokenizerBase):
    config: EncodecConfig
    base_model_prefix = ...
    main_input_name = ...

@auto_docstring(
    custom_intro="""
    The EnCodec neural audio codec model.
    """
)
class EncodecModel(EncodecPreTrainedModel):
    def __init__(self, config: EncodecConfig) -> None: ...
    def get_encoder(self):  # -> EncodecEncoder:
        ...
    def encode(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor | None = ...,
        bandwidth: float | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None, int] | EncodecEncoderOutput: ...
    def decode(
        self,
        audio_codes: torch.LongTensor,
        audio_scales: torch.Tensor,
        padding_mask: torch.Tensor | None = ...,
        return_dict: bool | None = ...,
        last_frame_pad_length: int | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor] | EncodecDecoderOutput: ...
    @auto_docstring
    def forward(
        self,
        input_values: torch.FloatTensor,
        padding_mask: torch.BoolTensor | None = ...,
        bandwidth: float | None = ...,
        audio_codes: torch.LongTensor | None = ...,
        audio_scales: torch.Tensor | None = ...,
        return_dict: bool | None = ...,
        last_frame_pad_length: int | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor] | EncodecOutput: ...

__all__ = ["EncodecModel", "EncodecPreTrainedModel"]
