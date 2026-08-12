from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import abc

from torch import nn

import torch

from .configuration_perceiver import PerceiverConfig
from ...modeling_outputs import BaseModelOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring

"""PyTorch Perceiver model."""
type ModalitySizeType = Mapping[str, int]
type PreprocessorOutputType = tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]
type PreprocessorType = Callable[..., PreprocessorOutputType]
type PostprocessorType = Callable[..., Any]
logger = ...

@dataclass
@auto_docstring(custom_intro=...)
class PerceiverModelOutput(ModelOutput):
    logits: torch.FloatTensor | None = ...
    last_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    cross_attentions: tuple[torch.FloatTensor] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class PerceiverDecoderOutput(ModelOutput):
    logits: torch.FloatTensor | None = ...
    cross_attentions: tuple[torch.FloatTensor] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class PerceiverMaskedLMOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    cross_attentions: tuple[torch.FloatTensor] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class PerceiverClassifierOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    cross_attentions: tuple[torch.FloatTensor] | None = ...

class PerceiverEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, batch_size: int):  # -> Tensor:
        ...

class PerceiverSelfAttention(nn.Module):
    def __init__(
        self,
        config,
        is_cross_attention=...,
        qk_channels=...,
        v_channels=...,
        num_heads=...,
        q_dim=...,
        kv_dim=...,
    ) -> None: ...
    def transpose_for_scores(self, x, channels_per_head): ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs: torch.FloatTensor | None = ...,
        inputs_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.Tensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor]: ...

class PerceiverSelfOutput(nn.Module):
    def __init__(self, config, input_channels, output_channels) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class PerceiverAttention(nn.Module):
    def __init__(
        self,
        config,
        is_cross_attention=...,
        qk_channels=...,
        v_channels=...,
        num_heads=...,
        q_dim=...,
        kv_dim=...,
        use_query_residual=...,
    ) -> None: ...
    def prune_heads(self, heads):  # -> None:
        ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs: torch.FloatTensor | None = ...,
        inputs_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.Tensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor]: ...

class PerceiverMLP(nn.Module):
    def __init__(self, config, input_size, widening_factor) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class PerceiverLayer(nn.Module):
    def __init__(
        self,
        config,
        is_cross_attention=...,
        qk_channels=...,
        v_channels=...,
        num_heads=...,
        q_dim=...,
        kv_dim=...,
        widening_factor=...,
        use_query_residual=...,
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs: torch.FloatTensor | None = ...,
        inputs_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.Tensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor]: ...
    def feed_forward_chunk(self, attention_output):  # -> Any:
        ...

class PerceiverEncoder(nn.Module):
    def __init__(self, config, kv_dim=...) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs: torch.FloatTensor | None = ...,
        inputs_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutputWithCrossAttentions: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple | BaseModelOutputWithCrossAttentions: ...

@auto_docstring
class PerceiverPreTrainedModel(PreTrainedModel):
    config: PerceiverConfig
    base_model_prefix = ...
    main_input_name = ...

@auto_docstring(custom_intro=...)
class PerceiverModel(PerceiverPreTrainedModel):
    def __init__(
        self,
        config,
        decoder: PerceiverAbstractDecoder | None = ...,
        input_preprocessor: PreprocessorType = ...,
        output_postprocessor: PostprocessorType = ...,
    ) -> None: ...
    def get_input_embeddings(self):  # -> Parameter:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        inputs: torch.FloatTensor,
        attention_mask: torch.FloatTensor | None = ...,
        subsampled_output_points: dict[str, torch.Tensor] | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
        return_dict: bool | None = ...,
    ) -> tuple | PerceiverModelOutput: ...

@auto_docstring(custom_intro=...)
class PerceiverForMaskedLM(PerceiverPreTrainedModel):
    def __init__(self, config: PerceiverConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        inputs: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        labels: torch.Tensor | None = ...,
        return_dict: bool | None = ...,
        input_ids: torch.Tensor | None = ...,
    ) -> tuple | PerceiverMaskedLMOutput: ...

@auto_docstring(custom_intro=...)
class PerceiverForSequenceClassification(PerceiverPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        inputs: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        labels: torch.Tensor | None = ...,
        return_dict: bool | None = ...,
        input_ids: torch.Tensor | None = ...,
    ) -> tuple | PerceiverClassifierOutput: ...

@auto_docstring(custom_intro=...)
class PerceiverForImageClassificationLearned(PerceiverPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        inputs: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        labels: torch.Tensor | None = ...,
        interpolate_pos_encoding: bool = ...,
        return_dict: bool | None = ...,
        pixel_values: torch.Tensor | None = ...,
    ) -> tuple | PerceiverClassifierOutput: ...

@auto_docstring(custom_intro=...)
class PerceiverForImageClassificationFourier(PerceiverPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        inputs: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        labels: torch.Tensor | None = ...,
        return_dict: bool | None = ...,
        pixel_values: torch.Tensor | None = ...,
    ) -> tuple | PerceiverClassifierOutput: ...

@auto_docstring(custom_intro=...)
class PerceiverForImageClassificationConvProcessing(PerceiverPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        inputs: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        labels: torch.Tensor | None = ...,
        return_dict: bool | None = ...,
        pixel_values: torch.Tensor | None = ...,
    ) -> tuple | PerceiverClassifierOutput: ...

@auto_docstring(custom_intro=...)
class PerceiverForOpticalFlow(PerceiverPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        inputs: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        labels: torch.Tensor | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | PerceiverClassifierOutput: ...

@auto_docstring(custom_intro=...)
class PerceiverForMultimodalAutoencoding(PerceiverPreTrainedModel):
    def __init__(self, config: PerceiverConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        inputs: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        subsampled_output_points: dict[str, torch.Tensor] | None = ...,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        labels: torch.Tensor | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | PerceiverClassifierOutput: ...

def build_position_encoding(
    position_encoding_type,
    out_channels=...,
    project_pos_dim=...,
    trainable_position_encoding_kwargs=...,
    fourier_position_encoding_kwargs=...,
):  # -> tuple[PerceiverTrainablePositionEncoding | PerceiverFourierPositionEncoding, Linear | Identity]:
    ...

class PerceiverAbstractDecoder(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def decoder_query(
        self, inputs, modality_sizes=..., inputs_without_pos=..., subsampled_points=...
    ): ...
    @property
    @abc.abstractmethod
    def num_query_channels(self): ...
    @abc.abstractmethod
    def forward(self, query, z, query_mask=...): ...

class PerceiverProjectionDecoder(PerceiverAbstractDecoder):
    def __init__(self, config) -> None: ...
    def decoder_query(
        self, inputs, modality_sizes=..., inputs_without_pos=..., subsampled_points=...
    ):  # -> None:
        ...
    def forward(
        self,
        query: torch.Tensor,
        z: torch.FloatTensor,
        query_mask: torch.FloatTensor | None = ...,
    ) -> torch.FloatTensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.FloatTensor: ...

class PerceiverBasicDecoder(PerceiverAbstractDecoder):
    def __init__(
        self,
        config: PerceiverConfig,
        output_num_channels: int,
        position_encoding_type: str | None = ...,
        output_index_dims: int | None = ...,
        num_channels: int | None = ...,
        subsampled_index_dims: int | None = ...,
        qk_channels: int | None = ...,
        v_channels: int | None = ...,
        num_heads: int | None = ...,
        widening_factor: int | None = ...,
        use_query_residual: bool | None = ...,
        concat_preprocessed_input: bool | None = ...,
        final_project: bool | None = ...,
        position_encoding_only: bool | None = ...,
        **position_encoding_kwargs,
    ) -> None: ...
    @property
    def num_query_channels(self) -> int: ...
    def decoder_query(
        self, inputs, modality_sizes=..., inputs_without_pos=..., subsampled_points=...
    ):  # -> Tensor | Any:
        ...
    def forward(
        self,
        query: torch.Tensor,
        z: torch.FloatTensor,
        query_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
    ) -> PerceiverDecoderOutput: ...
    def __call__(self, *args: Any, **kwargs: Any) -> PerceiverDecoderOutput: ...

class PerceiverClassificationDecoder(PerceiverAbstractDecoder):
    def __init__(self, config, **decoder_kwargs) -> None: ...
    @property
    def num_query_channels(self) -> int: ...
    def decoder_query(
        self, inputs, modality_sizes=..., inputs_without_pos=..., subsampled_points=...
    ):  # -> Tensor | Any:
        ...
    def forward(
        self,
        query: torch.Tensor,
        z: torch.FloatTensor,
        query_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
    ) -> PerceiverDecoderOutput: ...
    def __call__(self, *args: Any, **kwargs: Any) -> PerceiverDecoderOutput: ...

class PerceiverOpticalFlowDecoder(PerceiverAbstractDecoder):
    def __init__(
        self,
        config,
        output_image_shape,
        output_num_channels=...,
        rescale_factor=...,
        **decoder_kwargs,
    ) -> None: ...
    @property
    def num_query_channels(self) -> int: ...
    def decoder_query(
        self, inputs, modality_sizes=..., inputs_without_pos=..., subsampled_points=...
    ): ...
    def forward(
        self,
        query: torch.Tensor,
        z: torch.FloatTensor,
        query_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
    ) -> PerceiverDecoderOutput: ...
    def __call__(self, *args: Any, **kwargs: Any) -> PerceiverDecoderOutput: ...

class PerceiverBasicVideoAutoencodingDecoder(PerceiverAbstractDecoder):
    def __init__(
        self,
        config: PerceiverConfig,
        output_shape: list[int],
        position_encoding_type: str,
        **decoder_kwargs,
    ) -> None: ...
    @property
    def num_query_channels(self) -> int: ...
    def decoder_query(
        self, inputs, modality_sizes=..., inputs_without_pos=..., subsampled_points=...
    ):  # -> Tensor | Any:
        ...
    def forward(
        self,
        query: torch.Tensor,
        z: torch.FloatTensor,
        query_mask: torch.FloatTensor | None = ...,
    ) -> PerceiverDecoderOutput: ...
    def __call__(self, *args: Any, **kwargs: Any) -> PerceiverDecoderOutput: ...

def restructure(
    modality_sizes: ModalitySizeType, inputs: torch.Tensor
) -> Mapping[str, torch.Tensor]: ...

class PerceiverMultimodalDecoder(PerceiverAbstractDecoder):
    def __init__(
        self,
        config: PerceiverConfig,
        modalities: dict[str, PerceiverAbstractDecoder],
        num_outputs: int,
        output_num_channels: int,
        min_padding_size: int | None = ...,
        subsampled_index_dims: dict[str, PerceiverAbstractDecoder] | None = ...,
        **decoder_kwargs,
    ) -> None: ...
    @property
    def num_query_channels(self) -> int: ...
    def decoder_query(
        self, inputs, modality_sizes, inputs_without_pos=..., subsampled_points=...
    ):  # -> Tensor:
        ...
    def forward(
        self,
        query: torch.Tensor,
        z: torch.FloatTensor,
        query_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

def space_to_depth(
    frames: torch.Tensor, temporal_block_size: int = ..., spatial_block_size: int = ...
) -> torch.Tensor: ...

class Conv2dSamePadding(nn.Conv2d):
    def __init__(self, *args, **kwargs) -> None: ...
    def forward(self, input):  # -> Tensor:
        ...

class Conv2DDownsample(nn.Module):
    def __init__(
        self,
        num_layers: int = ...,
        in_channels: int = ...,
        out_channels: int = ...,
        use_batchnorm: bool = ...,
    ) -> None: ...
    def forward(self, inputs: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

def generate_fourier_features(
    pos, num_bands, max_resolution=..., concat_pos=..., sine_only=...
):  # -> Tensor:
    ...
def build_linear_positions(index_dims, output_range=...):  # -> Tensor:
    ...

class PerceiverAbstractPositionEncoding(nn.Module, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def num_dimensions(self) -> int: ...
    @abc.abstractmethod
    def output_size(self, *args, **kwargs) -> int: ...
    @abc.abstractmethod
    def forward(self, batch_size, pos): ...

class PerceiverTrainablePositionEncoding(PerceiverAbstractPositionEncoding):
    def __init__(self, index_dims, num_channels=...) -> None: ...
    @property
    def num_dimensions(self) -> int: ...
    def output_size(self, *args, **kwargs) -> int: ...
    def interpolate_pos_encoding(
        self, position_embeddings: torch.Tensor, height: int, width: int
    ) -> torch.Tensor: ...
    def forward(
        self,
        batch_size: int,
        interpolate_pos_encoding: bool = ...,
        input_size: torch.Size | None = ...,
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class PerceiverFourierPositionEncoding(PerceiverAbstractPositionEncoding):
    def __init__(
        self, num_bands, max_resolution, concat_pos=..., sine_only=...
    ) -> None: ...
    @property
    def num_dimensions(self) -> int: ...
    def output_size(self): ...
    def forward(
        self,
        index_dims: list[int],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        pos: torch.FloatTensor | None = ...,
    ) -> torch.FloatTensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.FloatTensor: ...

class AbstractPreprocessor(nn.Module):
    @property
    def num_channels(self) -> int: ...

class PerceiverTextPreprocessor(AbstractPreprocessor):
    def __init__(self, config: PerceiverConfig) -> None: ...
    @property
    def num_channels(self) -> int: ...
    def forward(
        self,
        inputs: torch.LongTensor,
        pos: torch.Tensor | None = ...,
        network_input_is_1d: bool = ...,
        interpolate_pos_encoding: bool = ...,
    ):  # -> tuple[Any, None, Any]:
        ...

class PerceiverEmbeddingDecoder(nn.Module):
    def __init__(self, config: PerceiverConfig) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, embedding_layer: torch.Tensor
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class PerceiverMultimodalPostprocessor(nn.Module):
    def __init__(
        self, modalities: Mapping[str, PostprocessorType], input_is_dict: bool = ...
    ) -> None: ...
    def forward(
        self,
        inputs: torch.Tensor,
        pos: torch.Tensor | None = ...,
        modality_sizes=...,
    ) -> Mapping[str, torch.Tensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Mapping[str, torch.Tensor]: ...

class PerceiverClassificationPostprocessor(nn.Module):
    def __init__(self, config: PerceiverConfig, in_channels: int) -> None: ...
    def forward(
        self, inputs, pos: torch.Tensor | None = ..., modality_sizes=...
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class PerceiverAudioPostprocessor(nn.Module):
    def __init__(
        self, config: PerceiverConfig, in_channels: int, postproc_type: str = ...
    ) -> None: ...
    def forward(
        self,
        inputs: torch.Tensor,
        pos: torch.Tensor | None = ...,
        modality_sizes=...,
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class PerceiverProjectionPostprocessor(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None: ...
    def forward(
        self,
        inputs: torch.Tensor,
        pos: torch.Tensor | None = ...,
        modality_sizes=...,
    ) -> torch.Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

class PerceiverImagePreprocessor(AbstractPreprocessor):
    def __init__(
        self,
        config,
        prep_type=...,
        spatial_downsample: int = ...,
        temporal_downsample: int = ...,
        position_encoding_type: str = ...,
        in_channels: int = ...,
        out_channels: int = ...,
        conv_after_patching: bool = ...,
        conv_after_patching_in_channels: int = ...,
        conv2d_use_batchnorm: bool = ...,
        concat_or_add_pos: str = ...,
        project_pos_dim: int = ...,
        **position_encoding_kwargs,
    ) -> None: ...
    @property
    def num_channels(self) -> int: ...
    def forward(
        self,
        inputs: torch.Tensor,
        pos: torch.Tensor | None = ...,
        network_input_is_1d: bool = ...,
        interpolate_pos_encoding: bool = ...,
    ):  # -> tuple[Tensor, None, Tensor]:
        ...

class PerceiverOneHotPreprocessor(AbstractPreprocessor):
    def __init__(self, config: PerceiverConfig) -> None: ...
    @property
    def num_channels(self) -> int: ...
    def forward(
        self,
        inputs: torch.Tensor,
        pos: torch.Tensor | None = ...,
        network_input_is_1d: bool = ...,
    ):  # -> tuple[Tensor, None, Tensor]:
        ...

class PerceiverAudioPreprocessor(AbstractPreprocessor):
    def __init__(
        self,
        config,
        prep_type: str = ...,
        samples_per_patch: int = ...,
        position_encoding_type: str = ...,
        concat_or_add_pos: str = ...,
        out_channels=...,
        project_pos_dim=...,
        **position_encoding_kwargs,
    ) -> None: ...
    @property
    def num_channels(self) -> int: ...
    def forward(
        self,
        inputs: torch.Tensor,
        pos: torch.Tensor | None = ...,
        network_input_is_1d: bool = ...,
        interpolate_pos_encoding: bool = ...,
    ):  # -> tuple[Tensor | Any, None, Any]:
        ...

class PerceiverMultimodalPreprocessor(AbstractPreprocessor):
    def __init__(
        self,
        modalities: Mapping[str, PreprocessorType],
        mask_probs: Mapping[str, float] | None = ...,
        min_padding_size: int = ...,
    ) -> None: ...
    @property
    def num_channels(self) -> int: ...
    def forward(
        self,
        inputs: Mapping[str, torch.Tensor],
        pos: torch.Tensor | None = ...,
        network_input_is_1d: bool = ...,
        interpolate_pos_encoding: bool = ...,
    ) -> PreprocessorOutputType: ...
    def __call__(self, *args: Any, **kwargs: Any) -> PreprocessorOutputType: ...

__all__ = [
    "PerceiverForImageClassificationConvProcessing",
    "PerceiverForImageClassificationFourier",
    "PerceiverForImageClassificationLearned",
    "PerceiverForMaskedLM",
    "PerceiverForMultimodalAutoencoding",
    "PerceiverForOpticalFlow",
    "PerceiverForSequenceClassification",
    "PerceiverLayer",
    "PerceiverModel",
    "PerceiverPreTrainedModel",
]
