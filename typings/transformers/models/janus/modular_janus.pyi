from collections.abc import Iterable
from dataclasses import dataclass

from torch import nn
from transformers.models.blip.image_processing_blip import BlipImageProcessor

import numpy as np
import PIL
import torch

from ..blip_2.modeling_blip_2 import Blip2VisionModel
from ..chameleon.configuration_chameleon import ChameleonVQVAEConfig
from ..chameleon.modeling_chameleon import (
    ChameleonVQVAE,
    ChameleonVQVAEEncoderAttnBlock,
    ChameleonVQVAEEncoderConvDownsample,
    ChameleonVQVAEEncoderResnetBlock,
    ChameleonVQVAEVectorQuantizer,
)
from ..idefics.modeling_idefics import (
    IdeficsBaseModelOutputWithPast,
    IdeficsCausalLMOutputWithPast,
)
from ..siglip.configuration_siglip import SiglipVisionConfig
from ..siglip.modeling_siglip import (
    SiglipEncoder,
    SiglipEncoderLayer,
    SiglipVisionEmbeddings,
)
from ...cache_utils import Cache
from ...configuration_utils import PretrainedConfig
from ...generation import GenerationMixin, LogitsProcessorList
from ...image_utils import ChannelDimension, ImageInput, PILImageResampling
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    filter_out_non_signature_kwargs,
)

logger = ...

class JanusVisionConfig(SiglipVisionConfig):
    model_type = ...
    base_config_key = ...
    def __init__(
        self,
        hidden_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        num_channels=...,
        patch_size=...,
        image_size=...,
        attention_dropout=...,
        layer_norm_eps=...,
        hidden_act=...,
        mlp_ratio=...,
        attention_bias=...,
        hidden_dropout_rate=...,
        projection_dim=...,
        projection_dropout=...,
        use_qk_norm=...,
        initializer_range=...,
        depth=...,
        num_image_tokens=...,
        **kwargs,
    ) -> None: ...

class JanusVQVAEConfig(ChameleonVQVAEConfig):
    def __init__(
        self,
        embed_dim: int = ...,
        num_embeddings: int = ...,
        double_latent: bool = ...,
        latent_channels: int = ...,
        num_patches: int = ...,
        in_channels: int = ...,
        out_channels: int = ...,
        base_channels: int = ...,
        channel_multiplier: list[int] = ...,
        num_res_blocks: int = ...,
        dropout: float = ...,
        initializer_range=...,
        projection_dim=...,
        num_hidden_layers=...,
        hidden_act=...,
        image_token_embed_dim=...,
        **kwargs,
    ) -> None: ...

class JanusConfig(PretrainedConfig):
    model_type = ...
    sub_configs = ...
    def __init__(
        self,
        text_config=...,
        vision_config=...,
        vq_config=...,
        image_token_id=...,
        **kwargs,
    ) -> None: ...

@auto_docstring
class JanusPreTrainedModel(PreTrainedModel):
    config: JanusConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _can_compile_fullgraph = ...
    _supports_param_buffer_assignment = ...

@dataclass
@auto_docstring(custom_intro=...)
class JanusVQVAEOutput(ModelOutput):
    decoded_pixel_values: torch.FloatTensor | None = ...
    embedding_loss: torch.FloatTensor | None = ...

class JanusBaseModelOutputWithPast(IdeficsBaseModelOutputWithPast): ...
class JanusCausalLMOutputWithPast(IdeficsCausalLMOutputWithPast): ...

class JanusVisionEmbeddings(SiglipVisionEmbeddings):
    def forward(
        self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = ...
    ) -> torch.Tensor: ...

class JanusVisionAttention(nn.Module):
    def __init__(self, config: JanusVisionConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ):  # -> tuple[Any, Any]:
        ...

class JanusVisionMLP(nn.Module):
    def __init__(self, config: JanusVisionConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class JanusVisionEncoderLayer(SiglipEncoderLayer):
    def __init__(self, config: JanusVisionConfig) -> None: ...

class JanusVisionEncoder(SiglipEncoder):
    def __init__(self, config: JanusVisionConfig) -> None: ...

class JanusVisionModel(Blip2VisionModel):
    def __init__(self, config: JanusVisionConfig) -> None: ...

class JanusVisionAlignerMLP(nn.Module):
    def __init__(self, config: JanusVisionConfig) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class JanusVQVAEVectorQuantizer(ChameleonVQVAEVectorQuantizer):
    def __init__(self, config: JanusVQVAEConfig) -> None: ...
    def get_codebook_entry(
        self, image_tokens: torch.LongTensor
    ) -> torch.FloatTensor: ...

class JanusVQVAEResnetBlock(ChameleonVQVAEEncoderResnetBlock): ...
class JanusVQVAEAttnBlock(ChameleonVQVAEEncoderAttnBlock): ...
class JanusVQVAEConvDownsample(ChameleonVQVAEEncoderConvDownsample): ...

class JanusVQVAEConvUpsample(nn.Module):
    def __init__(self, in_channels) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class JanusVQVAEMidBlock(nn.Module):
    def __init__(self, config: JanusVQVAEConfig, channels: int) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class JanusVQVAEEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, pixel_values: torch.LongTensor):  # -> Any:
        ...

class JanusVQVAEDecoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_state: torch.FloatTensor) -> torch.FloatTensor: ...

class JanusVQVAE(ChameleonVQVAE):
    _no_split_modules = ...
    main_input_name = ...
    def __init__(self, config: JanusVQVAEConfig) -> None: ...
    def decode(self, image_tokens: torch.LongTensor) -> torch.FloatTensor: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self, pixel_values: torch.FloatTensor
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]: ...

class JanusVQVAEAlignerMLP(nn.Module):
    def __init__(self, config: JanusVQVAEConfig) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class JanusVQVAEHead(nn.Module):
    def __init__(self, config: JanusVQVAEConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.tensor: ...

@auto_docstring(custom_intro=...)
class JanusModel(JanusPreTrainedModel):
    def __init__(self, config: JanusConfig) -> None: ...
    def get_input_embeddings(self):  # -> Any:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_image_features(self, pixel_values):  # -> Any:
        ...
    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor,
    ):  # -> Tensor | Any:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs,
    ):  # -> JanusBaseModelOutputWithPast:
        ...

class JanusForConditionalGeneration(JanusPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    _can_compile_fullgraph = ...
    def __init__(self, config: JanusConfig) -> None: ...
    def get_input_embeddings(self):  # -> Any:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def prepare_embeddings_for_image_generation(
        self, inputs: torch.Tensor
    ) -> torch.Tensor: ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs: Unpack[TransformersKwargs],
    ):  # -> JanusCausalLMOutputWithPast:
        ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        pixel_values=...,
        past_key_values=...,
        attention_mask=...,
        inputs_embeds=...,
        cache_position=...,
        logits_to_keep=...,
        **kwargs,
    ):  # -> dict[Any, Any]:
        ...
    def decode_image_tokens(self, image_tokens: torch.Tensor):  # -> Tensor:
        ...
    @torch.no_grad
    def generate(
        self,
        inputs: torch.Tensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        logits_processor: LogitsProcessorList | None = ...,
        **kwargs,
    ):  # -> GenerateOutput | LongTensor | Tensor:
        ...

class JanusImageProcessor(BlipImageProcessor):
    def __init__(
        self,
        do_resize: bool = ...,
        size: dict[str, int] | None = ...,
        min_size: int = ...,
        resample: PILImageResampling = ...,
        do_rescale: bool = ...,
        rescale_factor: float = ...,
        do_normalize: bool = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        do_convert_rgb: bool | None = ...,
        do_pad: bool | None = ...,
        **kwargs,
    ) -> None: ...
    def pad_to_square(
        self,
        image: np.ndarray,
        background_color: int | tuple[int, int, int] = ...,
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ) -> np.ndarray: ...
    def resize(
        self,
        image: np.ndarray,
        size: dict[str, int] | int,
        resample: PILImageResampling = ...,
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
        **kwargs,
    ) -> np.ndarray: ...
    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool | None = ...,
        size: dict[str, int] | None = ...,
        resample: PILImageResampling | None = ...,
        do_rescale: bool | None = ...,
        rescale_factor: float | None = ...,
        do_normalize: bool | None = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        return_tensors: str | TensorType | None = ...,
        do_convert_rgb: bool | None = ...,
        background_color: int | tuple[int, int, int] | None = ...,
        do_pad: bool | None = ...,
        data_format: ChannelDimension = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ) -> PIL.Image.Image: ...
    def postprocess(
        self,
        images: ImageInput,
        do_rescale: bool | None = ...,
        rescale_factor: float | None = ...,
        do_normalize: bool | None = ...,
        image_mean: list[float] | None = ...,
        image_std: list[float] | None = ...,
        input_data_format: str | None = ...,
        return_tensors: str | None = ...,
    ):  # -> ImageInput | Any | BatchFeature:
        ...
    def unnormalize(
        self,
        image: np.ndarray,
        image_mean: float | Iterable[float],
        image_std: float | Iterable[float],
        input_data_format: str | ChannelDimension | None = ...,
    ) -> np.ndarray: ...

__all__ = [
    "JanusConfig",
    "JanusForConditionalGeneration",
    "JanusImageProcessor",
    "JanusModel",
    "JanusPreTrainedModel",
    "JanusVQVAE",
    "JanusVQVAEConfig",
    "JanusVisionConfig",
    "JanusVisionModel",
]
