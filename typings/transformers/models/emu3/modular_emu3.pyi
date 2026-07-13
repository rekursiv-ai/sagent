from functools import cached_property

from torch import nn

import torch

from .configuration_emu3 import Emu3Config, Emu3TextConfig, Emu3VQVAEConfig
from ..chameleon.modeling_chameleon import (
    ChameleonPreTrainedModel,
    ChameleonVQVAEEncoderConvDownsample,
)
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    TransformersKwargs,
)
from ..siglip.modeling_siglip import SiglipAttention
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_outputs import CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import auto_docstring, can_return_tuple
from ...utils.deprecation import deprecate_kwarg

logger = ...

class Emu3Attention(LlamaAttention): ...

class Emu3DecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: Emu3Config, layer_idx: int) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor: ...

class Emu3VQVAEVectorQuantizer(nn.Module):
    def __init__(self, config: Emu3VQVAEConfig) -> None: ...
    def forward(self, hidden_state: torch.Tensor):  # -> Tensor:
        ...

class Emu3VQVAEEncoderConvDownsample(ChameleonVQVAEEncoderConvDownsample): ...

class Emu3VQVAEEncoderConvUpsample(nn.Module):
    def __init__(self, in_channels) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class Emu3VQVAEConv3d(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: tuple[int],
        stride: tuple[int],
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor):  # -> Tensor:
        ...

class Emu3VQVAESpatialNorm(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, quant_states: torch.Tensor
    ):  # -> Tensor:
        ...

class Emu3VQVAETemporalUpsample(nn.Module):
    def __init__(self, in_channel: int, out_channel: int) -> None: ...
    def forward(self, hidden_states: torch.Tensor):  # -> Tensor:
        ...

class Emu3VQVAETemporalDownsample(nn.Module):
    def __init__(self, in_channel: int, out_channel: int) -> None: ...
    def forward(self, hidden_states: torch.Tensor):  # -> Tensor:
        ...

class Emu3VQVAETemporalResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=...) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class Emu3VQVAEResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = ...,
        quant_channels: int | None = ...,
    ) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, quant_channels: torch.Tensor | None = ...
    ):  # -> Any | Tensor:
        ...

class Emu3VQVAEAttentionBlock(SiglipAttention):
    def __init__(self, config: Emu3VQVAEConfig) -> None: ...

class Emu3VQVAEGroupNorm(nn.GroupNorm):
    def __init__(self, **kwargs) -> None: ...
    def forward(self, input, quant_states=...):  # -> Tensor:
        ...

class Emu3VQVAEMiddleBlock(nn.Module):
    def __init__(self, config, in_channels, quant_channels=...) -> None: ...
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        quant_states: torch.FloatTensor | None = ...,
    ):  # -> FloatTensor:
        ...

class Emu3VQVAEDownBlock(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.FloatTensor):  # -> FloatTensor:
        ...

class Emu3VQVAEUpBlock(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: torch.FloatTensor, quant_states: torch.FloatTensor
    ):  # -> FloatTensor:
        ...

class Emu3VQVAEEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, pixel_values: torch.LongTensor):  # -> Any:
        ...

class Emu3VQVAEDecoder(nn.Module):
    def __init__(self, config: Emu3VQVAEConfig) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, quant_states: torch.Tensor
    ):  # -> Tensor:
        ...

@auto_docstring(custom_intro=...)
class Emu3VQVAE(PreTrainedModel):
    config: Emu3VQVAEConfig
    base_model_prefix = ...
    main_input_name = ...
    _supports_sdpa = ...
    _supports_flash_attn = ...
    _supports_flex_attn = ...
    _supports_attention_backend = ...
    _no_split_modules = ...
    def __init__(self, config: Emu3VQVAEConfig) -> None: ...
    def encode(
        self, pixel_values: torch.Tensor, image_sizes: torch.Tensor
    ):  # -> list[Any]:
        ...
    def decode(self, hidden_states: torch.Tensor):  # -> Any:
        ...

class Emu3ImageVocabularyMapping:
    def __init__(self, vocab_map) -> None: ...
    @cached_property
    def image_tokens(self):  # -> list[Any]:
        ...
    @cached_property
    def image_tokens_str(self):  # -> list[Any]:
        ...
    @cached_property
    def img2bpe(self):  # -> dict[int, Any]:
        ...
    @cached_property
    def bpe2img(self):  # -> dict[Any, int]:
        ...
    @cached_property
    def bpe2img_mapping_tensor(self):  # -> Tensor:
        ...
    @cached_property
    def img2bpe_mapping_tensor(self):  # -> Tensor:
        ...
    def convert_img2bpe(self, img_batch: list[torch.Tensor]) -> torch.Tensor: ...
    def convert_bpe2img(self, img_batch: torch.Tensor) -> torch.Tensor: ...

class Emu3PreTrainedModel(ChameleonPreTrainedModel, Emu3VQVAE):
    _no_split_modules = ...
    _supports_flex_attn = ...
    _supports_attention_backend = ...

class Emu3TextModel(LlamaModel, Emu3PreTrainedModel):
    _can_record_outputs = ...
    def __init__(self, config: Emu3Config) -> None: ...

class Emu3ForCausalLM(LlamaForCausalLM, Emu3PreTrainedModel, GenerationMixin):
    config: Emu3TextConfig
    def __init__(self, config) -> None: ...
    def forward(**super_kwargs):  # -> None:
        ...

class Emu3Model(Emu3PreTrainedModel):
    _checkpoint_conversion_mapping = ...
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> Emu3TextModel:
        ...
    def get_image_tokens(
        self, pixel_values: torch.FloatTensor, image_sizes: torch.LongTensor
    ):  # -> Tensor:
        ...
    def get_image_features(
        self, pixel_values: torch.FloatTensor, image_sizes: torch.LongTensor
    ):  # -> tuple[Tensor, ...]:
        ...
    @torch.no_grad
    def decode_image_tokens(
        self, image_tokens: torch.LongTensor, height: int, width: int
    ):  # -> Any:
        ...
    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor,
    ):  # -> Any:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        image_sizes: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | CausalLMOutputWithPast: ...

class Emu3ForConditionalGeneration(Emu3PreTrainedModel, GenerationMixin):
    base_model_prefix = ...
    _tied_weights_keys = ...
    _checkpoint_conversion_mapping = ...
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_output_embeddings(self) -> nn.Module: ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> Emu3TextModel:
        ...
    @property
    def text_model(self):  # -> Emu3TextModel:
        ...
    @property
    def vqmodel(self):  # -> Emu3VQVAE:
        ...
    @property
    def vocabulary_mapping(self):  # -> Emu3ImageVocabularyMapping:
        ...
    def decode_image_tokens(self, **kwargs):  # -> Any:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        image_sizes: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | CausalLMOutputWithPast: ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=...,
        attention_mask=...,
        inputs_embeds=...,
        cache_position=...,
        position_ids=...,
        use_cache=...,
        pixel_values=...,
        **kwargs,
    ):  # -> dict[Any, Any]:
        ...

__all__ = [
    "Emu3ForCausalLM",
    "Emu3ForConditionalGeneration",
    "Emu3Model",
    "Emu3PreTrainedModel",
    "Emu3TextModel",
    "Emu3VQVAE",
]
