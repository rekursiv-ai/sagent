import torch

from .configuration_fuyu import FuyuConfig
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_outputs import CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring, can_return_tuple

"""PyTorch Fuyu model."""
logger = ...

@auto_docstring
class FuyuPreTrainedModel(PreTrainedModel):
    config: FuyuConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _supports_attention_backend = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_flex_attn = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...

@auto_docstring(custom_intro=...)
class FuyuModel(FuyuPreTrainedModel):
    _checkpoint_conversion_mapping = ...
    def __init__(self, config: FuyuConfig) -> None: ...
    def get_input_embeddings(self):  # -> Any:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> Any:
        ...
    def gather_continuous_embeddings(
        self,
        word_embeddings: torch.Tensor,
        continuous_embeddings: list[torch.Tensor],
        image_patch_input_indices: torch.Tensor,
    ) -> torch.Tensor: ...
    def get_image_features(
        self, pixel_values: torch.FloatTensor, **kwargs
    ):  # -> list[Any]:
        ...
    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor,
    ):  # -> Tensor | Any:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        image_patches: torch.Tensor | None = ...,
        image_patches_indices: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        **kwargs,
    ) -> tuple | CausalLMOutputWithPast: ...

@auto_docstring(custom_intro=...)
class FuyuForCausalLM(FuyuPreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = ...
    _tied_weights_keys = ...
    def __init__(self, config: FuyuConfig) -> None: ...
    def get_input_embeddings(self):  # -> Any:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> Any:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        image_patches: torch.Tensor | None = ...,
        image_patches_indices: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        labels: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        logits_to_keep: int | None = ...,
        **kwargs,
    ) -> tuple | CausalLMOutputWithPast: ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=...,
        attention_mask=...,
        inputs_embeds=...,
        image_patches=...,
        image_patches_indices=...,
        cache_position=...,
        **kwargs,
    ):  # -> dict[Any, Any]:
        ...

__all__ = ["FuyuForCausalLM", "FuyuModel", "FuyuPreTrainedModel"]
