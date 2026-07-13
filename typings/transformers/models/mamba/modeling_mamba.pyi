from dataclasses import dataclass

from mambapy.pscan import pscan
from torch import nn

import torch

from .configuration_mamba import MambaConfig
from ...configuration_utils import PretrainedConfig
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring
from ...utils.import_utils import is_mamba_ssm_available, is_mambapy_available

"""PyTorch MAMBA model."""
logger = ...
if is_mambapy_available(): ...
else:
    pscan = ...
if is_mamba_ssm_available(): ...
else: ...
_causal_conv1d_cache = ...

class MambaCache:
    is_compileable = ...
    def __init__(
        self,
        config: PretrainedConfig,
        max_batch_size: int,
        dtype: torch.dtype = ...,
        device: torch.device | str | None = ...,
    ) -> None: ...
    def update_conv_state(
        self,
        layer_idx: int,
        new_conv_state: torch.Tensor,
        cache_position: torch.LongTensor,
    ) -> torch.Tensor: ...
    def update_ssm_state(
        self, layer_idx: int, new_ssm_state: torch.Tensor
    ):  # -> Tensor:
        ...
    def reset(self):  # -> None:
        ...

class MambaMixer(nn.Module):
    def __init__(self, config: MambaConfig, layer_idx: int) -> None: ...
    def warn_slow_implementation(self):  # -> None:
        ...
    def cuda_kernels_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: MambaCache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
    ):  # -> Any:
        ...
    def slow_forward(
        self,
        input_states,
        cache_params: MambaCache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
    ):  # -> Any:
        ...
    def forward(
        self,
        hidden_states,
        cache_params: MambaCache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
    ):  # -> Any:
        ...

class MambaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, hidden_states): ...
    def extra_repr(self):  # -> str:
        ...

class MambaBlock(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx) -> None: ...
    def forward(
        self,
        hidden_states,
        cache_params: MambaCache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
    ): ...

@auto_docstring
class MambaPreTrainedModel(PreTrainedModel):
    config: MambaConfig
    base_model_prefix = ...
    _no_split_modules = ...
    supports_gradient_checkpointing = ...
    _is_stateful = ...

@dataclass
@auto_docstring(
    custom_intro="""
    Class for the MAMBA model outputs.
    """
)
class MambaOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = ...
    cache_params: MambaCache | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...

@dataclass
@auto_docstring(custom_intro=...)
class MambaCausalLMOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    cache_params: MambaCache | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...

@auto_docstring
class MambaModel(MambaPreTrainedModel):
    def __init__(self, config) -> None: ...
    def load_hook(self, state_dict, prefix, *args):  # -> None:
        ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        inputs_embeds: torch.LongTensor | None = ...,
        cache_params: MambaCache | None = ...,
        use_cache: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
    ) -> tuple | MambaOutput: ...

@auto_docstring(custom_intro=...)
class MambaForCausalLM(MambaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        inputs_embeds=...,
        use_cache=...,
        cache_params: MambaCache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        **kwargs,
    ):  # -> dict[str, Any]:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        cache_params: MambaCache | None = ...,
        labels: torch.LongTensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
        **kwargs,
    ) -> tuple | MambaCausalLMOutput: ...

__all__ = ["MambaCache", "MambaForCausalLM", "MambaModel", "MambaPreTrainedModel"]
