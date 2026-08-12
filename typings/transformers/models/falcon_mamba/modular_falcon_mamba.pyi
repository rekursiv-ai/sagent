from mambapy.pscan import pscan

import torch

from ..mamba.configuration_mamba import MambaConfig
from ..mamba.modeling_mamba import (
    MambaBlock,
    MambaCache,
    MambaCausalLMOutput,
    MambaForCausalLM,
    MambaMixer,
    MambaModel,
    MambaOutput,
    MambaPreTrainedModel,
    MambaRMSNorm,
)
from ...utils import auto_docstring
from ...utils.import_utils import is_mamba_ssm_available, is_mambapy_available

"""PyTorch FALCONMAMBA model."""
logger = ...
if is_mambapy_available(): ...
else:
    pscan = ...
if is_mamba_ssm_available(): ...
else: ...
_causal_conv1d_cache = ...

class FalconMambaConfig(MambaConfig):
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        state_size=...,
        num_hidden_layers=...,
        layer_norm_epsilon=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        expand=...,
        conv_kernel=...,
        use_bias=...,
        use_conv_bias=...,
        hidden_act=...,
        initializer_range=...,
        residual_in_fp32=...,
        time_step_rank=...,
        time_step_scale=...,
        time_step_min=...,
        time_step_max=...,
        time_step_init_scheme=...,
        time_step_floor=...,
        rescale_prenorm_residual=...,
        use_cache=...,
        use_falcon_mambapy=...,
        mixer_rms_eps=...,
        **kwargs,
    ) -> None: ...

class FalconMambaCache(MambaCache): ...

def rms_forward(hidden_states, variance_epsilon=...): ...

class FalconMambaMixer(MambaMixer):
    def warn_slow_implementation(self):  # -> None:
        ...
    def __init__(self, config: FalconMambaConfig, layer_idx: int) -> None: ...
    def cuda_kernels_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: FalconMambaCache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
    ):  # -> None:
        ...
    def slow_forward(
        self,
        input_states,
        cache_params: FalconMambaCache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
    ):  # -> Any:
        ...
    def forward(
        self,
        hidden_states,
        cache_params: FalconMambaCache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
    ):  # -> None:
        ...

class FalconMambaRMSNorm(MambaRMSNorm):
    def forward(self, hidden_states): ...

class FalconMambaBlock(MambaBlock): ...

@auto_docstring
class FalconMambaPreTrainedModel(MambaPreTrainedModel): ...

class FalconMambaOutput(MambaOutput): ...
class FalconMambaCausalLMOutput(MambaCausalLMOutput): ...

class FalconMambaModel(MambaModel, FalconMambaPreTrainedModel):
    def __init__(self, config) -> None: ...
    def load_hook(self, state_dict, prefix, *args): ...

class FalconMambaForCausalLM(MambaForCausalLM): ...

__all__ = [
    "FalconMambaCache",
    "FalconMambaConfig",
    "FalconMambaForCausalLM",
    "FalconMambaModel",
    "FalconMambaPreTrainedModel",
]
