from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from torch import nn
from xlstm.xlstm_large.model import mLSTMStateType

import torch

from .configuration_xlstm import xLSTMConfig
from ...generation import GenerationMixin
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring, can_return_tuple, is_xlstm_available

"""PyTorch xLSTM Model."""
if is_xlstm_available():
    external_xlstm = ...
else:
    type mLSTMLayerStateType = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    mLSTMStateType = ...
    external_xlstm = ...
    def soft_cap(
        values: torch.Tensor, cap_value: float | torch.Tensor | None = ...
    ) -> torch.Tensor: ...
    def mlstm_chunkwise_recurrent_fw_C(
        matK: torch.Tensor,
        matV: torch.Tensor,
        vecB: torch.Tensor,
        vecI: torch.Tensor,
        matC_states: torch.Tensor | None = ...,
        vecN_states: torch.Tensor | None = ...,
        scaMinter_states: torch.Tensor | None = ...,
        matC_initial: torch.Tensor | None = ...,
        vecN_initial: torch.Tensor | None = ...,
        scaMinter_initial: torch.Tensor | None = ...,
        qk_scale: float | None = ...,
        chunk_size: int = ...,
        num_chunks: int = ...,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def mlstm_chunkwise_parallel_fw_H(
        matQ: torch.Tensor,
        matK: torch.Tensor,
        matV: torch.Tensor,
        matC_states: torch.Tensor,
        vecN_states: torch.Tensor,
        scaMinter_states: torch.Tensor,
        vecI: torch.Tensor,
        vecB: torch.Tensor,
        qk_scale: float,
        chunk_size: int = ...,
        num_chunks: int = ...,
        eps: float = ...,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def mlstm_chunkwise_fw(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        igate: torch.Tensor,
        fgate: torch.Tensor,
        cstate: torch.Tensor | None = ...,
        nstate: torch.Tensor | None = ...,
        mstate: torch.Tensor | None = ...,
        qk_scale: float | None = ...,
        return_last_states: bool = ...,
        return_all_states: bool = ...,
        chunk_size: int = ...,
        eps: float = ...,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    ]: ...
    def mlstm_chunkwise_native_autograd(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        igate: torch.Tensor,
        fgate: torch.Tensor,
        c_initial: torch.Tensor | None = ...,
        n_initial: torch.Tensor | None = ...,
        m_initial: torch.Tensor | None = ...,
        return_last_states: bool = ...,
        eps: float = ...,
        chunk_size: int = ...,
        **kwargs,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ): ...
    def mlstm_recurrent_step_native(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        igate: torch.Tensor,
        fgate: torch.Tensor,
        cstate: torch.Tensor,
        nstate: torch.Tensor,
        mstate: torch.Tensor,
        eps: float = ...,
        dtype_state: torch.dtype = ...,
        **kwargs,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: ...
    def mlstm_recurrent_sequence_native(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        igate: torch.Tensor,
        fgate: torch.Tensor,
        c_initial: torch.Tensor | None = ...,
        n_initial: torch.Tensor | None = ...,
        m_initial: torch.Tensor | None = ...,
        return_last_states: bool = ...,
        eps: float = ...,
        dtype_state: torch.dtype = ...,
        **kwargs,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    ]: ...
    def wrap_chunkwise_pad_zeros(
        mlstm_chunkwise_kernel: Callable,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        fgate: torch.Tensor,
        igate: torch.Tensor,
        c_initial: torch.Tensor | None = ...,
        n_initial: torch.Tensor | None = ...,
        m_initial: torch.Tensor | None = ...,
        return_last_states: bool = ...,
        eps: float = ...,
        autocast_kernel_dtype: torch.dtype = ...,
        chunk_size: int = ...,
        **kwargs,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ): ...
    def wrap_chunkwise_arbitrary_sequence_length(
        mlstm_chunkwise_kernel: Callable,
        mlstm_sequence_kernel: Callable,
        mlstm_step_kernel: Callable,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        fgate: torch.Tensor,
        igate: torch.Tensor,
        c_initial: torch.Tensor | None = ...,
        n_initial: torch.Tensor | None = ...,
        m_initial: torch.Tensor | None = ...,
        return_last_states: bool = ...,
        eps: float = ...,
        autocast_kernel_dtype: torch.dtype = ...,
        chunk_size: int = ...,
        enable_logging: bool = ...,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ): ...

    class xLSTMBackend(nn.Module):
        config_class = xLSTMConfig
        def __init__(self, config: xLSTMConfig) -> None: ...
        def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            igate: torch.Tensor,
            fgate: torch.Tensor,
            c_initial: torch.Tensor | None = ...,
            n_initial: torch.Tensor | None = ...,
            m_initial: torch.Tensor | None = ...,
            return_last_states: bool = ...,
            mode: Literal["train", "inference"] | None = ...,
        ) -> (
            torch.Tensor
            | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ): ...
        def extra_repr(self) -> str: ...

    class xLSTMRMSNorm(nn.Module):
        def __init__(
            self,
            num_features: int,
            eps: float = ...,
            use_weight: bool = ...,
            use_bias: bool = ...,
            force_float32_reductions: bool = ...,
        ) -> None: ...
        def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    class xLSTMMultiHeadLayerNorm(nn.Module):
        def __init__(
            self,
            num_heads: int,
            head_dim: int,
            eps: float = ...,
            use_weight: bool = ...,
            use_bias: bool = ...,
            force_float32_reductions: bool = ...,
        ) -> None: ...
        def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    class xLSTMFeedForward(nn.Module):
        def __init__(self, config: xLSTMConfig) -> None: ...
        def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    class xLSTMLayer(nn.Module):
        def __init__(self, config: xLSTMConfig) -> None: ...
        def forward(
            self, x: torch.Tensor, state: mLSTMLayerStateType | None = ...
        ) -> tuple[torch.Tensor, mLSTMLayerStateType | None]: ...

    class xLSTMBlock(nn.Module):
        def __init__(self, config: xLSTMConfig) -> None: ...
        def forward(
            self, x: torch.Tensor, state: mLSTMStateType | None = ...
        ) -> tuple[torch.Tensor, mLSTMStateType]: ...

def small_init_method(dim):  # -> Callable[..., Tensor]:
    ...
def wang_init_method(n_layers, dim):  # -> Callable[..., Tensor]:
    ...

class xLSTMPreTrainedModel(PreTrainedModel):
    config_class = xLSTMConfig
    base_model_prefix = ...
    _no_split_modules = ...
    supports_gradient_checkpointing = ...
    _is_stateful = ...

class xLSTMCache:
    def __init__(
        self,
        config: xLSTMConfig,
        max_batch_size: int,
        dtype: torch.dtype = ...,
        device: str | None = ...,
        **kwargs,
    ) -> None: ...
    def reset(self):  # -> None:
        ...

@dataclass
@auto_docstring
class xLSTMOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None
    cache_params: xLSTMCache | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...

@auto_docstring
class xLSTMModel(xLSTMPreTrainedModel):
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, new_embedding):  # -> None:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        inputs_embeds: torch.LongTensor | None = ...,
        cache_params: xLSTMCache | None = ...,
        use_cache: bool | None = ...,
        output_hidden_states: bool | None = ...,
        **kwargs,
    ) -> tuple | xLSTMOutput: ...

@dataclass
@auto_docstring
class xLSTMCausalLMOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    cache_params: xLSTMCache | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...

@auto_docstring
class xLSTMForCausalLM(xLSTMPreTrainedModel, GenerationMixin):
    def __init__(self, config) -> None: ...
    def get_output_embeddings(self):  # -> Linear:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask=...,
        inputs_embeds=...,
        use_cache=...,
        cache_params: xLSTMCache | None = ...,
        **kwargs,
    ):  # -> dict[str, Any]:
        ...
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        cache_params: xLSTMCache | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_hidden_states: bool | None = ...,
        **kwargs,
    ) -> tuple | xLSTMCausalLMOutput: ...

__all__ = ["xLSTMForCausalLM", "xLSTMModel", "xLSTMPreTrainedModel"]
