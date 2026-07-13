from dataclasses import dataclass

from torch import nn

import torch

from .configuration_trajectory_transformer import TrajectoryTransformerConfig
from ....cache_utils import Cache
from ....modeling_layers import GradientCheckpointingLayer
from ....modeling_utils import PreTrainedModel
from ....utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

"""PyTorch TrajectoryTransformer model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...

def load_tf_weights_in_trajectory_transformer(model, config, tf_checkpoint_path): ...

@dataclass
class TrajectoryTransformerOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

class TrajectoryTransformerPreTrainedModel(PreTrainedModel):
    config: TrajectoryTransformerConfig
    load_tf_weights = ...
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...

TRAJECTORY_TRANSFORMER_START_DOCSTRING = ...
TRAJECTORY_TRANSFORMER_INPUTS_DOCSTRING = ...

class EinLinear(nn.Module):
    def __init__(self, n_models, in_features, out_features, bias) -> None: ...
    def reset_parameters(self):  # -> None:
        ...
    def forward(self, input):  # -> Tensor:
        ...

class CausalSelfAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states: tuple[torch.FloatTensor] | None,
        layer_past: tuple[torch.Tensor] | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
    ):  # -> tuple[Any, tuple[Tensor | Any, Tensor | Any] | None, Any] | tuple[Any, tuple[Tensor | Any, Tensor | Any] | None]:
        ...

class Block(GradientCheckpointingLayer):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states: tuple[torch.FloatTensor] | None,
        layer_past: tuple[torch.Tensor] | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
    ):  # -> Any:
        ...

@add_start_docstrings(
    ...,
    TRAJECTORY_TRANSFORMER_START_DOCSTRING,
)
class TrajectoryTransformerModel(TrajectoryTransformerPreTrainedModel):
    def __init__(self, config) -> None: ...
    def get_block_size(self): ...
    def offset_tokens(self, trajectories): ...
    def pad_to_full_observation(self, hidden_states):  # -> tuple[Tensor, Any]:
        ...
    @add_start_docstrings_to_model_forward(
        TRAJECTORY_TRANSFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(
        output_type=TrajectoryTransformerOutput, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        trajectories: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        targets: torch.FloatTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | TrajectoryTransformerOutput: ...

__all__ = [
    "TrajectoryTransformerModel",
    "TrajectoryTransformerPreTrainedModel",
    "load_tf_weights_in_trajectory_transformer",
]
