from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import torch

"""
This file includes public APIs for FSDP such as the classes used for the
constructor arguments.
"""
__all__ = [
    "BackwardPrefetch",
    "CPUOffload",
    "FullOptimStateDictConfig",
    "FullStateDictConfig",
    "LocalOptimStateDictConfig",
    "LocalStateDictConfig",
    "MixedPrecision",
    "OptimStateDictConfig",
    "ShardedOptimStateDictConfig",
    "ShardedStateDictConfig",
    "ShardingStrategy",
    "StateDictConfig",
    "StateDictSettings",
    "StateDictType",
]

class ShardingStrategy(Enum):
    FULL_SHARD = ...
    SHARD_GRAD_OP = ...
    NO_SHARD = ...
    HYBRID_SHARD = ...
    _HYBRID_SHARD_ZERO2 = ...

class BackwardPrefetch(Enum):
    BACKWARD_PRE = ...
    BACKWARD_POST = ...

@dataclass
class MixedPrecision:
    param_dtype: torch.dtype | None = ...
    reduce_dtype: torch.dtype | None = ...
    buffer_dtype: torch.dtype | None = ...
    keep_low_precision_grads: bool = ...
    cast_forward_inputs: bool = ...
    cast_root_forward_inputs: bool = ...
    _module_classes_to_ignore: Sequence[type[torch.nn.Module]] = ...

@dataclass
class CPUOffload:
    offload_params: bool = ...

class StateDictType(Enum):
    FULL_STATE_DICT = ...
    LOCAL_STATE_DICT = ...
    SHARDED_STATE_DICT = ...

@dataclass
class StateDictConfig:
    offload_to_cpu: bool = ...

@dataclass
class FullStateDictConfig(StateDictConfig):
    rank0_only: bool = ...

@dataclass
class LocalStateDictConfig(StateDictConfig): ...

@dataclass
class ShardedStateDictConfig(StateDictConfig):
    _use_dtensor: bool = ...

@dataclass
class OptimStateDictConfig:
    offload_to_cpu: bool = ...

@dataclass
class FullOptimStateDictConfig(OptimStateDictConfig):
    rank0_only: bool = ...

@dataclass
class LocalOptimStateDictConfig(OptimStateDictConfig):
    offload_to_cpu: bool = ...

@dataclass
class ShardedOptimStateDictConfig(OptimStateDictConfig):
    _use_dtensor: bool = ...

@dataclass
class StateDictSettings:
    state_dict_type: StateDictType
    state_dict_config: StateDictConfig
    optim_state_dict_config: OptimStateDictConfig
