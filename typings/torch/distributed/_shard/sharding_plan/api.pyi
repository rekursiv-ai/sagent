from dataclasses import dataclass

import abc

from torch import nn
from torch.distributed._shard.sharder import Sharder
from torch.distributed._shard.sharding_spec import ShardingSpec

@dataclass
class ShardingPlan:
    plan: dict[str, ShardingSpec | Sharder]
    output_plan: dict[str, ShardingSpec] | None = ...
    return_local_tensor: list[str] | None = ...

class ShardingPlanner(abc.ABC):
    @abc.abstractmethod
    def build_plan(self, module: nn.Module) -> ShardingPlan: ...
