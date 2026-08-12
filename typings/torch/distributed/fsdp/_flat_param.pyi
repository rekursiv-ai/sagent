from collections.abc import Generator, Iterator, Sequence
from enum import Enum
from typing import Any, NamedTuple, no_type_check

import contextlib

from torch import Tensor, nn
from torch.nn.parameter import _ParameterMeta

import torch
import torch.distributed as dist

from ._fsdp_extensions import FSDPExtensions

__all__ = [
    "FlatParamHandle",
    "FlatParamShardMetadata",
    "FlatParameter",
    "HandleShardingStrategy",
    "ParamInfo",
    "SharedParamInfo",
]
logger = ...
_FSDP_USE_UNSAFE_SETATTR = ...
_FSDP_SKIP_WRITEBACK_CHECK = ...
_FSDP_USE_FULL_PREC_IN_EVAL = ...
_FLAT_PARAM_PADDING_VALUE = ...
_FSDP_USE_FAKE_ALL_GATHER = ...
_FSDP_USE_FAKE_REDUCE = ...

class HandleShardingStrategy(Enum):
    FULL_SHARD = ...
    SHARD_GRAD_OP = ...
    NO_SHARD = ...
    HYBRID_SHARD = ...
    _HYBRID_SHARD_ZERO2 = ...

RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES = ...
NO_RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES = ...

class ParamInfo(NamedTuple):
    param_name: str
    module: nn.Module
    module_name: str

class SharedParamInfo(NamedTuple):
    param_name: str
    module: nn.Module
    module_name: str
    prim_param_name: str
    prim_module: nn.Module
    prim_module_name: str

class _ShardParamInfo(NamedTuple):
    in_shard: bool
    offset_in_shard: int | None
    numel_in_shard: int | None
    intra_param_start_idx: int | None
    intra_param_end_idx: int | None

class FlatParamShardMetadata(NamedTuple):
    param_names: tuple[str, ...]
    param_shapes: tuple[torch.Size, ...]
    param_strides: tuple[tuple[int, ...], ...]
    param_contiguities: tuple[bool, ...]
    param_numels: tuple[int, ...]
    param_offsets: tuple[tuple[int, int], ...]

class _FlatParameterMeta(_ParameterMeta):
    def __instancecheck__(self, instance) -> bool: ...

class FlatParameter(nn.Parameter, metaclass=_FlatParameterMeta):
    _unpadded_unsharded_size: torch.Size
    _padded_unsharded_size: torch.Size
    _sharded_size: torch.Size
    _num_params: int
    _param_infos: tuple[ParamInfo, ...]
    _shapes: tuple[torch.Size, ...]
    _strides: tuple[tuple[int, ...], ...]
    _contiguities: tuple[bool, ...]
    _fqns: tuple[str, ...]
    _param_extensions: tuple[Any | None, ...]
    _numels_with_padding: tuple[int, ...]
    _numels: tuple[int, ...]
    _shard_param_infos: tuple[_ShardParamInfo, ...]
    _shared_param_infos: tuple[SharedParamInfo, ...]
    _modules: set[nn.Module]
    _shard_numel_padded: int
    _local_shard: Tensor
    _full_param_padded: Tensor
    _full_prec_full_param_padded: Tensor
    _post_backward_hook_state: tuple[Any, Any]
    _post_backward_hook_handle: Any
    _mp_shard: Tensor
    _cpu_grad: Tensor
    _saved_grad_shard: Tensor
    _params: list[nn.Parameter] | None
    _shared_params: list[nn.Parameter] | None
    _tensors: list[Tensor | None] | None
    _is_grad_none_mask: list[bool] | None
    _is_padding_mask: list[bool]
    def __new__(cls, data=..., requires_grad=...) -> Parameter: ...

class FlatParamHandle:
    def __init__(
        self,
        params: Sequence[nn.Parameter | Tensor],
        fully_sharded_module: nn.Module,
        device: torch.device,
        sharding_strategy: HandleShardingStrategy,
        offload_params: bool,
        mp_param_dtype: torch.dtype | None,
        mp_reduce_dtype: torch.dtype | None,
        keep_low_precision_grads: bool,
        process_group: dist.ProcessGroup,
        use_orig_params: bool,
        *,
        fsdp_extension: FSDPExtensions | None = ...,
    ) -> None: ...
    def flatten_tensors(self, tensors: list[Tensor], aligned_numel: int) -> Tensor: ...
    def flatten_tensors_into_flat_param(
        self, tensors: list[Tensor], aligned_numel: int, requires_grad: bool
    ) -> FlatParameter: ...
    @torch.no_grad()
    def shard(self) -> None: ...
    @no_type_check
    def shard_metadata(self) -> FlatParamShardMetadata: ...
    @no_type_check
    @torch.no_grad()
    def init_flat_param_attributes(self) -> None: ...
    def pre_unshard(self) -> bool: ...
    def unshard(self) -> None: ...
    def needs_unshard(self) -> bool: ...
    def post_unshard(self) -> None: ...
    @torch.no_grad()
    def unshard_grad(self) -> None: ...
    def reshard_grad(self) -> None: ...
    def prepare_gradient_for_backward(self) -> None: ...
    def prepare_gradient_for_optim(self) -> None: ...
    @contextlib.contextmanager
    def to_cpu(self) -> Generator[None, Any]: ...
    def reshard(self, free_unsharded_flat_param: bool) -> None: ...
    def post_reshard(self) -> None: ...
    @contextlib.contextmanager
    def unflatten_as_params(self) -> Generator: ...
    def flat_param_to(self, *args, **kwargs) -> None: ...
    def is_sharded(self, tensor: Tensor) -> bool: ...
    def param_module_names(self) -> Iterator[tuple[str, str]]: ...
    def shared_param_module_names(self) -> Iterator[tuple[str, str]]: ...
    @property
    def sharded_grad(self) -> Tensor | None: ...
    @property
    def uses_sharded_strategy(self) -> bool: ...
