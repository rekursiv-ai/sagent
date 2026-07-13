from torch._C._distributed_rpc import _TensorPipeRpcBackendOptionsBase

import torch

from . import _is_tensorpipe_available

type DeviceType = int | str | torch.device
__all__ = ["TensorPipeRpcBackendOptions"]
if _is_tensorpipe_available: ...
else:
    _TensorPipeRpcBackendOptionsBase = ...

class TensorPipeRpcBackendOptions(_TensorPipeRpcBackendOptionsBase):
    def __init__(
        self,
        *,
        num_worker_threads: int = ...,
        rpc_timeout: float = ...,
        init_method: str = ...,
        device_maps: dict[str, dict[DeviceType, DeviceType]] | None = ...,
        devices: list[DeviceType] | None = ...,
        _transports: list | None = ...,
        _channels: list | None = ...,
    ) -> None: ...
    def set_device_map(
        self, to: str, device_map: dict[DeviceType, DeviceType]
    ) -> None: ...
    def set_devices(self, devices: list[DeviceType]) -> None: ...
