from dataclasses import dataclass

import torch

log = ...

@dataclass(frozen=True)
class DeviceInfo:
    tops: dict[torch.dtype | str, float]
    dram_bw_gbs: float
    dram_gb: float

_device_mapping: dict[str, DeviceInfo] = ...

def lookup_device_info(name: str) -> DeviceInfo | None: ...
def datasheet_tops(dtype: torch.dtype, is_tf32: bool = ...) -> float | None: ...
