import torch

type _device_t = torch.device | str | int | None

def _get_device_index(device: _device_t, optional: bool = ...) -> int: ...
