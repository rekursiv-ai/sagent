import torch

__all__ = ["generate_methods_for_privateuse1_backend", "rename_privateuse1_backend"]
_privateuse1_backend_name = ...

def rename_privateuse1_backend(backend_name: str) -> None: ...
def generate_methods_for_privateuse1_backend(
    for_tensor: bool = ...,
    for_module: bool = ...,
    for_packed_sequence: bool = ...,
    for_storage: bool = ...,
    unsupported_dtype: list[torch.dtype] | None = ...,
) -> None: ...
