from kernels import Device, LayerRepository

_kernels_available = ...
_KERNEL_MAPPING: dict[str, dict[Device | str, LayerRepository]] = ...

def is_kernel(attn_implementation: str | None) -> bool: ...
def load_and_register_kernel(attn_implementation: str) -> None: ...

__all__ = [
    "LayerRepository",
    "register_kernel_mapping",
    "replace_kernel_forward_from_hub",
    "use_kernel_forward_from_hub",
]
