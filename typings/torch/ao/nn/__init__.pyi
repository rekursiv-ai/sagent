from types import ModuleType as ModuleType

from torch.ao.nn import (
    intrinsic as intrinsic,
    qat as qat,
    quantizable as quantizable,
    quantized as quantized,
    sparse as sparse,
)

__all__ = ["intrinsic", "qat", "quantizable", "quantized", "sparse"]

def __getattr__(name: str) -> ModuleType: ...
