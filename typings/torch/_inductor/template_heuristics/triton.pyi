from collections.abc import Callable, Generator
from functools import partial
from threading import Lock
from typing import Any

import dataclasses

from torch._inductor.template_heuristics.triton_addmm import AddMMConfigMixin
from triton import Config as TritonConfig

import sympy
import torch

from .gemm import GemmMaxAutotuneTemplateConfigHeuristics
from .registry import register_template_heuristic
from ..ir import Layout
from ..kernel.bmm import bmm_template
from ..kernel.mm import (
    mm_template,
    persistent_tma_mm_template,
    scaled_mm_device_tma_template,
)
from ..kernel.mm_plus_mm import mm_plus_mm_template
from ..kernel_inputs import KernelInputs

@dataclasses.dataclass
class BaseConfig:
    block_m: int
    block_n: int
    block_k: int
    num_stages: int
    num_warps: int
    hint_override: int | None = ...

@dataclasses.dataclass
class GemmConfig(BaseConfig):
    group_m: int = ...

ConvConfig = BaseConfig

@dataclasses.dataclass
class FlexConfig:
    block_m: int
    block_n: int
    num_stages: int
    num_warps: int

@dataclasses.dataclass
class FlexDecodeConfig:
    block_n: int
    num_stages: int
    num_warps: int

@dataclasses.dataclass
class ROCmGemmConfig(GemmConfig):
    matrix_instr_nonkdim: int = ...
    waves_per_eu: int = ...
    kpack: int = ...

@dataclasses.dataclass
class ROCmConvConfig(ConvConfig):
    matrix_instr_nonkdim: int = ...
    waves_per_eu: int = ...
    kpack: int = ...

@dataclasses.dataclass
class ROCmFlexConfig(FlexConfig):
    matrix_instr_nonkdim: int = ...
    waves_per_eu: int = ...
    kpack: int = ...

@dataclasses.dataclass
class ROCmFlexDecodeConfig(FlexDecodeConfig):
    matrix_instr_nonkdim: int = ...
    waves_per_eu: int = ...
    kpack: int = ...

class BaseHeuristicSingleton(type):
    _instances: dict[type[Any], Any] = ...
    _lock: Lock = ...
    def __call__(
        cls: BaseHeuristicSingleton, *args: Any, **kwargs: Any
    ) -> BaseConfigHeuristic: ...

class BaseConfigHeuristic(metaclass=BaseHeuristicSingleton):
    def __init__(self) -> None: ...
    def preprocess_mm_configs(
        self,
        m: int,
        n: int,
        k: int,
        configs: list[BaseConfig],
        has_int8_tensor: bool = ...,
        scale: float = ...,
        exclude: Callable[[sympy.Integer, sympy.Integer, sympy.Integer], bool] = ...,
        dtype_size: int = ...,
        op_name: str = ...,
    ) -> Generator[TritonConfig]: ...
    def triton_config(
        self, num_stages: int, num_warps: int, **kwargs: Any
    ) -> TritonConfig: ...
    def get_mm_configs(self) -> partial[Generator[TritonConfig]]: ...
    def get_exhaustive_mm_configs(self) -> partial[Generator[TritonConfig]]: ...
    def get_conv_configs(self) -> partial[Generator[TritonConfig]]: ...
    def get_flex_attn_fwd_configs(
        self, head_dim: int, dtype: Any
    ) -> list[FlexConfig]: ...
    def get_flex_attn_bwd_configs(
        self, head_dim: int, dtype: Any
    ) -> list[FlexConfig]: ...
    def get_flex_decode_configs(
        self, head_dim: int, dtype: Any
    ) -> list[FlexDecodeConfig]: ...

class CPUConfigHeuristic(BaseConfigHeuristic):
    def preprocess_mm_configs(
        self,
        m: int,
        n: int,
        k: int,
        configs: list[BaseConfig],
        has_int8_tensor: bool = ...,
        scale: float = ...,
        exclude: Callable[[sympy.Integer, sympy.Integer, sympy.Integer], bool] = ...,
        dtype_size: int = ...,
        op_name: str = ...,
    ) -> Generator[TritonConfig]: ...

class CUDAConfigHeuristic(BaseConfigHeuristic):
    def __init__(self) -> None: ...
    def get_flex_attn_fwd_configs(
        self, head_dim: int, dtype: Any
    ) -> list[FlexConfig]: ...
    def get_flex_attn_bwd_configs(
        self, head_dim: int, dtype: Any
    ) -> list[FlexConfig]: ...
    def get_flex_decode_configs(
        self, head_dim: int, dtype: Any
    ) -> list[FlexDecodeConfig]: ...

class ROCmConfigHeuristic(BaseConfigHeuristic):
    def __init__(self) -> None: ...
    def get_flex_attn_fwd_configs(
        self, head_dim: int, dtype: Any
    ) -> list[FlexConfig]: ...
    def get_flex_attn_bwd_configs(
        self, head_dim: int, dtype: Any
    ) -> list[FlexConfig]: ...
    def get_flex_decode_configs(
        self, head_dim: int, dtype: Any
    ) -> list[FlexDecodeConfig]: ...

class XPUConfigHeuristic(BaseConfigHeuristic):
    def __init__(self) -> None: ...
    def get_flex_attn_fwd_configs(
        self, head_dim: int, dtype: Any
    ) -> list[FlexConfig]: ...
    def get_flex_attn_bwd_configs(
        self, head_dim: int, dtype: Any
    ) -> list[FlexConfig]: ...
    def get_flex_decode_configs(
        self, head_dim: int, dtype: Any
    ) -> list[FlexDecodeConfig]: ...

class MTIAConfigHeuristic(BaseConfigHeuristic): ...

class MMTemplateConfigMixin(GemmMaxAutotuneTemplateConfigHeuristics):
    get_mm_configs: Callable[[], partial[Generator[TritonConfig]]]
    get_exhaustive_mm_configs: Callable[[], partial[Generator[TritonConfig]]]
    _filter_configs: Callable[[list[BaseConfig]], list[BaseConfig]]

class INT8MMTemplateConfigMixin(MMTemplateConfigMixin):
    def __init__(self) -> None: ...

class MMPlusMMTemplateConfigMixin(MMTemplateConfigMixin):
    def __init__(self) -> None: ...

class TMAWorkspaceMixin(MMTemplateConfigMixin):
    def get_extra_kwargs(
        self, kernel_inputs: KernelInputs, layout: Layout, op_name: str
    ) -> dict[str, Any]: ...

class TMATemplateConfigMixin(TMAWorkspaceMixin, MMTemplateConfigMixin): ...

class BaseScaledMMConfigMixin(MMTemplateConfigMixin):
    def adjust_kernel_inputs(
        self, kernel_inputs: KernelInputs, op_name: str
    ) -> KernelInputs: ...

class ScaledMMConfigMixin(BaseScaledMMConfigMixin):
    def get_extra_kwargs(
        self, kernel_inputs: KernelInputs, layout: Layout, op_name: str
    ) -> dict[str, Any]: ...

class ScaledTMAConfigMixin(TMAWorkspaceMixin, BaseScaledMMConfigMixin): ...

@register_template_heuristic(
    mm_template.uid, "cuda", register=torch.version.hip is None
)
@register_template_heuristic(
    bmm_template.uid, "cuda", register=torch.version.hip is None
)
class CUDAMMTemplateConfigHeuristic(MMTemplateConfigMixin, CUDAConfigHeuristic): ...

@register_template_heuristic(
    mm_template.uid, "cuda", register=torch.version.hip is None, op_name="addmm"
)
@register_template_heuristic(
    bmm_template.uid, "cuda", register=torch.version.hip is None, op_name="baddbmm"
)
class CUDAAddMMTemplateConfigHeuristic(
    AddMMConfigMixin, CUDAMMTemplateConfigHeuristic
): ...

@register_template_heuristic(
    mm_template.uid, "cuda", register=torch.version.hip is None, op_name="mm-ah"
)
class CUDAMMAHTemplateConfigHeuristic(MMTemplateConfigMixin, CUDAConfigHeuristic):
    def __init__(self) -> None: ...

@register_template_heuristic(
    persistent_tma_mm_template.uid, "cuda", register=torch.version.hip is None
)
class CUDAPersistentTMATemplateConfigHeuristic(
    TMATemplateConfigMixin, CUDAConfigHeuristic
):
    def __init__(self) -> None: ...

@register_template_heuristic(
    persistent_tma_mm_template.uid,
    "cuda",
    register=torch.version.hip is None,
    op_name="addmm",
)
class CUDAAddmmPersistentTMATemplateConfigHeuristic(
    AddMMConfigMixin, CUDAPersistentTMATemplateConfigHeuristic
): ...

@register_template_heuristic(
    mm_template.uid, "cuda", register=torch.version.hip is None, op_name="scaled_mm"
)
class CUDAScaledMMTemplateConfigHeuristic(ScaledMMConfigMixin, CUDAConfigHeuristic):
    def __init__(self) -> None: ...

@register_template_heuristic(
    scaled_mm_device_tma_template.uid, "cuda", register=torch.version.hip is None
)
class CUDAScaledTMATemplateConfigHeuristic(ScaledTMAConfigMixin, CUDAConfigHeuristic):
    def __init__(self) -> None: ...

@register_template_heuristic(
    mm_plus_mm_template.uid, "cuda", register=torch.version.hip is None
)
class CUDAMMPlusMMTemplateConfigHeuristic(
    MMPlusMMTemplateConfigMixin, CUDAConfigHeuristic
):
    def __init__(self) -> None: ...

@register_template_heuristic(
    mm_template.uid, "cuda", register=torch.version.hip is None, op_name="int_mm"
)
class CUDAInt8MMTemplateConfigHeuristic(INT8MMTemplateConfigMixin, CUDAConfigHeuristic):
    def __init__(self) -> None: ...

@register_template_heuristic(
    mm_template.uid, "cuda", register=torch.version.hip is not None
)
@register_template_heuristic(
    bmm_template.uid, "cuda", register=torch.version.hip is not None
)
class ROCmMMTemplateConfigHeuristic(MMTemplateConfigMixin, ROCmConfigHeuristic): ...

@register_template_heuristic(
    mm_template.uid, "cuda", register=torch.version.hip is not None, op_name="addmm"
)
@register_template_heuristic(
    bmm_template.uid, "cuda", register=torch.version.hip is not None, op_name="baddbmm"
)
class ROCmAddMMTemplateConfigHeuristic(
    AddMMConfigMixin, ROCmMMTemplateConfigHeuristic
): ...

@register_template_heuristic("mm-ah", "cuda", register=torch.version.hip is not None)
class ROCmMMAHTemplateConfigHeuristic(MMTemplateConfigMixin, ROCmConfigHeuristic):
    def __init__(self) -> None: ...

@register_template_heuristic(
    mm_template.uid, "cuda", register=torch.version.hip is not None, op_name="scaled_mm"
)
class ROCmScaledMMTemplateConfigHeuristic(ScaledMMConfigMixin, ROCmConfigHeuristic):
    def __init__(self) -> None: ...

@register_template_heuristic(
    mm_template.uid, "cuda", register=torch.version.hip is not None, op_name="int_mm"
)
class ROCmInt8MMTemplateConfigHeuristic(INT8MMTemplateConfigMixin, ROCmConfigHeuristic):
    def __init__(self) -> None: ...

@register_template_heuristic(
    mm_plus_mm_template.uid, "cuda", register=torch.version.hip is not None
)
class ROCmMMPlusMMTemplateConfigHeuristic(
    MMPlusMMTemplateConfigMixin, ROCmConfigHeuristic
):
    def __init__(self) -> None: ...

@register_template_heuristic(mm_template.uid, "cpu")
@register_template_heuristic(bmm_template.uid, "cpu")
class CPUMMTemplateConfigHeuristic(MMTemplateConfigMixin, CPUConfigHeuristic): ...

@register_template_heuristic(mm_template.uid, "cpu", op_name="addmm")
@register_template_heuristic(bmm_template.uid, "cpu", op_name="baddbmm")
class CPUAddmmTemplateConfigHeuristic(
    AddMMConfigMixin, CPUMMTemplateConfigHeuristic
): ...

@register_template_heuristic(mm_template.uid, "cpu", op_name="scaled_mm")
class CPUScaledMMTemplateConfigHeuristic(ScaledMMConfigMixin, CPUConfigHeuristic):
    def __init__(self) -> None: ...

@register_template_heuristic(mm_template.uid, "cpu", op_name="int_mm")
class CPUInt8MMTemplateConfigHeuristic(INT8MMTemplateConfigMixin, CPUConfigHeuristic):
    def __init__(self) -> None: ...

@register_template_heuristic(mm_plus_mm_template.uid, "cpu")
class CPUMMPlusMMTemplateConfigHeuristic(
    MMPlusMMTemplateConfigMixin, CPUConfigHeuristic
):
    def __init__(self) -> None: ...

@register_template_heuristic(mm_template.uid, "xpu")
@register_template_heuristic(bmm_template.uid, "xpu")
class XPUMMTemplateConfigHeuristic(MMTemplateConfigMixin, XPUConfigHeuristic): ...

@register_template_heuristic(mm_template.uid, "xpu", op_name="addmm")
@register_template_heuristic(bmm_template.uid, "xpu", op_name="baddbmm")
class XPUAddmmTemplateConfigHeuristic(
    AddMMConfigMixin, XPUMMTemplateConfigHeuristic
): ...

@register_template_heuristic(persistent_tma_mm_template.uid, "xpu")
class XPUPersistentTMATemplateConfigHeuristic(
    TMATemplateConfigMixin, XPUConfigHeuristic
):
    def __init__(self) -> None: ...

@register_template_heuristic(persistent_tma_mm_template.uid, "xpu", op_name="addmm")
class XPUAddmmPersistentTMATemplateConfigHeuristic(
    AddMMConfigMixin, XPUPersistentTMATemplateConfigHeuristic
): ...

@register_template_heuristic(mm_template.uid, "xpu", op_name="scaled_mm")
class XPUScaledMMTemplateConfigHeuristic(ScaledMMConfigMixin, XPUConfigHeuristic):
    def __init__(self) -> None: ...

@register_template_heuristic(mm_template.uid, "xpu", op_name="int_mm")
class XPUInt8MMTemplateConfigHeuristic(INT8MMTemplateConfigMixin, XPUConfigHeuristic):
    def __init__(self) -> None: ...

@register_template_heuristic(mm_plus_mm_template.uid, "xpu")
class XPUMMPlusMMTemplateConfigHeuristic(
    MMPlusMMTemplateConfigMixin, XPUConfigHeuristic
):
    def __init__(self) -> None: ...

@register_template_heuristic(mm_template.uid, "mtia")
@register_template_heuristic(bmm_template.uid, "mtia")
class MTIAMMTemplateConfigHeuristic(MMTemplateConfigMixin, MTIAConfigHeuristic): ...

@register_template_heuristic(mm_template.uid, "mtia", op_name="addmm")
@register_template_heuristic(bmm_template.uid, "mtia", op_name="baddbmm")
class MTIAAddMMTemplateConfigHeuristic(
    AddMMConfigMixin, MTIAMMTemplateConfigHeuristic
): ...

@register_template_heuristic(mm_template.uid, "mtia", op_name="scaled_mm")
class MTIAScaledMMTemplateConfigHeuristic(ScaledMMConfigMixin, MTIAConfigHeuristic):
    def __init__(self) -> None: ...

@register_template_heuristic(mm_template.uid, "mtia", op_name="int_mm")
class MTIAInt8MMTemplateConfigHeuristic(INT8MMTemplateConfigMixin, MTIAConfigHeuristic):
    def __init__(self) -> None: ...

@register_template_heuristic(mm_plus_mm_template.uid, "mtia")
class MTIAMMPlusMMTemplateConfigHeuristic(
    MMPlusMMTemplateConfigMixin, MTIAConfigHeuristic
):
    def __init__(self) -> None: ...
