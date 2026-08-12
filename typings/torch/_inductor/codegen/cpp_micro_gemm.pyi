from types import ModuleType
from typing import Any, Literal, Self
from collections.abc import Callable
from enum import Enum

import dataclasses

import torch

from .cpp_template_kernel import CppTemplateKernel
from .cpp_utils import GemmBlocking
from .. import ir
from ..cpu_vec_isa import VecAMX, VecAVX2, VecAVX512, VecISA, VecNEON, VecSVE256

class LayoutType(Enum):
    NORMAL = ...
    VNNI2 = ...
    VNNI4 = ...

_IS_WINDOWS = ...

def get_restrict_keyword() -> str: ...

class CppMicroGemm:
    DECLARE_KERNEL = ...
    def __init__(
        self,
        name,
        input_dtype,
        input2_dtype,
        output_dtype,
        compute_dtype,
        register_blocking,
        alpha=...,
    ) -> None: ...
    def get_common_options(
        self,
    ) -> dict[str, ModuleType | Any | str | int | bool | Self]: ...
    def get_kernel_declaration(self) -> Any: ...
    def get_kernel_extra_args_declare(self) -> str: ...
    def get_kernel_extra_args(self, **kwargs) -> list[str]: ...
    def codegen_define(self, kernel: CppTemplateKernel) -> str: ...
    def codegen_call(
        self,
        kernel: CppTemplateKernel,
        A: ir.Buffer,
        B: ir.Buffer,
        C: ir.Buffer,
        accum: bool,
        prefetch: bool = ...,
        **kwargs_for_extra_args,
    ) -> str: ...
    def use_local_vnni_blocking(self, should_block_weight: bool) -> None: ...
    def codegen_init(self, kernel: CppTemplateKernel) -> str: ...
    def codegen_finalize(self, kernel: CppTemplateKernel) -> str: ...
    def get_b_layout(self) -> LayoutType: ...

    ALLOCATE_WEIGHT_BUFFER = ...
    def codegen_allocate_weight_buffer(
        self, buffer_name: str, buffer_dtype: str, *size_args
    ) -> str: ...
    def is_woq_int4(self) -> Literal[False]: ...

@dataclasses.dataclass
class CppMicroGemmConfig:
    input_dtype: torch.dtype
    input2_dtype: torch.dtype
    output_dtype: torch.dtype
    compute_dtype: torch.dtype
    vec_isa_cls: type[VecISA]
    register_blocking: GemmBlocking
    extra_check: Callable[..., bool] | None = ...

micro_gemm_configs: dict[type[CppMicroGemm], list[CppMicroGemmConfig]] = ...

def register_micro_gemm(*configs) -> Callable[..., Any]: ...
def generate_gemm_config(
    vec_isa_cls,
    register_blockings,
    input_dtype=...,
    input2_dtype=...,
    output_dtype=...,
    compute_dtype=...,
    extra_check=...,
) -> list[CppMicroGemmConfig]: ...

class CppMicroGemmRef(CppMicroGemm):
    TEMPLATE_ENTRY = ...
    def __init__(
        self, name, input_dtype, input2_dtype, output_dtype, compute_dtype, alpha
    ) -> None: ...
    def codegen_define(self, kernel: CppTemplateKernel) -> str: ...

def is_int8_woq_gemm_small_m_dim_corner_case(config, m, n, k): ...
def check_int8_woq_small_m_dim(
    config, m, n, k, alpha, num_threads, **kwargs
) -> bool: ...
def do_not_use_with_small_m_for_int8_woq(
    config, m, n, k, alpha, num_threads, **kwargs
) -> bool: ...

@register_micro_gemm(
    *generate_gemm_config(
        VecAVX512, [(8, 48, 1), (8, 32, 1), (16, 16, 1)], input_dtype=torch.float
    ),
    *generate_gemm_config(
        VecAVX512,
        [(8, 48, 1), (8, 32, 1), (16, 16, 1)],
        input_dtype=torch.bfloat16,
        output_dtype=torch.float,
    ),
    *generate_gemm_config(
        VecAVX512,
        [(8, 48, 1), (8, 32, 1), (16, 16, 1)],
        input_dtype=torch.half,
        output_dtype=torch.float,
    ),
    *generate_gemm_config(
        VecAVX512,
        [(8, 48, 1), (8, 32, 1), (16, 16, 1)],
        input_dtype=torch.bfloat16,
        input2_dtype=torch.int8,
        output_dtype=torch.float,
        compute_dtype=torch.float,
        extra_check=do_not_use_with_small_m_for_int8_woq,
    ),
    *generate_gemm_config(
        VecAVX512,
        [(4, 32, 64), (8, 32, 64)],
        input_dtype=torch.bfloat16,
        input2_dtype=torch.int8,
        output_dtype=torch.float,
        compute_dtype=torch.float,
        extra_check=check_int8_woq_small_m_dim,
    ),
    *generate_gemm_config(
        VecAVX2, [(4, 24, 1), (4, 16, 1), (8, 8, 1)], input_dtype=torch.float
    ),
    *generate_gemm_config(
        VecAVX2,
        [(4, 24, 1), (4, 16, 1), (8, 8, 1)],
        input_dtype=torch.bfloat16,
        output_dtype=torch.float,
    ),
    *generate_gemm_config(
        VecAVX2,
        [(4, 24, 1), (4, 16, 1), (8, 8, 1)],
        input_dtype=torch.half,
        output_dtype=torch.float,
    ),
    *generate_gemm_config(
        VecAVX2,
        [(4, 24, 1), (4, 16, 1), (8, 8, 1)],
        input_dtype=torch.bfloat16,
        input2_dtype=torch.int8,
        output_dtype=torch.float,
        compute_dtype=torch.float,
        extra_check=do_not_use_with_small_m_for_int8_woq,
    ),
    *generate_gemm_config(
        VecAVX2,
        [(2, 16, 64), (4, 16, 64)],
        input_dtype=torch.bfloat16,
        input2_dtype=torch.int8,
        output_dtype=torch.float,
        compute_dtype=torch.float,
        extra_check=check_int8_woq_small_m_dim,
    ),
    *generate_gemm_config(
        VecNEON,
        [(4, 24, 1), (4, 16, 1), (8, 8, 1)],
        input_dtype=torch.float,
        input2_dtype=torch.float,
        output_dtype=torch.float,
        compute_dtype=torch.float,
    ),
    *generate_gemm_config(
        VecSVE256,
        [(4, 24, 1), (4, 16, 1), (8, 8, 1)],
        input_dtype=torch.float,
        input2_dtype=torch.float,
        output_dtype=torch.float,
        compute_dtype=torch.float,
    ),
)
class CppMicroGemmFP32Vec(CppMicroGemm):
    TEMPLATE_ENTRY = ...
    TEMPLATE_KERNEL = ...
    def __init__(
        self,
        name,
        input_dtype,
        input2_dtype,
        output_dtype,
        compute_dtype,
        register_blocking,
        alpha=...,
        tail_n=...,
        trans_b=...,
    ) -> None: ...
    def codegen_define(self, kernel: CppTemplateKernel) -> str: ...

def check_amx_extra(config, m, n, k, alpha, num_threads, **kwargs): ...
def check_int8_bf16_amx_extra(config, m, n, k, alpha, num_threads, **kwargs): ...
def check_amx_fp16_extra(config, m, n, k, alpha, num_threads, **kwargs): ...

@register_micro_gemm(
    *generate_gemm_config(
        VecAMX,
        [(32, 32, 64), (48, 16, 64)],
        input_dtype=torch.int8,
        input2_dtype=torch.int8,
        output_dtype=torch.int32,
        compute_dtype=torch.int32,
        extra_check=check_amx_extra,
    ),
    *generate_gemm_config(
        VecAMX,
        [(32, 32, 32), (48, 16, 32)],
        input_dtype=torch.bfloat16,
        input2_dtype=torch.int8,
        output_dtype=torch.float,
        compute_dtype=torch.float,
        extra_check=check_int8_bf16_amx_extra,
    ),
    *generate_gemm_config(
        VecAMX,
        [(32, 16, 32), (32, 32, 32), (48, 16, 32), (16, 48, 32)],
        input_dtype=torch.bfloat16,
        output_dtype=torch.float,
        extra_check=check_amx_extra,
    ),
    *generate_gemm_config(
        VecAMX,
        [(32, 32, 32), (48, 16, 32), (16, 48, 32)],
        input_dtype=torch.float16,
        output_dtype=torch.float,
        extra_check=check_amx_fp16_extra,
    ),
    *generate_gemm_config(
        VecAMX,
        [(32, 32, 64), (48, 16, 64)],
        input_dtype=torch.uint8,
        input2_dtype=torch.int8,
        output_dtype=torch.int32,
        compute_dtype=torch.int32,
        extra_check=check_amx_extra,
    ),
)
class CppMicroGemmAMX(CppMicroGemm):
    TEMPLATE_ENTRY = ...
    TEMPLATE_KERNEL = ...
    def codegen_define(self, kernel: CppTemplateKernel) -> str: ...
    def codegen_init(self, kernel: CppTemplateKernel) -> str: ...
    def codegen_finalize(self, kernel: CppTemplateKernel) -> str: ...
    def get_kernel_extra_args_declare(self) -> str: ...
    def get_kernel_extra_args(self, **kwargs) -> list[str]: ...
    def get_b_layout(self) -> Literal[LayoutType.VNNI4, LayoutType.VNNI2]: ...

def check_brgemm_extra(
    config, m, n, k, alpha, num_threads, **kwargs
) -> Literal[False]: ...

@register_micro_gemm(
    *generate_gemm_config(
        VecAMX,
        [(32, 32, 32), (48, 16, 32), (16, 48, 32)],
        input_dtype=torch.half,
        output_dtype=torch.float,
        extra_check=check_brgemm_extra,
    )
)
class CppMicroBrgemm(CppMicroGemm):
    TEMPLATE_ENTRY = ...
    def codegen_define(self, kernel: CppTemplateKernel) -> str: ...
    def codegen_finalize(self, kernel: CppTemplateKernel) -> str: ...
    def get_b_layout(self) -> Literal[LayoutType.VNNI2]: ...

def check_woq_int4_extra(
    config, m, n, k, alpha, num_threads, **kwargs
) -> Literal[False]: ...

@register_micro_gemm(
    *generate_gemm_config(
        VecAVX512,
        [(4, 64, 32), (4, 64, 64), (4, 64, 128)],
        input_dtype=torch.bfloat16,
        input2_dtype=torch.uint8,
        output_dtype=torch.float,
        compute_dtype=torch.float,
        extra_check=check_woq_int4_extra,
    )
)
class CppMicroGemmWoQInt4Avx512(CppMicroGemmFP32Vec):
    TEMPLATE_ENTRY = ...
    TEMPLATE_KERNEL = ...
    def get_kernel_extra_args_declare(self) -> str: ...
    def get_kernel_extra_args(self, **kwargs) -> list[str]: ...
    def is_woq_int4(self) -> Literal[True]: ...

@register_micro_gemm(
    *generate_gemm_config(
        VecAMX,
        [(16, 32, 32), (32, 32, 32)],
        input_dtype=torch.bfloat16,
        input2_dtype=torch.uint8,
        output_dtype=torch.float,
        compute_dtype=torch.float,
        extra_check=check_amx_extra,
    )
)
class CppMicroGemmWoQInt4Amx(CppMicroGemmAMX):
    TEMPLATE_ENTRY = ...
    def get_kernel_extra_args_declare(self) -> str: ...
    def get_kernel_extra_args(self, **kwargs) -> list[str]: ...
    def is_woq_int4(self) -> Literal[True]: ...

def create_micro_gemm(
    name,
    m,
    n,
    k,
    input_dtype,
    input2_dtype,
    output_dtype=...,
    compute_dtype=...,
    alpha=...,
    num_threads=...,
    use_ref=...,
    q_group_size=...,
) -> CppMicroGemm | None: ...
