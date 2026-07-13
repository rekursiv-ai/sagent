from typing import Any

"""Provides optimal triton kernel parameters.

Aim
---

The usage of optimal triton kernel parameters may increase the
performance of operations several times. For example, for large tensor
shapes, the usage of a bsr tensor as mat1 argument in addmm-based
operations typically outperforms the corresponding operation with
strided-only inputs when the blocked representation of a tensor
provides a better alignment with memory access than what the strided
representation would provide.

Pre-computed kernel parameters
------------------------------

This script finds and stores the optimal triton kernel parameters for
a specific set of shape configurations. For instance, the set of shape
configurations of the bsr_dense_addmm kernel is defined as

  input, out: M x N strided tensor
  mat1: M x K bsr tensor with blocksize (BM, BK) and given sparsity
  mat2: M x N strided tensor
  dtype = float16, bfloat16, float32
  sparsity = 0.5
  M = 256, 512, ..., 16384
  K = M
  N = 256, 512, ..., 131072
  BM = 16, 32, ..., 128
  BK = BM
  alpha = 1
  beta = 0, 1
  GPUs: NVIDIA A100-SXM4-80GB

Approximations
--------------

It is practically infeasible to pre-compute optimal kernel parameter
for all possible shape configurations as well as for all existing
GPUs. Therefore, we'll assume that the pre-computed optimal parameters
are good enough approximations when
1) the used GPU is any of NVIDIA A100 Tensor Core GPUs,
2) the actual sparsity of mat1 is different from sparsity value 0.5.

If a particular shape configuration does not fall in the set of
pre-computed kernel parameters, or it does not match with the listed
approximations above, or the used GPU device is not a NVIDIA A100 GPU,
then a reference set of triton kernel parameters will be used when
executing operations. The reference kernel parameters are defined in
torch/sparse/_triton_ops.py, see bsr_dense_addmm_meta function, for
instance.

Computing optimal kernel parameters
-----------------------------------

If the approximations listed above are unacceptable, e.g. when one
seeks a maximal performance possible, the optimal kernel parameters
for a particular GPU can be computed by simply running this script in
the pytorch development tree::

  cd /path/to/pytorch
  python -m pip install --no-build-isolation -v -e .
  python torch/sparse/_triton_ops_meta.py

This will compute the optimal kernel parameters for the GPU device
available in the host system for all shape configurations listed in
"Pre-computed kernel parameters" above. The results will be stored in
the database of kernel parameters. Currently, this database is defined
as this module (see "BEGIN GENERATED DATA" comment below) that will be
modified when the script is run. Create a pytorch PR with the
corresponding modifications in this file to make the computed optimal
kernel parameters available for other users as pre-computed kernel
parameters.

Moreover, one can compute the optimal kernel parameters for a specific
set of shape configurations and specific sparsity patterns. For that,
use tuning functions provided by this module:

  tune_bsr_dense_addmm(input, mat1, mat2, beta=1, alpha=1, out=None, verbose=False, store=False) -> meta

The tuning functions return a dictionary of optimal kernel parameters
that can be passed to the corresponding operation, e.g.

  bsr_dense_addmm(..., meta=meta)

Or, when store==True, the optimal kernel parameters will be stored in
the database of pre-computed kernel parameters in runtime so that all
addmm-based operations such as torch.addmm, torch.mm,
torch.nn.functional.linear will benefit from using the computed
optimal set of kernel parameters.

Note that running tune_bsr_dense_addmm can take several minutes. So,
use it wisely, e.g. by implementing persistent storage of optimized
kernel parameters. See the source code of get_meta and
tune_bsr_dense_addmm to learn how to register a custom set of optimal
kernel parameters for addmm-based operations.

"""
__all__ = ["get_meta", "tune__int_bsr_dense_addmm", "tune_bsr_dense_addmm"]

def get_meta(
    op, key, device_name=..., version=..., exact=...
) -> (
    dict[Literal[GROUP_SIZE, SPLIT_N, TILE_M, TILE_N, num_stages, num_warps], Any]
    | dict[Literal[GROUP_SIZE_ROW, SPLIT_N, num_stages, num_warps], Any]
    | dict[Any, Any]
    | None
): ...
def update(op, device_name, version, key, value) -> None: ...
def dump() -> None: ...
def minimize(
    target_func,
    initial_parameters,
    reference_parameters,
    step_func,
    max_step=...,
    verbose=...,
    all_values=...,
) -> (
    tuple[dict[Any, Any], Literal[-1], Literal[-1], str]
    | tuple[dict[Any, Any], Any, Any, str]
): ...
def create_blocked_tensor(B, M, N, blocksize, sparsity, dtype, device) -> Tensor: ...
def optimize_scatter_mm(
    m, k, n, bm, bk, dtype=..., device=..., sparsity=..., force=...
) -> None: ...
def tune__int_bsr_dense_addmm(
    input,
    bsr,
    dense,
    *,
    beta=...,
    alpha=...,
    out=...,
    store=...,
    verbose=...,
    force=...,
) -> (
    dict[Literal[GROUP_SIZE, SPLIT_N, TILE_M, TILE_N, num_stages, num_warps], Any]
    | dict[Literal[GROUP_SIZE_ROW, SPLIT_N, num_stages, num_warps], Any]
    | dict[Any, Any]
): ...
def tune_bsr_dense_addmm(
    input,
    bsr,
    dense,
    *,
    beta=...,
    alpha=...,
    left_alpha=...,
    right_alpha=...,
    out=...,
    store=...,
    verbose=...,
    force=...,
    opname=...,
) -> (
    dict[Literal[GROUP_SIZE, SPLIT_N, TILE_M, TILE_N, num_stages, num_warps], Any]
    | dict[Literal[GROUP_SIZE_ROW, SPLIT_N, num_stages, num_warps], Any]
    | dict[Any, Any]
): ...
def optimize_bsr_dense_addmm(
    m,
    k,
    n,
    bm,
    bk,
    beta=...,
    alpha=...,
    use_left_alpha=...,
    use_right_alpha=...,
    dtype=...,
    out_dtype=...,
    device=...,
    sparsity=...,
    force=...,
    verbose=...,
    opname=...,
) -> None: ...
def main(op=..., force=..., dtype=..., verbose=...): ...

_operation_device_version_data: dict[Any, dict] = ...
