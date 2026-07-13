from dataclasses import dataclass
from typing import Any

from torch import fx
from torch._dynamo.backends.registry import CompiledFn, CompilerFn
from torch.fx.node import Node

import torch

"""
This module implements distributed training optimizations for TorchDynamo backends.

It provides functionality to optimize models wrapped in DistributedDataParallel (DDP)
by intelligently splitting compiled graphs to align with DDP's gradient synchronization
boundaries. Key features include:

- Graph partitioning based on parameter bucket sizes
- Optimization of allreduce operations for distributed training
- Support for parameter ignoring and buffer handling
- Submodule compilation and management
- Debugging utilities for distributed training

The main component is the DDPOptimizer class, which handles graph splitting and
recompilation to enable efficient distributed training while maintaining the benefits
of compilation.
"""
log = ...
ddp_graph_log = ...

def args_str(args: Any) -> str: ...

@dataclass
class Bucket:
    size: int = ...
    params: list[str] = ...
    nodes: list[fx.Node] = ...
    param_ids: list[int] = ...
    opcount_increased_to_capture_external_output: int = ...
    paramsize_before_opcount_increase: int = ...

def bucket_has_external_output(bucket: Bucket) -> bool: ...
def pretty_print_buckets(buckets: list[Bucket], bucket_bytes_cap: int) -> None: ...
def has_higher_order_op(gm: fx.GraphModule) -> bool: ...
def propagate_metadata(orig_gm: fx.GraphModule, split_gm: fx.GraphModule) -> None: ...
def propagate_dynamo_source(
    orig_gm: fx.GraphModule, split_gm: fx.GraphModule
) -> None: ...

class DDPOptimizerContext:
    def __init__(self) -> None: ...

class SubmodCompiler(torch.fx.interpreter.Interpreter):
    def __init__(
        self,
        module: fx.GraphModule,
        compiler: CompilerFn,
        fake_mode: torch._subclasses.fake_tensor.FakeTensorMode,
    ) -> None: ...
    def compile_submod(
        self, input_mod: fx.GraphModule, args: list[torch.Tensor], kwargs: Any
    ) -> Any: ...
    def run_node(self, n: Node) -> Any: ...

class DDPOptimizer:
    def __init__(
        self,
        bucket_bytes_cap: int,
        backend_compile_fn: CompilerFn,
        first_bucket_cap: int | None = ...,
    ) -> None: ...
    def add_param(
        self, bucket: Bucket, param: torch.nn.Parameter, name: str
    ) -> None: ...
    def add_module_params_to_bucket(
        self,
        mod: torch.nn.Module,
        bucket: Bucket,
        processed_modules: set[torch.nn.Module],
        prefix: str,
    ) -> None: ...
    def add_param_args(self, bucket: Bucket, node: fx.Node) -> None: ...
    def compile_fn(
        self, gm: fx.GraphModule, example_inputs: list[torch.Tensor]
    ) -> CompiledFn: ...
