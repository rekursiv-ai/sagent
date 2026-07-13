import torch.distributed as dist

"""
Manages process groups for distributed compilation in TorchDynamo.

This module handles the initialization and management of process groups used for
distributed compilation. Key features:

- Lazy initialization of compilation process groups
- Only creates groups when distributed mode is enabled and available
- Integrates with compiler_collectives configuration setting
- Provides a single global process group for compilation coordination

The process group is created only when needed and if the distributed environment
is properly initialized, making it safe to import and use this module even in
non-distributed scenarios.
"""
_COMPILE_PG: dist.ProcessGroup | None = ...
_GUARD_PG: dist.ProcessGroup | None = ...

def get_compile_pg() -> dist.ProcessGroup | None: ...
def get_guard_pg() -> dist.ProcessGroup | None: ...
