from torch.utils import (
    backcompat as backcompat,
    benchmark as benchmark,
    collect_env as collect_env,
    data as data,
    deterministic as deterministic,
    hooks as hooks,
)
from torch.utils.backend_registration import (
    generate_methods_for_privateuse1_backend as generate_methods_for_privateuse1_backend,
    rename_privateuse1_backend as rename_privateuse1_backend,
)
from torch.utils.cpp_backtrace import get_cpp_backtrace as get_cpp_backtrace
from torch.utils.throughput_benchmark import ThroughputBenchmark as ThroughputBenchmark

def set_module(obj, mod) -> None: ...

cmake_prefix_path = ...

def swap_tensors(t1, t2) -> None: ...
