from typing import (
    Any as Any,
    Unpack as Unpack,
)
from typing_extensions import (
    TypeVarTuple as TypeVarTuple,
)

import os

from torch._C._autograd import (
    DeviceType as DeviceType,
    _supported_activities as _supported_activities,
    kineto_available as kineto_available,
)
from torch._C._profiler import (
    ProfilerActivity as ProfilerActivity,
    RecordScope as RecordScope,
    _ExperimentalConfig as _ExperimentalConfig,
)
from torch._environment import is_fbcode as is_fbcode
from torch.autograd.profiler import (
    KinetoStepTracker as KinetoStepTracker,
    record_function as record_function,
)
from torch.optim.optimizer import (
    Optimizer as Optimizer,
    register_optimizer_step_post_hook as register_optimizer_step_post_hook,
)

from . import itt as itt
from .profiler import (
    ExecutionTraceObserver as ExecutionTraceObserver,
    ProfilerAction as ProfilerAction,
    _KinetoProfile as _KinetoProfile,
    profile as profile,
    schedule as schedule,
    supported_activities as supported_activities,
    tensorboard_trace_handler as tensorboard_trace_handler,
)

r"""
PyTorch Profiler is a tool that allows the collection of performance metrics during training and inference.
Profiler's context manager API can be used to better understand what model operators are the most expensive,
examine their input shapes and stack traces, study device kernel activity and visualize the execution trace.

.. note::
    An earlier version of the API in :mod:`torch.autograd` module is considered legacy and will be deprecated.

"""
__all__ = [
    "DeviceType",
    "ExecutionTraceObserver",
    "ProfilerAction",
    "ProfilerActivity",
    "kineto_available",
    "profile",
    "record_function",
    "schedule",
    "supported_activities",
    "tensorboard_trace_handler",
]
_Ts = TypeVarTuple("_Ts")
if os.environ.get("KINETO_USE_DAEMON", "") or (
    is_fbcode() and os.environ.get("KINETO_FORCE_OPTIMIZER_HOOK", "")
):
    _ = ...
