from collections import defaultdict
from typing_extensions import deprecated

__all__ = [
    "EventList",
    "FormattedTimesMixin",
    "FunctionEvent",
    "FunctionEventAvg",
    "Interval",
    "Kernel",
    "MemRecordsAcc",
    "StringTable",
]

class EventList(list[FunctionEvent | FunctionEventAvg]):
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def self_cpu_time_total(self) -> int: ...
    def table(
        self,
        sort_by=...,
        row_limit=...,
        max_src_column_width=...,
        max_name_column_width=...,
        max_shapes_column_width=...,
        header=...,
        top_level_events_only=...,
        time_unit=...,
    ): ...
    def export_chrome_trace(self, path) -> None: ...
    def supported_export_stacks_metrics(self) -> list[str]: ...
    def export_stacks(self, path: str, metric: str) -> None: ...
    def key_averages(
        self,
        group_by_input_shapes=...,
        group_by_stack_n=...,
        group_by_overload_name=...,
    ) -> EventList: ...
    def total_average(self) -> FunctionEventAvg: ...

class FormattedTimesMixin:
    cpu_time_str = ...
    device_time_str = ...
    cpu_time_total_str = ...
    device_time_total_str = ...
    self_cpu_time_total_str = ...
    self_device_time_total_str = ...
    @property
    def cpu_time(self) -> float: ...
    @property
    def device_time(self) -> float: ...
    @property
    @deprecated(
        "`cuda_time` is deprecated, please use `device_time` instead.",
        category=FutureWarning,
    )
    def cuda_time(self) -> float: ...

class Interval:
    def __init__(self, start, end) -> None: ...
    def elapsed_us(self): ...

Kernel = ...

class FunctionEvent(FormattedTimesMixin):
    def __init__(
        self,
        id,
        name,
        thread,
        start_us,
        end_us,
        overload_name=...,
        fwd_thread=...,
        input_shapes=...,
        stack=...,
        scope=...,
        use_device=...,
        cpu_memory_usage=...,
        device_memory_usage=...,
        is_async=...,
        is_remote=...,
        sequence_nr=...,
        node_id=...,
        device_type=...,
        device_index=...,
        device_resource_id=...,
        is_legacy=...,
        flops=...,
        trace_name=...,
        concrete_inputs=...,
        kwinputs=...,
        is_user_annotation=...,
    ) -> None: ...
    def append_kernel(self, name, device, duration) -> None: ...
    def append_cpu_child(self, child) -> None: ...
    def set_cpu_parent(self, parent) -> None: ...
    @property
    def self_cpu_memory_usage(self) -> int: ...
    @property
    def self_device_memory_usage(self) -> int: ...
    @property
    @deprecated(
        "`self_cuda_memory_usage` is deprecated. Use `self_device_memory_usage` instead.",
        category=FutureWarning,
    )
    def self_cuda_memory_usage(self) -> int: ...
    @property
    def cpu_time_total(self) -> Literal[0]: ...
    @property
    def self_cpu_time_total(self) -> int: ...
    @property
    def device_time_total(self) -> int: ...
    @property
    @deprecated(
        "`cuda_time_total` is deprecated. Use `device_time_total` instead.",
        category=FutureWarning,
    )
    def cuda_time_total(self) -> int: ...
    @property
    def self_device_time_total(self) -> int: ...
    @property
    @deprecated(
        "`self_cuda_time_total` is deprecated. Use `self_device_time_total` instead.",
        category=FutureWarning,
    )
    def self_cuda_time_total(self) -> int: ...
    @property
    def key(self) -> str: ...

class FunctionEventAvg(FormattedTimesMixin):
    count: int
    def __init__(self) -> None: ...
    def add(self, other) -> Self: ...
    def __iadd__(self, other) -> Self: ...
    @property
    def key(self) -> str: ...
    @property
    def self_device_time_total(self) -> float: ...

class StringTable(defaultdict):
    def __missing__(self, key): ...

class MemRecordsAcc:
    def __init__(self, mem_records) -> None: ...
    def in_interval(self, start_us, end_us) -> Generator[Any, Any, None]: ...

MEMORY_EVENT_NAME = ...
OUT_OF_MEMORY_EVENT_NAME = ...
