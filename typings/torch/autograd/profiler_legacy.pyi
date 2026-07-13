from typing_extensions import deprecated

__all__ = ["profile"]

@deprecated(
    "`torch.autograd.profiler_legacy.profile` is deprecated and will be removed in a future release. Please use `torch.profiler` instead.",
    category=None,
)
class profile:
    def __init__(
        self,
        enabled=...,
        *,
        use_cuda=...,
        record_shapes=...,
        with_flops=...,
        profile_memory=...,
        with_stack=...,
        with_modules=...,
    ) -> None: ...
    def config(self) -> ProfilerConfig: ...
    def __enter__(self) -> Self | None: ...
    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False] | None: ...
    def table(
        self,
        sort_by=...,
        row_limit=...,
        max_src_column_width=...,
        max_name_column_width=...,
        max_shapes_column_width=...,
        header=...,
        top_level_events_only=...,
    ): ...
    def export_chrome_trace(self, path) -> None: ...
    def export_stacks(self, path: str, metric: str = ...) -> None: ...
    def key_averages(
        self, group_by_input_shape=..., group_by_stack_n=...
    ) -> EventList: ...
    def total_average(self) -> FunctionEventAvg: ...
    @property
    def self_cpu_time_total(self) -> int: ...
