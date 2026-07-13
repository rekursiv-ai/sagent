import functools

@functools.cache
def gen_cutlass_presets() -> dict[int, dict[str, list[str]]]: ...
