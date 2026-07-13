import onnx_ir

"""Utility to lazily import modules."""

class _LazyModule:
    def __init__(self, module_name: str) -> None: ...
    def __getattr__(self, attr: str) -> object: ...

onnxscript_ir = onnx_ir
