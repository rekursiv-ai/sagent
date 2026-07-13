from torch import _C

"""ONNX exporter exceptions."""
__all__ = ["OnnxExporterWarning", "SymbolicValueError", "UnsupportedOperatorError"]

class OnnxExporterWarning(UserWarning): ...
class OnnxExporterError(RuntimeError): ...

class UnsupportedOperatorError(OnnxExporterError):
    def __init__(
        self, name: str, version: int, supported_version: int | None
    ) -> None: ...

class SymbolicValueError(OnnxExporterError):
    def __init__(self, msg: str, value: _C.Value) -> None: ...
