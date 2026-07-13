from collections.abc import Callable

from torch.onnx._internal.exporter import _registration

def get_onnx_implemented_overloads(
    registry: _registration.ONNXRegistry,
) -> list[_registration.TorchOp]: ...
def create_onnx_friendly_decomposition_table(
    onnx_registered_ops: set[_registration.TorchOp],
) -> dict[_registration.TorchOp, Callable]: ...
