from dataclasses import dataclass

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._op_schema import OpStrategy

@dataclass
class EinsumDims:
    contracting_dims: list[str]
    batch_dims: list[str]
    lhs_out_only_dims: list[str]
    rhs_out_only_dims: list[str]
    @classmethod
    def parse_equation(cls, equation: str) -> tuple[list[str], str]: ...
    @classmethod
    def parse_dims(cls, input_dims: list[str], output_dim: str) -> EinsumDims: ...

def gen_einsum_strategies(
    equation: str, mesh: DeviceMesh, *, linearity: bool = ...
) -> OpStrategy: ...
