from torch.onnx._internal.exporter import _registration

import torch
import torch.export
import torch.fx

def decompose_with_registry(
    exported_program: torch.export.ExportedProgram, registry: _registration.ONNXRegistry
) -> torch.export.ExportedProgram: ...
def insert_type_promotion_nodes(graph_module: torch.fx.GraphModule) -> None: ...
def remove_assertion_nodes(
    graph_module: torch.fx.GraphModule,
) -> torch.fx.GraphModule: ...
