from torch.export.exported_program import ModuleCallSignature
from torch.export.graph_signature import ExportGraphSignature
from torch.fx.passes.infra.pass_base import PassBase, PassResult

import torch

__all__ = ["CollectTracepointsPass"]

class CollectTracepointsPass(PassBase):
    def __init__(
        self, specs: dict[str, ModuleCallSignature], sig: ExportGraphSignature
    ) -> None: ...
    def call(self, gm: torch.fx.GraphModule) -> PassResult | None: ...
