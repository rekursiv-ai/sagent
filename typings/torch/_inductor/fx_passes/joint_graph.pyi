from collections.abc import Sequence
from typing import Any

from torch._inductor.constant_folding import ConstantFolder
from torch.utils._ordered_set import OrderedSet

import torch

from ..pattern_matcher import (
    Arg,
    CallFunction,
    KeywordArg,
    Match,
    init_once_fakemode,
    register_graph_pattern,
)

log = ...
patterns = ...
aten = ...
prims = ...
pass_patterns = ...

@init_once_fakemode
def lazy_init() -> None: ...
def remove_no_ops(
    gm: torch.fx.GraphModule,
    zeros: OrderedSet[torch.fx.Node],
    ones: OrderedSet[torch.fx.Node],
) -> None: ...
def remove_redundant_views(gm: torch.fx.GraphModule) -> None: ...

class UniformValueConstantFolder(ConstantFolder):
    def __init__(self, gm, skip_constructors=...) -> None: ...
    def insertable_tensor_check(self, t: torch.Tensor) -> bool: ...
    def add_node_replacement(
        self, node: torch.fx.Node, tensor: torch.Tensor
    ) -> None: ...
    def insert_placerholder_values(self, env: dict[torch.fx.Node, Any]) -> None: ...

def constant_fold_uniform_value(gm: torch.fx.GraphModule) -> None: ...
def canonicalize_quant_mapping(gm: torch.fx.GraphModule) -> None: ...
def canonicalize_aten_ir_passes(gm: torch.fx.GraphModule) -> None: ...
def joint_graph_passes(graph: torch.fx.GraphModule) -> GraphModule: ...
@register_graph_pattern(
    CallFunction(
        torch.ops.prims.iota.default,
        KeywordArg("length"),
        start=KeywordArg("start"),
        step=KeywordArg("step"),
        dtype=KeywordArg("dtype"),
        device=KeywordArg("device"),
        requires_grad=KeywordArg("requires_grad"),
    ),
    pass_dict=patterns,
)
def fix_iota_device(
    match: Match, length, start, step, dtype, device, requires_grad
) -> None: ...
@register_graph_pattern(
    CallFunction(
        torch.ops.prims.convert_element_type.default,
        CallFunction(
            torch.ops.prims.convert_element_type.default,
            KeywordArg("arg"),
            KeywordArg("dtype1"),
        ),
        KeywordArg("dtype2"),
    ),
    pass_dict=patterns,
)
def pointless_convert(
    match: Match, arg, dtype1: torch.dtype, dtype2: torch.dtype
) -> None: ...
def definitely_equal(
    old_sizes: Sequence[torch.SymInt | int],
    new_sizes: Sequence[torch.SymInt | torch.fx.Node | int],
) -> bool: ...
@register_graph_pattern(
    CallFunction(torch.ops.aten.view.default, KeywordArg("arg"), KeywordArg("size")),
    pass_dict=patterns,
)
def pointless_view(match: Match, arg, size) -> None: ...
@register_graph_pattern(
    CallFunction(
        aten.view.default,
        CallFunction(aten.view.default, KeywordArg("arg"), KeywordArg("size1")),
        KeywordArg("size2"),
    ),
    pass_dict=patterns,
)
def pointless_view_pair(match: Match, arg, size1, size2) -> None: ...
@register_graph_pattern(
    CallFunction(
        aten.permute.default,
        CallFunction(aten.permute.default, KeywordArg("arg"), KeywordArg("perm1")),
        KeywordArg("perm2"),
    ),
    pass_dict=patterns,
)
def pointless_permute_pair(match: Match, arg, perm1, perm2) -> None: ...
@register_graph_pattern(CallFunction(aten.bmm, Arg(), Arg()), pass_dict=patterns)
def bmm_to_mm(match: Match, mat1: torch.fx.Node, mat2: torch.fx.Node) -> None: ...
def mul_softmax_pattern(
    match: Match, *, inp, other, dim, keepdim, dtype=...
) -> None: ...
def div_softmax_pattern(
    match: Match, *, inp, other, dim, keepdim, dtype=...
) -> None: ...
