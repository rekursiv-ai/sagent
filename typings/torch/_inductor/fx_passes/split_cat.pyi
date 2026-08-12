from collections.abc import Generator
from collections.abc import Callable
from typing import Any

import operator

from torch.utils._ordered_set import OrderedSet

import torch

from ..pattern_matcher import (
    MULTIPLE,
    CallFunction,
    CallFunctionVarArgs,
    CallMethodVarArgs,
    Ignored,
    KeywordArg,
    ListOf,
    Match,
    MatchContext,
    PatternMatcherPass,
    RepeatedExpr,
    register_graph_pattern,
)

log = ...
type _Arguments = tuple[torch.fx.node.Argument, ...]
type _TransformParam = tuple[
    _Arguments | None, _Arguments | None, _Arguments | None, _Arguments | None
]
type _Range = tuple[int, int]
PRE_GRAD_PATTERNS: dict[str, PatternMatcherPass] = ...
POST_GRAD_PATTERNS: dict[str, PatternMatcherPass] = ...
pre_grad_pass_names = ...
post_grad_pass_names = ...
backend = ...

def construct_pattern_matcher_pass(pass_name: str) -> PatternMatcherPass: ...
def normalize_split_base(
    match: Match,
    _get_split_args: Callable[
        [torch.fx.Node], tuple[torch.fx.Node | None, Any | None, int | None]
    ],
) -> None: ...
@register_graph_pattern(
    CallFunctionVarArgs(torch.split, users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("normalization_pass"),
)
@register_graph_pattern(
    CallMethodVarArgs("split", users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("normalization_pass"),
)
def normalize_split_default(match: Match, *args, **kwargs) -> None: ...
@register_graph_pattern(
    CallFunctionVarArgs(torch.split, users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("remove_split_with_size_one_pass"),
)
@register_graph_pattern(
    CallMethodVarArgs("split", users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("remove_split_with_size_one_pass"),
)
def remove_split_with_size_one(match: Match, *args, **kwargs) -> None: ...
@register_graph_pattern(
    CallFunctionVarArgs(torch.unbind, users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("normalization_pass"),
)
@register_graph_pattern(
    CallMethodVarArgs("unbind", users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("normalization_pass"),
)
def normalize_unbind_default(match: Match, *args, **kwargs) -> None: ...
@register_graph_pattern(
    CallFunctionVarArgs([torch.cat, torch.concat], users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("normalization_pass"),
)
def normalize_cat_default(match: Match, *args, **kwargs) -> None: ...
@register_graph_pattern(
    CallFunctionVarArgs(torch.stack, users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("normalization_pass"),
)
def normalize_stack_default(match: Match, *args, **kwargs) -> None: ...
def find_next_users(split_node: torch.fx.Node) -> list[torch.fx.Node]: ...
@register_graph_pattern(
    CallMethodVarArgs("squeeze", users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("normalization_pass"),
)
def normalize_squeeze_default(match: Match, *args, **kwargs) -> None: ...
@register_graph_pattern(
    CallMethodVarArgs("reshape", users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("normalization_pass"),
)
def normalize_reshape_default(match: Match, *args, **kwargs) -> None: ...
@register_graph_pattern(
    CallMethodVarArgs("clamp", users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("normalization_pass"),
)
@register_graph_pattern(
    CallFunctionVarArgs(torch.clamp, users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("normalization_pass"),
)
def normalize_clamp_default(match: Match, *args, **kwargs) -> None: ...
@register_graph_pattern(
    CallMethodVarArgs("detach", users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("normalization_pass"),
)
def normalize_detach_default(match: Match, *args, **kwargs) -> None: ...

class TorchSplit(CallFunction):
    def __init__(self, arg, sizes, func=...) -> None: ...

@register_graph_pattern(
    TorchSplit(
        CallFunction(
            operator.getitem,
            TorchSplit(
                KeywordArg("first_split_input"), KeywordArg("first_split_sections")
            ),
            Ignored(),
        ),
        KeywordArg("next_split_sections"),
    ),
    pass_dict=construct_pattern_matcher_pass("merge_splits_pass"),
)
def merge_splits(
    match: Match,
    first_split_input: torch.fx.Node,
    first_split_sections: list[int],
    next_split_sections: list[int],
    dim: int,
) -> None: ...

class SplitCatSimplifier:
    def simplify(
        self,
        graph: torch.fx.Graph,
        split_node: torch.fx.Node,
        split_sections: list[int],
    ) -> None: ...
    def get_user_input_list(
        self, split_node: torch.fx.Node, next_users: list[torch.fx.Node]
    ) -> list[list[torch.fx.Node | _Range]]: ...
    def get_merged_user_inputs(
        self, split_node: torch.fx.Node, cat_node: torch.fx.Node
    ) -> list[torch.fx.Node | _Range]: ...
    def get_non_cat_node_input(
        self, split_node: torch.fx.Node, node: torch.fx.Node
    ) -> list[_Range]: ...
    def merge_consecutive_inputs(
        self, inputs: list[torch.fx.Node | int]
    ) -> list[torch.fx.Node | _Range]: ...
    def get_simplified_split_ranges(
        self,
        split_sections,
        next_users,
        user_inputs_list: list[list[torch.fx.Node | _Range]],
    ) -> list[_Range] | None: ...
    def has_non_overlapping_ranges(self, ranges: list[_Range]) -> bool: ...
    def fill_gaps(self, ranges: list[_Range], min_: int, max_: int) -> list[_Range]: ...
    def get_transform_params(
        self,
        split_node: torch.fx.Node,
        next_users: list[torch.fx.Node],
        user_inputs_list: list[list[torch.fx.Node | _Range]],
    ) -> list[list[_TransformParam]] | None: ...
    def replace_split(
        self,
        graph: torch.fx.Graph,
        split_node: torch.fx.Node,
        split_sections: list[int],
        user_inputs_list: list[list[torch.fx.Node | _Range]],
        split_ranges: list[_Range],
    ) -> list[list[torch.fx.Node]]: ...
    def replace_cat(
        self,
        graph: torch.fx.Graph,
        split_node: torch.fx.Node,
        next_users: list[torch.fx.Node],
        user_inputs_list_new,
        transform_params_list: list[list[_TransformParam]],
    ) -> None: ...
    def erase_old_nodes(
        self,
        graph: torch.fx.Graph,
        split_node: torch.fx.Node,
        next_users: list[torch.fx.Node],
    ) -> None: ...

class UnbindCatRemover(SplitCatSimplifier):
    def remove_unbind(
        self, graph: torch.fx.Graph, unbind_node: torch.fx.Node
    ) -> None: ...
    def get_simplified_split_ranges(
        self,
        split_sections: list[int],
        next_users: list[torch.fx.Node],
        user_inputs_list: list[list[torch.fx.Node | _Range]],
    ) -> list[_Range] | None: ...
    def get_transform_params(
        self,
        split_node: torch.fx.Node,
        next_users: list[torch.fx.Node],
        user_inputs_list: list[list[torch.fx.Node | _Range]],
    ) -> list[list[_TransformParam]] | None: ...

class GetItem(CallFunction):
    def __init__(self, arg, index, _users=...) -> None: ...
    def find_anchor_nodes(
        self, ctx: MatchContext, searched: OrderedSet[torch.fx.Node]
    ) -> Generator[Node, Any, None]: ...

@register_graph_pattern(
    RepeatedExpr(
        CallFunction(
            torch.squeeze,
            GetItem(
                TorchSplit(KeywordArg("split_input"), KeywordArg("split_sizes")),
                Ignored(),
            ),
            KeywordArg("dim"),
            _users=MULTIPLE,
        )
    ),
    pass_dict=construct_pattern_matcher_pass("split_cat_pass"),
)
@register_graph_pattern(
    RepeatedExpr(
        CallFunction(
            torch.squeeze,
            GetItem(
                TorchSplit(KeywordArg("split_input"), KeywordArg("split_sizes")),
                Ignored(),
            ),
            dim=KeywordArg("dim"),
            _users=MULTIPLE,
        )
    ),
    pass_dict=construct_pattern_matcher_pass("split_cat_pass"),
)
def merge_split_squeeze(
    match: Match, split_input: torch.fx.Node, split_sizes: list[int], dim: int
) -> None: ...

getitem_unbind = ...

@register_graph_pattern(
    CallFunction([torch.stack, torch.cat], getitem_unbind, Ignored(), _users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("unbind_stack_pass"),
)
@register_graph_pattern(
    CallFunction(
        [torch.stack, torch.cat], getitem_unbind, dim=Ignored(), _users=MULTIPLE
    ),
    pass_dict=construct_pattern_matcher_pass("unbind_stack_pass"),
)
@register_graph_pattern(
    CallFunction(
        [torch.stack, torch.cat], tensors=getitem_unbind, dim=Ignored(), _users=MULTIPLE
    ),
    pass_dict=construct_pattern_matcher_pass("unbind_stack_pass"),
)
def merge_unbind_stack(match: Match, unbind_input: torch.fx.Node, dim: int) -> None: ...

getitem_split = ...
reshape_getitem_split = ...

@register_graph_pattern(
    CallFunction(
        [torch.stack, torch.cat], tensors=getitem_split, dim=Ignored(), _users=MULTIPLE
    ),
    pass_dict=construct_pattern_matcher_pass("split_cat_pass"),
)
@register_graph_pattern(
    CallFunction(
        [torch.stack, torch.cat], getitem_split, dim=Ignored(), _users=MULTIPLE
    ),
    pass_dict=construct_pattern_matcher_pass("split_cat_pass"),
)
@register_graph_pattern(
    CallFunction([torch.stack, torch.cat], getitem_split, Ignored(), _users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("split_cat_pass"),
)
def simplify_split_cat(match: Match, split_sections: list[int], dim: int) -> None: ...
def has_same_parent_node(node: torch.fx.Node) -> bool: ...
def remove_zeros(split_sections: list[int]) -> tuple[list[Any], dict[Any, Any]]: ...
def is_sorted_and_consecutive(arr: list[int]) -> bool: ...
def calculate_fused_tensor_size(
    split_node: torch.fx.Node, indices: list[int]
) -> int: ...
@register_graph_pattern(
    CallFunction(torch.cat, getitem_split, dim=Ignored(), _users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("merge_getitem_cat_pass"),
)
def merge_getitem_cat(match: Match, split_sections: list[int], dim: int) -> None: ...
@register_graph_pattern(
    CallFunction(torch.cat, getitem_split, dim=Ignored(), _users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("mutate_cat_pass"),
)
def mutate_cat_node(match: Match, split_sections: list[int], dim: int) -> None: ...

getitem_split_aten = ...

@register_graph_pattern(
    CallFunctionVarArgs(torch.ops.aten.split.Tensor, users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("normalization_aten_pass"),
)
def normalize_split_default_aten(match: Match, *args, **kwargs) -> None: ...
@register_graph_pattern(
    CallFunctionVarArgs(torch.ops.aten.split_with_sizes.default, users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("normalization_aten_pass"),
)
def normalize_split_with_size_default_aten(match: Match, *args, **kwargs) -> None: ...
@register_graph_pattern(
    CallFunction(
        torch.ops.aten.cat.default, getitem_split_aten, dim=Ignored(), _users=MULTIPLE
    ),
    pass_dict=construct_pattern_matcher_pass("split_cat_aten_pass"),
)
def merge_split_cat_aten(match: Match, *args, **kwargs) -> None: ...
@register_graph_pattern(
    CallFunction(
        torch.ops.aten.cat.default,
        ListOf(
            CallFunctionVarArgs(torch.ops.aten.select.int, users=MULTIPLE), partial=True
        ),
        dim=Ignored(),
        _users=MULTIPLE,
    ),
    pass_dict=construct_pattern_matcher_pass("select_cat_aten_pass"),
)
def merge_select_cat_aten(match: Match, *args, **kwargs) -> None: ...
@register_graph_pattern(
    CallFunctionVarArgs(torch.ops.aten.cat.default, users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("normalization_aten_pass"),
)
def normalize_cat_default_aten(match: Match, *args, **kwargs) -> None: ...
@register_graph_pattern(
    CallFunction(
        torch.ops.aten.cat,
        ListOf(CallFunctionVarArgs(torch.ops.aten.unsqueeze)),
        _users=MULTIPLE,
    ),
    pass_dict=construct_pattern_matcher_pass("unbind_stack_aten_pass"),
)
def merge_unbind_stack_aten(match: Match, *args, **kwargs) -> None: ...
def divide_into_consecutive_sublists(indices: list[int]) -> list[list[int]]: ...
def update_args_from_split_getitem(
    graph: torch.fx.Graph,
    node: torch.fx.Node,
    getitem_indices: list[int],
    parents_seen: list[torch.fx.Node],
    new_cat_args: list[torch.fx.Node],
    new_cat_args_meta: list[torch.fx.Node],
    idx_to_getitems: dict[int, torch.fx.Node],
    threshold_to_cat: int = ...,
) -> None: ...
def reshape_cat_node(
    graph: torch.fx.Graph,
    cat_node: torch.fx.Node,
    unbind_input: torch.fx.Node,
    cat_dim: int,
    unbind_dim: int,
    cat_shape: torch.Size,
) -> torch.fx.Node: ...
def update_args_from_unbind_getitem(
    graph: torch.fx.Graph,
    node: torch.fx.Node,
    getitem_indices: list[int],
    parents_seen: list[torch.fx.Node],
    new_cat_args: list[torch.fx.Node],
    new_cat_args_meta: list[torch.fx.Node],
    idx_to_getitems: dict[int, torch.fx.Node],
    threshold_to_cat: int = ...,
) -> None: ...
def construct_cat_args(
    graph: torch.fx.Graph,
    cat_or_stack_node: torch.fx.Node,
    inputs: list[torch.fx.Node],
    split_or_unbind_node: torch.fx.Node,
    threshold_to_cat: int = ...,
    run_update_func: Callable = ...,
) -> tuple[list[torch.fx.Node], list[torch.Tensor]]: ...
def remove_split_unbind_children(
    graph: torch.fx.Graph, inputs: list[torch.fx.Node]
) -> None: ...
@register_graph_pattern(
    CallFunction(torch.cat, getitem_split, dim=Ignored(), _users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("split_cat_to_slices_pass"),
)
def split_cat_to_slices(match: Match, split_sections: list[int], dim: int) -> None: ...
@register_graph_pattern(
    CallFunction(torch.cat, getitem_unbind, dim=Ignored(), _users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("unbind_cat_to_view_pass"),
)
def unbind_cat_to_view(match: Match, unbind_input: torch.fx.Node, dim: int) -> None: ...
def reshape_cat_node_to_stack(
    graph: torch.fx.Graph,
    cat_node: torch.fx.Node,
    stack_node: torch.fx.Node,
    split_or_unbind_dim: int,
) -> None: ...
def convert_reshape_cat_arg_to_stack(
    graph: torch.fx.Graph,
    cat_node: torch.fx.Node,
    stack_node: torch.fx.Node,
    stack_node_shape: torch.Size,
    stack_dim: int,
    split_dim: int,
) -> torch.fx.Node: ...
@register_graph_pattern(
    CallFunction(torch.stack, getitem_split, dim=Ignored(), _users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("split_stack_to_cats_pass"),
)
def split_stack_to_cats(match: Match, split_sections: list[int], dim: int) -> None: ...
@register_graph_pattern(
    CallFunction(torch.stack, getitem_unbind, dim=Ignored(), _users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("unbind_stack_to_slices_pass"),
)
def unbind_stack_to_slices(
    match: Match, unbind_input: torch.fx.Node, dim: int
) -> None: ...
def get_view_shape_list(cat_arg: torch.fx.Node, stack_dim: int) -> list[int]: ...
@register_graph_pattern(
    CallFunction(torch.stack, reshape_getitem_split, dim=Ignored(), _users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("move_reshape_out_of_split_stack_pass"),
)
def move_reshape_out_of_split_stack(match: Match, *args, **kwargs) -> None: ...

view_getitem_split_aten = ...

@register_graph_pattern(
    CallFunction(
        torch.ops.aten.cat.default,
        view_getitem_split_aten,
        dim=Ignored(),
        _users=MULTIPLE,
    ),
    pass_dict=construct_pattern_matcher_pass("move_view_after_cat_aten_pass"),
)
def move_view_after_cat(match: Match, *args, **kwargs) -> None: ...
def match_einsum_strings(s: str) -> bool: ...
@register_graph_pattern(
    CallFunctionVarArgs(torch.functional.einsum, users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("einsum_to_pointwise_pass"),
)
def replace_einsum_to_pointwise(match: Match, *args, **kwargs) -> None: ...
