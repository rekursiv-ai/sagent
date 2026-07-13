from collections.abc import Callable

from torch._functorch._activation_checkpointing.graph_info_provider import (
    GraphInfoProvider,
)

class KnapsackEvaluator:
    def __init__(self, graph_info_provider: GraphInfoProvider) -> None: ...
    def evaluate_knapsack_output(
        self,
        saved_nodes_idxs: list[int],
        recomputable_node_idxs: list[int],
        account_for_backward_pass: bool = ...,
    ) -> dict[str, float]: ...
    def evaluate_distribution_of_results_for_knapsack_algo(
        self,
        knapsack_algo: Callable[
            [list[float], list[float], float], tuple[float, list[int], list[int]]
        ],
        memory_budget_values: list[float],
    ) -> list[dict[str, float]]: ...
    def get_knee_point_memory_budget(
        self,
        knapsack_algo: Callable[
            [list[float], list[float], float], tuple[float, list[int], list[int]]
        ],
        max_mem_budget: float = ...,
        min_mem_budget: float = ...,
        iterations: int = ...,
    ) -> float: ...
