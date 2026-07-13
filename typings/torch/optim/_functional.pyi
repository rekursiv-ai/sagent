from torch import Tensor

r"""Functional interface."""

def sparse_adam(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    state_steps: list[int],
    *,
    eps: float,
    beta1: float,
    beta2: float,
    lr: float,
    maximize: bool,
) -> None: ...
