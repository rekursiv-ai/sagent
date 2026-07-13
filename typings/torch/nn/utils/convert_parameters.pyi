from collections.abc import Iterable

import torch

def parameters_to_vector(parameters: Iterable[torch.Tensor]) -> torch.Tensor: ...
def vector_to_parameters(
    vec: torch.Tensor, parameters: Iterable[torch.Tensor]
) -> None: ...
