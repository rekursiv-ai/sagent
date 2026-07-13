from torch._library.fake_profile import OpProfile

import torch

def insert_custom_op_guards(
    gm: torch.fx.GraphModule, ops_to_guard: set[str]
) -> None: ...
def get_op_profiles(
    gm: torch.fx.GraphModule, ops_to_guard: set[str]
) -> dict[str, set[OpProfile]]: ...
