from typing import overload

import torch
import torch.types

from .rigid_utils import Rigid

@overload
def pseudo_beta_fn(
    aatype: torch.Tensor, all_atom_positions: torch.Tensor, all_atom_masks: None
) -> torch.Tensor: ...
@overload
def pseudo_beta_fn(
    aatype: torch.Tensor, all_atom_positions: torch.Tensor, all_atom_masks: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]: ...
def pseudo_beta_fn(
    aatype, all_atom_positions, all_atom_masks
):  # -> tuple[Tensor, Tensor] | Tensor:
    ...
def atom14_to_atom37(
    atom14: torch.Tensor, batch: dict[str, torch.Tensor]
) -> torch.Tensor: ...
def build_template_angle_feat(
    template_feats: dict[str, torch.Tensor],
) -> torch.Tensor: ...
def build_template_pair_feat(
    batch: dict[str, torch.Tensor],
    min_bin: torch.types.Number,
    max_bin: torch.types.Number,
    no_bins: int,
    use_unit_vector: bool = ...,
    eps: float = ...,
    inf: float = ...,
) -> torch.Tensor: ...
def build_extra_msa_feat(batch: dict[str, torch.Tensor]) -> torch.Tensor: ...
def torsion_angles_to_frames(
    r: Rigid, alpha: torch.Tensor, aatype: torch.Tensor, rrgdf: torch.Tensor
) -> Rigid: ...
def frames_and_literature_positions_to_atom14_pos(
    r: Rigid,
    aatype: torch.Tensor,
    default_frames: torch.Tensor,
    group_idx: torch.Tensor,
    atom_mask: torch.Tensor,
    lit_positions: torch.Tensor,
) -> torch.Tensor: ...
