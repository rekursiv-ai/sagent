from collections.abc import Mapping, Sequence
from typing import Any

import dataclasses

import numpy as np

"""Protein data type."""
type FeatureDict = Mapping[str, np.ndarray]
type ModelOutput = Mapping[str, Any]
PICO_TO_ANGSTROM = ...

@dataclasses.dataclass(frozen=True)
class Protein:
    atom_positions: np.ndarray
    aatype: np.ndarray
    atom_mask: np.ndarray
    residue_index: np.ndarray
    b_factors: np.ndarray
    chain_index: np.ndarray | None = ...
    remark: str | None = ...
    parents: Sequence[str] | None = ...
    parents_chain_index: Sequence[int] | None = ...

def from_proteinnet_string(proteinnet_str: str) -> Protein: ...
def get_pdb_headers(prot: Protein, chain_id: int = ...) -> list[str]: ...
def add_pdb_headers(prot: Protein, pdb_str: str) -> str: ...
def to_pdb(prot: Protein) -> str: ...
def ideal_atom_mask(prot: Protein) -> np.ndarray: ...
def from_prediction(
    features: FeatureDict,
    result: ModelOutput,
    b_factors: np.ndarray | None = ...,
    chain_index: np.ndarray | None = ...,
    remark: str | None = ...,
    parents: Sequence[str] | None = ...,
    parents_chain_index: Sequence[int] | None = ...,
) -> Protein: ...
