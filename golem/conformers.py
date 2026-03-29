"""RDKit conformer generation utilities for offline 3D targets."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from golem.config import ConformerConfig

logger = logging.getLogger(__name__)


@dataclass
class LowestEnergyConformer:
    """The single optimized conformer used for 3D descriptor targets."""

    mol: Chem.Mol
    conformer_id: int


def _molecule_seed(smiles: str, seed: int) -> int:
    digest = hashlib.sha256(smiles.encode()).digest()
    return (seed ^ int.from_bytes(digest[:4], byteorder="little")) & 0x7FFFFFFF


def _optimize_conformers(mol: Chem.Mol, method: str) -> dict[int, float] | None:
    if method == "MMFF":
        if not AllChem.MMFFHasAllMoleculeParams(mol):
            return None
        optimize = AllChem.MMFFOptimizeMoleculeConfs
    elif method == "UFF":
        optimize = AllChem.UFFOptimizeMoleculeConfs
    else:
        raise ValueError(f"Unsupported optimization method: {method!r}")

    try:
        results = optimize(mol, numThreads=0)
    except Exception:
        logger.debug("Conformer optimization failed", exc_info=True)
        return None

    energies = {
        conformer.GetId(): float(energy)
        for conformer, (_, energy) in zip(mol.GetConformers(), results, strict=False)
        if np.isfinite(energy)
    }
    return energies or None


def generate_lowest_energy_conformer(
    smiles: str,
    config: ConformerConfig,
    *,
    seed: int,
) -> LowestEnergyConformer | None:
    """Generate conformers and keep only the lowest-energy optimized conformer."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = _molecule_seed(smiles, seed)
    params.pruneRmsThresh = -1.0

    try:
        conf_ids = list(
            AllChem.EmbedMultipleConfs(mol, numConfs=config.n_generate, params=params)
        )
    except Exception:
        logger.debug("Conformer embedding failed for %s", smiles, exc_info=True)
        return None

    if not conf_ids:
        return None

    energies = _optimize_conformers(mol, "MMFF") or _optimize_conformers(mol, "UFF")
    if energies is None:
        return None

    best_conformer_id = min(energies.items(), key=lambda item: item[1])[0]

    return LowestEnergyConformer(
        mol=mol,
        conformer_id=best_conformer_id,
    )
