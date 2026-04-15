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

_BOLTZMANN_TEMPERATURE_K = 298.15
_GAS_CONSTANT_KCAL_PER_MOL_K = 0.0019872041


@dataclass(frozen=True)
class OptimizedConformer:
    """An optimized conformer with its post-optimization energy."""

    conformer_id: int
    energy: float
    delta_energy: float


@dataclass
class OptimizedConformerPool:
    """A molecule and its optimized conformers sorted by energy."""

    mol: Chem.Mol
    conformers: list[OptimizedConformer]


@dataclass
class LowestEnergyConformer:
    """The single optimized conformer used for lowest-energy 3D targets."""

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


def generate_optimized_conformer_pool(
    smiles: str,
    config: ConformerConfig,
    *,
    seed: int,
) -> OptimizedConformerPool | None:
    """Generate and optimize a conformer pool sorted by ascending energy."""
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

    sorted_conformers = sorted(energies.items(), key=lambda item: item[1])
    best_energy = sorted_conformers[0][1]
    conformers = [
        OptimizedConformer(
            conformer_id=conformer_id,
            energy=energy,
            delta_energy=energy - best_energy,
        )
        for conformer_id, energy in sorted_conformers
    ]
    return OptimizedConformerPool(mol=mol, conformers=conformers)


def retain_boltzmann_conformers(
    pool: OptimizedConformerPool,
    config: ConformerConfig,
) -> OptimizedConformerPool:
    """Filter a conformer pool to the best low-energy Boltzmann support set."""
    retained = [
        conformer
        for conformer in pool.conformers
        if conformer.delta_energy <= config.max_delta_energy_kcal
    ][: config.n_keep_best]
    return OptimizedConformerPool(mol=pool.mol, conformers=retained)


def compute_boltzmann_weights(
    delta_energies: np.ndarray,
    *,
    temperature_kelvin: float = _BOLTZMANN_TEMPERATURE_K,
) -> np.ndarray:
    """Return normalized Boltzmann weights for conformer delta energies."""
    if delta_energies.ndim != 1:
        raise ValueError("delta_energies must be a 1D array.")
    if delta_energies.size == 0:
        return np.zeros(0, dtype=np.float64)

    scaled = -delta_energies.astype(np.float64, copy=False) / (
        _GAS_CONSTANT_KCAL_PER_MOL_K * temperature_kelvin
    )
    scaled -= np.max(scaled)
    weights = np.exp(scaled)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Could not normalize Boltzmann weights from the conformer energies.")
    return weights / total


def generate_lowest_energy_conformer(
    smiles: str,
    config: ConformerConfig,
    *,
    seed: int,
) -> LowestEnergyConformer | None:
    """Generate conformers and keep only the lowest-energy optimized conformer."""
    pool = generate_optimized_conformer_pool(smiles, config, seed=seed)
    if pool is None or not pool.conformers:
        return None

    return LowestEnergyConformer(
        mol=pool.mol,
        conformer_id=pool.conformers[0].conformer_id,
    )
