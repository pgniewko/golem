"""RDKit conformer generation utilities for offline 3D targets."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

from golem.config import ConformerConfig

logger = logging.getLogger(__name__)


@dataclass
class ConformerEnsemble:
    """Retained conformers and their relative energies."""

    mol: Chem.Mol
    conformer_ids: List[int]
    relative_energies_kcal: np.ndarray


def _molecule_seed(smiles: str, seed: int) -> int:
    """Derive a deterministic per-molecule embedding seed."""
    digest = hashlib.sha256(smiles.encode()).digest()
    smi_seed = int.from_bytes(digest[:4], byteorder="little", signed=False)
    # RDKit embed params expect a non-negative signed 32-bit seed.
    return (seed ^ smi_seed) & 0x7FFFFFFF


def _embedding_params(embedding: str, seed: int):
    """Build RDKit embedding parameters."""
    if embedding != "ETKDGv3":
        raise ValueError(f"Unsupported embedding method: {embedding!r}")

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = -1.0
    return params


def _run_forcefield_optimization(
    mol: Chem.Mol,
    method: str,
) -> dict[int, float] | None:
    """Optimize all conformers with the requested force field."""
    if method == "MMFF":
        if not AllChem.MMFFHasAllMoleculeParams(mol):
            return None
        try:
            results = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0)
        except Exception:
            logger.debug("MMFF optimization failed", exc_info=True)
            return None
    elif method == "UFF":
        try:
            results = AllChem.UFFOptimizeMoleculeConfs(mol, numThreads=0)
        except Exception:
            logger.debug("UFF optimization failed", exc_info=True)
            return None
    else:
        raise ValueError(f"Unsupported optimization method: {method!r}")

    energies: dict[int, float] = {}
    for conformer, (_, energy) in zip(mol.GetConformers(), results, strict=False):
        if np.isfinite(energy):
            energies[conformer.GetId()] = float(energy)
    return energies or None


def _select_conformers(
    mol: Chem.Mol,
    energies: dict[int, float],
    config: ConformerConfig,
) -> list[tuple[int, float]]:
    """Filter by energy and prune near-duplicate conformers by RMSD."""
    min_energy = min(energies.values())
    candidates = [
        (conf_id, energy - min_energy)
        for conf_id, energy in sorted(energies.items(), key=lambda item: item[1])
        if (energy - min_energy) <= config.energy_window_kcal
    ]

    kept: list[tuple[int, float]] = []
    for conf_id, relative_energy in candidates:
        if not kept:
            kept.append((conf_id, relative_energy))
        else:
            is_distinct = True
            for kept_id, _ in kept:
                try:
                    rms = rdMolAlign.GetBestRMS(mol, mol, prbId=conf_id, refId=kept_id)
                except Exception:
                    logger.debug("RMS comparison failed", exc_info=True)
                    is_distinct = False
                    break
                if rms < config.prune_rms:
                    is_distinct = False
                    break
            if is_distinct:
                kept.append((conf_id, relative_energy))

        if len(kept) >= config.n_keep:
            break

    return kept


def generate_conformer_ensemble(
    smiles: str,
    config: ConformerConfig,
    *,
    seed: int,
) -> ConformerEnsemble | None:
    """Generate and retain a low-energy, RMS-pruned conformer ensemble."""
    if config.backend != "rdkit":
        raise ValueError(f"Unsupported conformer backend: {config.backend!r}")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    params = _embedding_params(config.embedding, _molecule_seed(smiles, seed))

    try:
        conf_ids = list(
            AllChem.EmbedMultipleConfs(mol, numConfs=config.n_generate, params=params)
        )
    except Exception:
        logger.debug("Conformer embedding failed for %s", smiles, exc_info=True)
        return None

    if not conf_ids:
        return None

    energies = _run_forcefield_optimization(mol, config.optimize)
    if energies is None:
        energies = _run_forcefield_optimization(mol, config.fallback_optimize)
    if energies is None:
        return None

    kept = _select_conformers(mol, energies, config)
    if not kept:
        return None

    return ConformerEnsemble(
        mol=mol,
        conformer_ids=[conf_id for conf_id, _ in kept],
        relative_energies_kcal=np.array(
            [relative_energy for _, relative_energy in kept],
            dtype=np.float64,
        ),
    )
