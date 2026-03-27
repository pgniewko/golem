"""RDKit conformer generation utilities for offline 3D targets."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

from golem.config import ConformerConfig

logger = logging.getLogger(__name__)


@dataclass
class ConformerEnsemble:
    """Retained conformers and their relative energies."""

    mol: Chem.Mol
    conformer_ids: list[int]
    relative_energies_kcal: np.ndarray


def _molecule_seed(smiles: str, seed: int) -> int:
    digest = hashlib.sha256(smiles.encode()).digest()
    return (seed ^ int.from_bytes(digest[:4], byteorder="little")) & 0x7FFFFFFF


def _optimize_conformers(mol: Chem.Mol, method: str) -> dict[int, float] | None:
    if method == "MMFF":
        if not AllChem.MMFFHasAllMoleculeParams(mol):
            return None
        runner = AllChem.MMFFOptimizeMoleculeConfs
    elif method == "UFF":
        runner = AllChem.UFFOptimizeMoleculeConfs
    else:
        raise ValueError(f"Unsupported optimization method: {method!r}")

    try:
        results = runner(mol, numThreads=0)
    except Exception:
        logger.debug("Conformer optimization failed", exc_info=True)
        return None

    energies = {
        conformer.GetId(): float(energy)
        for conformer, (_, energy) in zip(mol.GetConformers(), results, strict=False)
        if np.isfinite(energy)
    }
    return energies or None


def _timed_out(started_at: float, timeout_seconds: int) -> bool:
    return timeout_seconds > 0 and (time.monotonic() - started_at) >= timeout_seconds


def _is_timeout_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "timed out" in message or "timeout" in message


def generate_conformer_ensemble(
    smiles: str,
    config: ConformerConfig,
    *,
    seed: int,
) -> tuple[ConformerEnsemble | None, str | None]:
    """Generate a low-energy, RMS-pruned conformer ensemble."""
    started_at = time.monotonic()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "invalid_smiles"

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = _molecule_seed(smiles, seed)
    params.pruneRmsThresh = -1.0
    params.timeout = max(int(config.timeout_seconds), 0)

    try:
        conf_ids = list(
            AllChem.EmbedMultipleConfs(mol, numConfs=config.n_generate, params=params)
        )
    except Exception as exc:
        logger.debug("Conformer embedding failed for %s", smiles, exc_info=True)
        return None, "timeout" if _is_timeout_error(exc) else "embedding_failed"
    if not conf_ids:
        if _timed_out(started_at, config.timeout_seconds):
            return None, "timeout"
        return None, "no_conformers"

    energies = _optimize_conformers(mol, config.optimize)
    if energies is None:
        energies = _optimize_conformers(mol, config.fallback_optimize)
    if energies is None:
        if _timed_out(started_at, config.timeout_seconds):
            return None, "timeout"
        return None, "optimization_failed"

    min_energy = min(energies.values())
    candidates = [
        (conf_id, energy - min_energy)
        for conf_id, energy in sorted(energies.items(), key=lambda item: item[1])
        if energy - min_energy <= config.energy_window_kcal
    ]

    kept: list[tuple[int, float]] = []
    for conf_id, rel_energy in candidates:
        is_duplicate = False
        for kept_id, _ in kept:
            try:
                if rdMolAlign.GetBestRMS(mol, mol, prbId=conf_id, refId=kept_id) < config.prune_rms:
                    is_duplicate = True
                    break
            except Exception:
                logger.debug("RMS comparison failed for %s", smiles, exc_info=True)
                is_duplicate = True
                break
        if is_duplicate:
            continue
        kept.append((conf_id, rel_energy))
        if len(kept) >= config.n_keep:
            break

    if not kept:
        if _timed_out(started_at, config.timeout_seconds):
            return None, "timeout"
        return None, "rms_pruned"

    return (
        ConformerEnsemble(
            mol=mol,
            conformer_ids=[conf_id for conf_id, _ in kept],
            relative_energies_kcal=np.array(
                [rel_energy for _, rel_energy in kept],
                dtype=np.float64,
            ),
        ),
        None,
    )
