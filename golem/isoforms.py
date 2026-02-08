"""Isoform enumeration: tautomers, protonation states, neutralization.

Each enumeration function creates local RDKit/Dimorphite instances (no global
singletons) for thread safety.  All isoforms are deduplicated by canonical
SMILES, with the original molecule always at index 0.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from tqdm import tqdm

from golem.config import IsoformConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _canonical(mol: Chem.Mol) -> Optional[str]:
    """Return canonical SMILES or None on failure."""
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def _enumerate_tautomers(mol: Chem.Mol, max_tautomers: int = 25) -> List[Chem.Mol]:
    """Enumerate tautomers using a **local** TautomerEnumerator instance."""
    try:
        enumerator = rdMolStandardize.TautomerEnumerator()
        enumerator.SetMaxTautomers(max_tautomers)
        return list(enumerator.Enumerate(mol))
    except Exception as e:
        logger.debug("Tautomer enumeration failed for %s: %s", _canonical(mol), e)
        return []


def _enumerate_protonation(
    mol: Chem.Mol,
    ph_range: tuple[float, float] = (6.4, 8.4),
    max_protomers: int = 25,
) -> List[Chem.Mol]:
    """Enumerate protonation states with Dimorphite-DL (primary) or Uncharger (fallback)."""
    smi = _canonical(mol)
    if smi is None:
        return []

    # --- Primary: Dimorphite-DL ---
    try:
        from dimorphite_dl import DimorphiteDL

        dl = DimorphiteDL(
            min_ph=ph_range[0],
            max_ph=ph_range[1],
            max_variants=max_protomers,
            label_states=False,
        )
        protomer_smiles = dl.protonate(smi)

        protomers: List[Chem.Mol] = []
        for psmi in protomer_smiles:
            pmol = Chem.MolFromSmiles(psmi)
            if pmol is not None:
                protomers.append(pmol)
        return protomers

    except ImportError:
        logger.warning(
            "dimorphite_dl not installed – falling back to RDKit Uncharger. "
            "Install with: pip install dimorphite_dl"
        )
    except Exception as e:
        logger.debug("Dimorphite-DL failed for %s: %s – falling back to Uncharger", smi, e)

    # --- Fallback: RDKit Uncharger ---
    try:
        uncharger = rdMolStandardize.Uncharger()
        uncharged = uncharger.uncharge(mol)
        if uncharged is not None:
            return [uncharged]
    except Exception as e:
        logger.debug("Uncharger fallback failed for %s: %s", smi, e)

    return []


def _neutralize(mol: Chem.Mol) -> List[Chem.Mol]:
    """Return the neutralized form of *mol* (local Uncharger instance)."""
    try:
        uncharger = rdMolStandardize.Uncharger()
        neutralized = uncharger.uncharge(mol)
        if neutralized is not None:
            return [neutralized]
    except Exception as e:
        logger.debug("Neutralization failed for %s: %s", _canonical(mol), e)
    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enumerate_isoforms(smiles: str, config: IsoformConfig) -> List[str]:
    """Enumerate isoforms for a single SMILES string.

    Returns a deduplicated list of canonical SMILES.  The original molecule
    is always at index 0.

    Args:
        smiles: Input SMILES string.
        config: IsoformConfig controlling which enumerations to run.

    Returns:
        List of unique canonical SMILES (original first).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning("Cannot parse SMILES: %s", smiles)
        return [smiles]  # keep original even if unparseable

    # Always include the original
    original_can = _canonical(mol)
    if original_can is None:
        return [smiles]

    seen: Set[str] = {original_can}
    isoforms: List[str] = [original_can]

    def _add(mols: List[Chem.Mol]) -> None:
        for m in mols:
            if len(isoforms) >= config.max_isoforms:
                return
            can = _canonical(m)
            if can is not None and can not in seen:
                seen.add(can)
                isoforms.append(can)

    # Tautomers
    if config.tautomers and len(isoforms) < config.max_isoforms:
        _add(_enumerate_tautomers(mol, max_tautomers=config.max_tautomers))

    # Protonation states
    if config.protonation and len(isoforms) < config.max_isoforms:
        _add(_enumerate_protonation(mol, ph_range=config.ph_range, max_protomers=config.max_isoforms))

    # Neutralization
    if config.neutralization and len(isoforms) < config.max_isoforms:
        _add(_neutralize(mol))

    return isoforms


def enumerate_isoforms_batch(
    smiles_list: List[str],
    config: IsoformConfig,
) -> Dict[str, List[str]]:
    """Batch isoform enumeration with progress bar.

    Args:
        smiles_list: List of parent SMILES strings.
        config: IsoformConfig controlling which enumerations to run.

    Returns:
        Dict mapping each parent SMILES to its list of isoform SMILES
        (including the parent itself as the first element).
    """
    results: Dict[str, List[str]] = {}
    total_isoforms = 0

    for smi in tqdm(smiles_list, desc="Enumerating isoforms", unit="mol"):
        isoforms = enumerate_isoforms(smi, config)
        results[smi] = isoforms
        total_isoforms += len(isoforms)

    logger.info(
        "Isoform enumeration complete: %d parents → %d total isoforms (%.1f× expansion)",
        len(smiles_list),
        total_isoforms,
        total_isoforms / max(len(smiles_list), 1),
    )
    return results
