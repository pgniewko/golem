"""Isoform enumeration: tautomers, protonation states, neutralization.

Each enumeration function creates local RDKit/Dimorphite instances (no global
singletons) for thread safety.  All isoforms are deduplicated by canonical
SMILES, with the original molecule always at index 0.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from tqdm import tqdm

from golem.config import IsoformConfig

# Suppress RDKit C++ warnings (tautomer limit, kekulization errors, etc.)
# These bypass Python logging and write directly to stderr.
RDLogger.DisableLog('rdApp.*')

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


def _enumerate_tautomers(mol: Chem.Mol, max_tautomers: int = 10) -> List[Chem.Mol]:
    """Enumerate tautomers using a **local** TautomerEnumerator instance."""
    try:
        enumerator = rdMolStandardize.TautomerEnumerator()
        enumerator.SetMaxTautomers(max_tautomers)
        return list(enumerator.Enumerate(mol))
    except Exception as e:
        logger.debug("Tautomer enumeration failed for %s: %s", _canonical(mol), e)
        return []


def _is_valid_protomer(protomer_mol: Chem.Mol, original_smi: str) -> bool:
    """Validate a protomer against known Dimorphite-DL failure modes.

    Rejects:
    1. Nitrogen with >=4 total hydrogens (NH4+ on organic N)
    2. Tertiary amide false protonation (H added to amide N that had no H)
    3. RDKit sanitization / kekulization failures
    """
    # 1. RDKit sanitization check
    try:
        Chem.SanitizeMol(protomer_mol)
    except Exception:
        logger.debug("Protomer failed sanitization: %s", _canonical(protomer_mol))
        return False

    # 2. Nitrogen with >= 4 hydrogens (invalid for organic N)
    for atom in protomer_mol.GetAtoms():
        if atom.GetAtomicNum() == 7 and atom.GetTotalNumHs() >= 4:
            logger.debug(
                "Rejected protomer with N(4+H): %s", _canonical(protomer_mol)
            )
            return False

    # 3. Tertiary amide false protonation
    # Parse the original to find amide N atoms that have no H
    orig_mol = Chem.MolFromSmiles(original_smi)
    if orig_mol is not None:
        # Build set of amide N atom indices in original (N bonded to C=O, no H)
        orig_amide_n_indices: set = set()
        for atom in orig_mol.GetAtoms():
            if atom.GetAtomicNum() != 7:
                continue
            if atom.GetTotalNumHs() > 0:
                continue  # already has H — not a tertiary amide N
            # Check if bonded to a C=O
            for neighbor in atom.GetNeighbors():
                if neighbor.GetAtomicNum() == 6:  # carbon
                    for bond in neighbor.GetBonds():
                        other = bond.GetOtherAtom(neighbor)
                        if (
                            other.GetAtomicNum() == 8
                            and bond.GetBondTypeAsDouble() == 2.0
                        ):
                            orig_amide_n_indices.add(atom.GetIdx())
                            break

        # Check protomer: if any of those amide N now has H, reject
        if orig_amide_n_indices:
            for idx in orig_amide_n_indices:
                if idx < protomer_mol.GetNumAtoms():
                    patom = protomer_mol.GetAtomWithIdx(idx)
                    if patom.GetAtomicNum() == 7 and patom.GetTotalNumHs() > 0:
                        logger.debug(
                            "Rejected protomer with protonated amide N: %s",
                            _canonical(protomer_mol),
                        )
                        return False

    return True


def _enumerate_protonation(
    smi: str,
    ph_range: tuple[float, float] = (6.4, 8.4),
    max_protomers: int = 10,
) -> List[Chem.Mol]:
    """Enumerate protonation states with Dimorphite-DL (primary) or Uncharger (fallback).

    Args:
        smi: Input SMILES string.
        ph_range: (min_pH, max_pH) for protonation enumeration.
        max_protomers: Maximum number of protomers to generate.

    Returns:
        List of valid protomer Mol objects.
    """
    # --- Primary: Dimorphite-DL protonate_smiles ---
    try:
        from dimorphite_dl import protonate_smiles

        protomer_smiles = protonate_smiles(
            smi,
            ph_min=ph_range[0],
            ph_max=ph_range[1],
            max_variants=max_protomers,
        )

        protomers: List[Chem.Mol] = []
        for psmi in protomer_smiles:
            pmol = Chem.MolFromSmiles(psmi)
            if pmol is not None and _is_valid_protomer(pmol, smi):
                protomers.append(pmol)
        return protomers

    except ImportError:
        logger.warning(
            "dimorphite_dl not installed – falling back to RDKit Uncharger. "
            "Install with: pip install dimorphite-dl==2.0.2"
        )
    except Exception as e:
        logger.debug("Dimorphite-DL failed for %s: %s – falling back to Uncharger", smi, e)

    # --- Fallback: RDKit Uncharger ---
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return []
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
            can = _canonical(m)
            if can is not None and can not in seen:
                seen.add(can)
                isoforms.append(can)

    # Tautomers
    if config.tautomers:
        _add(_enumerate_tautomers(mol, max_tautomers=config.max_tautomers))

    # Protonation states — takes SMILES string directly
    if config.protonation:
        _add(
            _enumerate_protonation(
                original_can,
                ph_range=config.ph_range,
                max_protomers=config.max_protomers,
            )
        )

    # Neutralization
    if config.neutralization:
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
