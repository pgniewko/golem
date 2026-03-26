"""Configurable 2D and optional 3D descriptor target generation."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

from golem.config import ConformerConfig, DescriptorConfig, Descriptor3DConfig
from golem.conformers import generate_conformer_ensemble
from golem.descriptors import compute_mordred_descriptors

logger = logging.getLogger(__name__)

_BOLTZMANN_KT_KCAL = 0.593
_DEFAULT_3D_FAMILIES = ("rdkit3d", "usrcat", "electroshape")
DEFAULT_3D_PACK_ID = "rdkit3d+usrcat+electroshape:v1"


def _build_3d_calculators(config: Descriptor3DConfig) -> Dict[str, object]:
    """Build the fixed v1 3D calculator pack."""
    try:
        from molfeat.calc.descriptors import RDKitDescriptors3D
        from molfeat.calc.shape import ElectroShapeDescriptors, USRDescriptors
    except ImportError as exc:
        raise RuntimeError(
            "3D descriptor targets require molfeat to be installed."
        ) from exc

    ignore_descrs = [] if config.rdkit3d_include_getaway else ["CalcGETAWAY"]
    return {
        "rdkit3d": RDKitDescriptors3D(ignore_descrs=ignore_descrs),
        "usrcat": USRDescriptors(method="USRCAT"),
        "electroshape": ElectroShapeDescriptors(
            charge_model=config.electroshape_charge_model
        ),
    }


def _calculator_columns(calculator: object) -> List[str]:
    """Return stable calculator column names."""
    columns = getattr(calculator, "columns", None)
    if callable(columns):
        columns = columns()
    if columns is None:
        columns = getattr(calculator, "_columns", None)
    if columns is None:
        raise RuntimeError(f"Could not determine columns for calculator {calculator!r}")
    return list(columns)


def _aggregate_boltzmann_mean(
    descriptor_values: np.ndarray,
    validity_mask: np.ndarray,
    relative_energies_kcal: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate per-conformer descriptors with Boltzmann weights."""
    weights = np.exp(-relative_energies_kcal / _BOLTZMANN_KT_KCAL)
    weighted_mask = validity_mask * weights[:, None]
    weighted_sum = np.where(validity_mask, descriptor_values * weights[:, None], 0.0).sum(axis=0)
    weight_sum = weighted_mask.sum(axis=0)
    aggregated = np.divide(
        weighted_sum,
        weight_sum,
        out=np.zeros(descriptor_values.shape[1], dtype=np.float64),
        where=weight_sum > 0,
    )
    return aggregated.astype(np.float32), weight_sum > 0


def _aggregate_family_for_molecule(
    family: str,
    calculator: object,
    mol,
    conformer_ids: List[int],
    relative_energies_kcal: np.ndarray,
    aggregation: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate one 3D descriptor family over a retained conformer ensemble."""
    columns = _calculator_columns(calculator)
    width = len(columns)

    if aggregation != "boltz_mean":
        raise ValueError(f"Unsupported 3D aggregation mode: {aggregation!r}")

    descriptor_rows = []
    validity_rows = []
    for conf_id in conformer_ids:
        values = np.asarray(calculator(mol, conformer_id=conf_id), dtype=np.float64)
        if values.shape != (width,):
            raise RuntimeError(
                f"{family} returned shape {values.shape}, expected {(width,)}"
            )
        mask = np.isfinite(values)
        descriptor_rows.append(np.where(mask, values, 0.0))
        validity_rows.append(mask)

    if not descriptor_rows:
        return np.zeros(width, dtype=np.float32), np.zeros(width, dtype=np.bool_)

    return _aggregate_boltzmann_mean(
        np.vstack(descriptor_rows),
        np.vstack(validity_rows),
        relative_energies_kcal,
    )


def compute_aggregated_3d_descriptors(
    smiles_list: List[str],
    three_d: Descriptor3DConfig,
    conformers: ConformerConfig,
    *,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Compute the fixed v1 3D pack for a list of SMILES."""
    calculators = _build_3d_calculators(three_d)
    ordered_families = [
        (family, calculators[family]) for family in _DEFAULT_3D_FAMILIES
    ]

    family_names = [
        [f"{family}:{column}" for column in _calculator_columns(calculator)]
        for family, calculator in ordered_families
    ]
    descriptor_names = [
        name for family_columns in family_names for name in family_columns
    ]
    total_width = len(descriptor_names)

    values = np.zeros((len(smiles_list), total_width), dtype=np.float32)
    validity_mask = np.zeros((len(smiles_list), total_width), dtype=np.bool_)

    for row_idx, smiles in enumerate(smiles_list):
        ensemble = generate_conformer_ensemble(smiles, conformers, seed=seed)
        if ensemble is None:
            continue

        offset = 0
        for family, calculator in ordered_families:
            width = len(_calculator_columns(calculator))
            try:
                family_values, family_mask = _aggregate_family_for_molecule(
                    family,
                    calculator,
                    ensemble.mol,
                    ensemble.conformer_ids,
                    ensemble.relative_energies_kcal,
                    three_d.aggregation,
                )
            except Exception:
                logger.debug(
                    "3D family %s failed for %s", family, smiles, exc_info=True
                )
                family_values = np.zeros(width, dtype=np.float32)
                family_mask = np.zeros(width, dtype=np.bool_)

            values[row_idx, offset : offset + width] = family_values
            validity_mask[row_idx, offset : offset + width] = family_mask
            offset += width

    logger.info(
        "3D descriptors: %d molecules × %d descriptors (%.1f%% valid entries)",
        values.shape[0],
        values.shape[1],
        validity_mask.mean() * 100 if validity_mask.size else 0.0,
    )
    return values, validity_mask, descriptor_names


def compute_descriptor_targets(
    smiles_list: List[str],
    descriptors: DescriptorConfig,
    conformers: ConformerConfig,
    *,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Compute the configured descriptor-target matrix."""
    if not descriptors.include_2d_targets and not descriptors.use_3d_targets:
        raise ValueError("At least one descriptor target family must be enabled.")

    value_blocks: list[np.ndarray] = []
    mask_blocks: list[np.ndarray] = []
    name_blocks: list[list[str]] = []

    if descriptors.include_2d_targets:
        values_2d, mask_2d, names_2d = compute_mordred_descriptors(smiles_list)
        value_blocks.append(values_2d)
        mask_blocks.append(mask_2d)
        name_blocks.append(names_2d)

    if descriptors.use_3d_targets:
        values_3d, mask_3d, names_3d = compute_aggregated_3d_descriptors(
            smiles_list,
            descriptors.three_d,
            conformers,
            seed=seed,
        )
        value_blocks.append(values_3d)
        mask_blocks.append(mask_3d)
        name_blocks.append(names_3d)

    if len(value_blocks) == 1:
        return value_blocks[0], mask_blocks[0], name_blocks[0]

    return (
        np.concatenate(value_blocks, axis=1),
        np.concatenate(mask_blocks, axis=1),
        [name for block in name_blocks for name in block],
    )
