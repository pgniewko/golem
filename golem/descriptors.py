"""Descriptor computation utilities for Golem pretraining.

The pipeline:
1. Compute all Mordred 2D descriptors for a list of SMILES.
2. Optionally compute fixed 3D descriptor targets either from the
   lowest-energy conformer or from a retained Boltzmann-weighted conformer pool.
3. Drop descriptor columns that are invalid everywhere.
4. Build boolean validity masks and store invalid entries as 0.0.

Scaling:
- ``NaNAwareStandardScaler`` fits mean/std on the **training split only**,
  ignoring invalid entries.
- Boltzmann 3D mode fits the 3D slice against the full retained conformer
  distribution weighted by each compound's Boltzmann probabilities.
- ``transform()`` scales and then winsorises to a configurable range
  (default ``[-6, 6]``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence, Tuple

import numpy as np
from rdkit import Chem
from tqdm import tqdm

from golem.config import ConformerConfig, DescriptorConfig, Descriptor3DSettings
from golem.conformers import (
    compute_boltzmann_weights,
    generate_optimized_conformer_pool,
    retain_boltzmann_conformers,
)

logger = logging.getLogger(__name__)

_THREE_D_FAMILIES = ("rdkit3d", "usrcat", "electroshape")


@dataclass
class Conformer3DPool:
    """Cached 3D descriptor rows for one molecule's retained conformer pool."""

    values: np.ndarray
    validity_mask: np.ndarray
    boltzmann_weights: np.ndarray


@dataclass
class PreparedDescriptorTargets:
    """Descriptor targets plus optional cached Boltzmann conformer pools."""

    values: np.ndarray
    validity_mask: np.ndarray
    descriptor_names: List[str]
    num_2d_descriptors: int
    boltzmann_3d_pools: list[Conformer3DPool] | None = None

    @property
    def num_3d_descriptors(self) -> int:
        return len(self.descriptor_names) - self.num_2d_descriptors

    @property
    def three_d_slice(self) -> slice:
        return slice(self.num_2d_descriptors, len(self.descriptor_names))

    @property
    def has_boltzmann_3d(self) -> bool:
        return (
            self.boltzmann_3d_pools is not None
            and self.num_3d_descriptors > 0
        )


def _empty_conformer_3d_pool(width: int) -> Conformer3DPool:
    return Conformer3DPool(
        values=np.zeros((0, width), dtype=np.float32),
        validity_mask=np.zeros((0, width), dtype=np.bool_),
        boltzmann_weights=np.zeros(0, dtype=np.float32),
    )


def _pool_effective_weights(pool: Conformer3DPool) -> np.ndarray:
    weights = pool.boltzmann_weights.astype(np.float64, copy=False)[:, None]
    return weights * pool.validity_mask.astype(np.float64, copy=False)


# ---------------------------------------------------------------------------
# Mordred descriptor computation
# ---------------------------------------------------------------------------

def compute_mordred_descriptors(
    smiles_list: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Compute Mordred 2D descriptors for a list of SMILES."""
    from mordred import Calculator, descriptors as mordred_descriptors

    calc = Calculator(mordred_descriptors, ignore_3D=True)

    mols: List[Chem.Mol | None] = []
    for smi in tqdm(smiles_list, desc="Parsing SMILES", unit="mol"):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            Chem.SanitizeMol(mol)
        mols.append(mol)

    logger.info("Computing Mordred 2D descriptors for %d molecules …", len(mols))
    try:
        df = calc.pandas(mols, quiet=False)
    except (EOFError, OSError, PermissionError):
        logger.warning(
            "Falling back to single-process Mordred computation after "
            "multiprocessing setup failed",
            exc_info=True,
        )
        df = calc.pandas(mols, nproc=1, quiet=False)

    df = df.apply(
        lambda col: col.map(
            lambda v: float(v)
            if isinstance(v, (int, float, np.floating, np.integer))
            else np.nan
        )
    )

    all_nan_cols = df.columns[df.isna().all()]
    if len(all_nan_cols) > 0:
        logger.info("Dropping %d all-NaN descriptor columns", len(all_nan_cols))
        df = df.drop(columns=all_nan_cols)

    descriptor_names = df.columns.tolist()
    raw = df.values.astype(np.float64)
    validity_mask = np.isfinite(raw)
    values = np.where(validity_mask, raw, 0.0).astype(np.float32)

    logger.info(
        "Mordred descriptors: %d molecules × %d descriptors (%.1f%% valid entries)",
        values.shape[0],
        values.shape[1],
        validity_mask.mean() * 100,
    )

    return values, validity_mask.astype(np.bool_), descriptor_names


def _build_3d_calculators(config: Descriptor3DSettings) -> Dict[str, object]:
    """Build the fixed 3D descriptor calculator pack."""
    try:
        from molfeat.calc.descriptors import RDKitDescriptors3D
        from molfeat.calc.shape import ElectroShapeDescriptors, USRDescriptors
    except ImportError as exc:
        raise RuntimeError(
            "3D descriptor targets require molfeat to be installed."
        ) from exc

    ignore_descrs = [] if config.rdkit_include_getaway else ["CalcGETAWAY"]
    return {
        "rdkit3d": RDKitDescriptors3D(ignore_descrs=ignore_descrs),
        "usrcat": USRDescriptors(method="USRCAT"),
        "electroshape": ElectroShapeDescriptors(charge_model="gasteiger"),
    }


def _calculator_columns(calculator: object) -> List[str]:
    columns = getattr(calculator, "columns", None)
    if callable(columns):
        columns = columns()
    if columns is None:
        columns = getattr(calculator, "_columns", None)
    if columns is None:
        raise RuntimeError(f"Could not determine descriptor columns for {calculator!r}")
    return list(columns)


def _compute_3d_descriptor_row(
    mol: Chem.Mol,
    conformer_id: int,
    calculators: Dict[str, object],
    columns_by_family: Dict[str, List[str]],
    *,
    smiles: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute the concatenated 3D descriptor row for one conformer."""
    descriptor_width = sum(len(columns_by_family[family]) for family in _THREE_D_FAMILIES)
    values = np.zeros(descriptor_width, dtype=np.float32)
    validity_mask = np.zeros(descriptor_width, dtype=np.bool_)
    descriptor_failures = 0
    offset = 0
    for family in _THREE_D_FAMILIES:
        width = len(columns_by_family[family])
        try:
            row = np.asarray(
                calculators[family](mol, conformer_id=conformer_id),
                dtype=np.float64,
            )
            if row.shape != (width,):
                raise RuntimeError(
                    f"{family} returned shape {row.shape}, expected {(width,)}"
                )
            family_mask = np.isfinite(row)
            family_values = np.where(family_mask, row, 0.0).astype(np.float32)
        except Exception:
            logger.debug(
                "3D descriptor family %s failed for %s conformer %s",
                family,
                smiles,
                conformer_id,
                exc_info=True,
            )
            family_values = np.zeros(width, dtype=np.float32)
            family_mask = np.zeros(width, dtype=np.bool_)
            descriptor_failures += 1

        values[offset : offset + width] = family_values
        validity_mask[offset : offset + width] = family_mask
        offset += width

    return values, validity_mask, descriptor_failures


def _drop_invalid_3d_columns(
    pools: Sequence[Conformer3DPool],
    descriptor_names: list[str],
) -> tuple[list[Conformer3DPool], list[str]]:
    if not descriptor_names:
        return list(pools), descriptor_names

    global_valid = np.zeros(len(descriptor_names), dtype=np.bool_)
    for pool in pools:
        if pool.validity_mask.size == 0:
            continue
        global_valid |= pool.validity_mask.any(axis=0)

    if global_valid.all():
        return list(pools), descriptor_names

    dropped = int((~global_valid).sum())
    logger.info("Dropping %d all-invalid 3D descriptor columns", dropped)
    kept_names = [
        name
        for name, keep in zip(descriptor_names, global_valid, strict=False)
        if keep
    ]
    trimmed_pools = [
        Conformer3DPool(
            values=pool.values[:, global_valid],
            validity_mask=pool.validity_mask[:, global_valid],
            boltzmann_weights=pool.boltzmann_weights.copy(),
        )
        for pool in pools
    ]
    return trimmed_pools, kept_names


def materialize_boltzmann_mean_targets(
    pools: Sequence[Conformer3DPool],
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-molecule expected 3D targets from retained conformer pools."""
    if not pools:
        empty = np.zeros((0, 0), dtype=np.float32)
        return empty, empty.astype(np.bool_)

    width = pools[0].values.shape[1]
    values = np.zeros((len(pools), width), dtype=np.float32)
    validity_mask = np.zeros((len(pools), width), dtype=np.bool_)

    for row_idx, pool in enumerate(pools):
        if pool.values.shape[0] == 0 or width == 0:
            continue

        effective_weights = _pool_effective_weights(pool)
        weight_sum = effective_weights.sum(axis=0)
        validity_mask[row_idx] = weight_sum > 0.0
        weighted_sum = (effective_weights * pool.values.astype(np.float64)).sum(axis=0)
        values[row_idx] = np.divide(
            weighted_sum,
            weight_sum,
            out=np.zeros(width, dtype=np.float64),
            where=weight_sum > 0.0,
        ).astype(np.float32)

    return values, validity_mask


def compute_boltzmann_weighted_3d_statistics(
    pools: Sequence[Conformer3DPool],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit weighted mean/std for a cached retained-conformer 3D target pool."""
    if not pools:
        return np.zeros(0, dtype=np.float64), np.ones(0, dtype=np.float64)

    width = pools[0].values.shape[1]
    if width == 0:
        return np.zeros(0, dtype=np.float64), np.ones(0, dtype=np.float64)

    weight_sum = np.zeros(width, dtype=np.float64)
    weighted_sum = np.zeros(width, dtype=np.float64)
    weighted_sq_sum = np.zeros(width, dtype=np.float64)

    for pool in pools:
        if pool.values.shape[0] == 0:
            continue
        effective_weights = _pool_effective_weights(pool)
        pool_values = pool.values.astype(np.float64, copy=False)
        weight_sum += effective_weights.sum(axis=0)
        weighted_sum += (effective_weights * pool_values).sum(axis=0)
        weighted_sq_sum += (effective_weights * np.square(pool_values)).sum(axis=0)

    mean = np.divide(
        weighted_sum,
        weight_sum,
        out=np.zeros(width, dtype=np.float64),
        where=weight_sum > 0.0,
    )
    second_moment = np.divide(
        weighted_sq_sum,
        weight_sum,
        out=np.ones(width, dtype=np.float64),
        where=weight_sum > 0.0,
    )
    std = np.sqrt(np.maximum(second_moment - np.square(mean), 0.0))
    return NaNAwareStandardScaler._finalize_stats(
        mean,
        std,
        invalid_mask=weight_sum == 0.0,
    )


def scale_boltzmann_3d_pools(
    pools: Sequence[Conformer3DPool],
    mean: np.ndarray,
    std: np.ndarray,
    winsorize_range: tuple[float, float],
) -> list[Conformer3DPool]:
    """Scale cached retained-conformer 3D descriptor rows once up front."""
    scaled_pools: list[Conformer3DPool] = []
    for pool in pools:
        if pool.values.shape[0] == 0 or pool.values.shape[1] == 0:
            scaled_values = pool.values.copy()
        else:
            scaled_values = NaNAwareStandardScaler.transform_with_stats(
                pool.values,
                mean,
                std,
                winsorize_range=winsorize_range,
            )
        scaled_pools.append(
            Conformer3DPool(
                values=scaled_values,
                validity_mask=pool.validity_mask.copy(),
                boltzmann_weights=pool.boltzmann_weights.copy(),
            )
        )
    return scaled_pools


def _compute_3d_targets(
    smiles_list: List[str],
    three_d_settings: Descriptor3DSettings,
    conformers: ConformerConfig,
    *,
    seed: int,
    target_mode: Literal["lowest_energy", "boltzmann"],
) -> tuple[np.ndarray, np.ndarray, List[str], list[Conformer3DPool], int, int, list[int]]:
    calculators = _build_3d_calculators(three_d_settings)
    columns_by_family = {
        family: _calculator_columns(calculators[family]) for family in _THREE_D_FAMILIES
    }
    descriptor_names = [
        f"{family}:{column}"
        for family in _THREE_D_FAMILIES
        for column in columns_by_family[family]
    ]
    pools: list[Conformer3DPool] = []
    conformer_failures = 0
    descriptor_failures = 0
    retained_counts: list[int] = []
    progress_label = "3D descriptor pools" if target_mode == "boltzmann" else "3D descriptors"

    for smiles in tqdm(smiles_list, desc=progress_label, unit="mol"):
        optimized_pool = generate_optimized_conformer_pool(smiles, conformers, seed=seed)
        if optimized_pool is None:
            conformer_failures += 1
            pools.append(_empty_conformer_3d_pool(len(descriptor_names)))
            continue

        selected_conformers = (
            retain_boltzmann_conformers(optimized_pool, conformers).conformers
            if target_mode == "boltzmann"
            else optimized_pool.conformers[:1]
        )
        if not selected_conformers:
            pools.append(_empty_conformer_3d_pool(len(descriptor_names)))
            continue

        conformer_rows: list[np.ndarray] = []
        conformer_masks: list[np.ndarray] = []
        delta_energies: list[float] = []
        for conformer in selected_conformers:
            row, row_mask, row_failures = _compute_3d_descriptor_row(
                optimized_pool.mol,
                conformer.conformer_id,
                calculators,
                columns_by_family,
                smiles=smiles,
            )
            descriptor_failures += row_failures
            if not row_mask.any():
                continue
            conformer_rows.append(row)
            conformer_masks.append(row_mask)
            delta_energies.append(conformer.delta_energy)

        if not conformer_rows:
            pools.append(_empty_conformer_3d_pool(len(descriptor_names)))
            continue

        if target_mode == "boltzmann":
            weights = compute_boltzmann_weights(
                np.asarray(delta_energies, dtype=np.float64)
            ).astype(np.float32)
        else:
            weights = np.ones(len(conformer_rows), dtype=np.float32)
        pool_values = np.stack(conformer_rows).astype(np.float32)
        pool_masks = np.stack(conformer_masks).astype(np.bool_)
        pools.append(
            Conformer3DPool(
                values=pool_values,
                validity_mask=pool_masks,
                boltzmann_weights=weights,
            )
        )
        if target_mode == "boltzmann":
            retained_counts.append(pool_values.shape[0])

    pools, descriptor_names = _drop_invalid_3d_columns(pools, descriptor_names)
    values, validity_mask = materialize_boltzmann_mean_targets(pools)
    return (
        values,
        validity_mask,
        descriptor_names,
        pools,
        conformer_failures,
        descriptor_failures,
        retained_counts,
    )


def compute_3d_descriptors(
    smiles_list: List[str],
    three_d_settings: Descriptor3DSettings,
    conformers: ConformerConfig,
    *,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Compute the fixed 3D descriptor pack on the lowest-energy conformer."""
    (
        values,
        validity_mask,
        descriptor_names,
        _,
        conformer_failures,
        descriptor_failures,
        _,
    ) = _compute_3d_targets(
        smiles_list,
        three_d_settings,
        conformers,
        seed=seed,
        target_mode="lowest_energy",
    )

    logger.info(
        "3D descriptors: %d molecules × %d descriptors (%.1f%% valid entries)",
        values.shape[0],
        values.shape[1],
        validity_mask.mean() * 100 if validity_mask.size else 0.0,
    )
    if conformer_failures or descriptor_failures:
        logger.info(
            "3D target generation kept all molecules in-place "
            "(%d conformer failures, %d descriptor-family failures; invalid entries were masked)",
            conformer_failures,
            descriptor_failures,
        )
    return values, validity_mask, descriptor_names


def compute_boltzmann_3d_descriptors(
    smiles_list: List[str],
    three_d_settings: Descriptor3DSettings,
    conformers: ConformerConfig,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, List[str], list[Conformer3DPool]]:
    """Build retained Boltzmann conformer pools and deterministic mean targets."""
    (
        values,
        validity_mask,
        descriptor_names,
        pools,
        conformer_failures,
        descriptor_failures,
        retained_counts,
    ) = _compute_3d_targets(
        smiles_list,
        three_d_settings,
        conformers,
        seed=seed,
        target_mode="boltzmann",
    )

    logger.info(
        "Boltzmann 3D descriptor means: %d molecules × %d descriptors (%.1f%% valid entries)",
        values.shape[0],
        values.shape[1],
        validity_mask.mean() * 100 if validity_mask.size else 0.0,
    )
    if retained_counts:
        logger.info(
            "Retained conformers per molecule after delta-energy filtering: mean=%.2f min=%d max=%d",
            float(np.mean(retained_counts)),
            int(np.min(retained_counts)),
            int(np.max(retained_counts)),
        )
    if conformer_failures or descriptor_failures:
        logger.info(
            "Boltzmann 3D target generation kept all molecules in-place "
            "(%d conformer generation failures, %d descriptor-family failures; invalid entries were masked)",
            conformer_failures,
            descriptor_failures,
        )
    return values, validity_mask, descriptor_names, pools


def prepare_descriptor_targets(
    smiles_list: List[str],
    descriptors: DescriptorConfig,
    conformers: ConformerConfig,
    *,
    seed: int,
) -> PreparedDescriptorTargets:
    """Compute the configured descriptor targets and optional cached 3D pools."""
    if not descriptors.include_2d_targets and not descriptors.include_3d_targets:
        raise ValueError("At least one descriptor target family must be enabled.")

    blocks: list[tuple[np.ndarray, np.ndarray, list[str]]] = []
    num_2d_descriptors = 0
    boltzmann_pools: list[Conformer3DPool] | None = None

    if descriptors.include_2d_targets:
        two_d_values, two_d_mask, two_d_names = compute_mordred_descriptors(smiles_list)
        num_2d_descriptors = len(two_d_names)
        blocks.append((two_d_values, two_d_mask, two_d_names))

    if descriptors.include_3d_targets:
        if descriptors.three_d_settings.target_mode == "boltzmann":
            three_d_values, three_d_mask, three_d_names, boltzmann_pools = (
                compute_boltzmann_3d_descriptors(
                    smiles_list,
                    descriptors.three_d_settings,
                    conformers,
                    seed=seed,
                )
            )
        else:
            three_d_values, three_d_mask, three_d_names = compute_3d_descriptors(
                smiles_list,
                descriptors.three_d_settings,
                conformers,
                seed=seed,
            )
        blocks.append((three_d_values, three_d_mask, three_d_names))

    if len(blocks) == 1:
        values, masks, names = blocks[0]
    else:
        values = np.concatenate([block[0] for block in blocks], axis=1)
        masks = np.concatenate([block[1] for block in blocks], axis=1)
        names = [name for _, _, block_names in blocks for name in block_names]

    if values.shape[1] == 0:
        raise ValueError(
            "No valid descriptor targets remained after dropping all-invalid columns."
        )
    return PreparedDescriptorTargets(
        values=values,
        validity_mask=masks,
        descriptor_names=names,
        num_2d_descriptors=num_2d_descriptors,
        boltzmann_3d_pools=boltzmann_pools,
    )


def compute_descriptor_targets(
    smiles_list: List[str],
    descriptors: DescriptorConfig,
    conformers: ConformerConfig,
    *,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Backwards-compatible wrapper returning only the static target matrix."""
    prepared = prepare_descriptor_targets(
        smiles_list,
        descriptors,
        conformers,
        seed=seed,
    )
    return prepared.values, prepared.validity_mask, prepared.descriptor_names


# ---------------------------------------------------------------------------
# NaN-aware standard scaler
# ---------------------------------------------------------------------------

class NaNAwareStandardScaler:
    """Per-feature zero-mean / unit-variance scaler that ignores invalid positions."""

    def __init__(self, winsorize_range: Tuple[float, float] = (-6.0, 6.0)) -> None:
        self.winsorize_range = winsorize_range
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    @staticmethod
    def _finalize_stats(
        mean: np.ndarray,
        std: np.ndarray,
        *,
        invalid_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if invalid_mask.any():
            logger.info(
                "Setting mean=0.0, std=1.0 for %d all-invalid descriptors in train",
                int(invalid_mask.sum()),
            )
            mean[invalid_mask] = 0.0
            std[invalid_mask] = 1.0

        zero_std = std < 1e-12
        if zero_std.any():
            logger.info(
                "Setting std=1.0 for %d constant/near-constant descriptors",
                int(zero_std.sum()),
            )
            std[zero_std] = 1.0
        return mean, std

    def fit(
        self,
        X: np.ndarray,
        validity_mask: np.ndarray,
    ) -> "NaNAwareStandardScaler":
        """Compute per-feature mean and std from valid entries only."""
        X64 = X.astype(np.float64, copy=False)
        valid = validity_mask.astype(bool, copy=False)
        valid_count = valid.sum(axis=0)
        masked_values = np.where(valid, X64, 0.0)

        self.mean_ = np.divide(
            masked_values.sum(axis=0),
            valid_count,
            out=np.zeros(X64.shape[1], dtype=np.float64),
            where=valid_count > 0,
        )
        centered = np.where(valid, X64 - self.mean_, 0.0)
        self.std_ = np.sqrt(
            np.divide(
                np.square(centered).sum(axis=0),
                valid_count,
                out=np.ones(X64.shape[1], dtype=np.float64),
                where=valid_count > 0,
            )
        )
        self.mean_, self.std_ = self._finalize_stats(
            self.mean_,
            self.std_,
            invalid_mask=valid_count == 0,
        )
        return self

    @staticmethod
    def transform_with_stats(
        X: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        *,
        winsorize_range: Tuple[float, float],
    ) -> np.ndarray:
        scaled = (X.astype(np.float64) - mean) / std
        lo, hi = winsorize_range
        scaled = np.clip(scaled, lo, hi)
        return scaled.astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale *X* and winsorise."""
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler has not been fit yet")
        return self.transform_with_stats(
            X,
            self.mean_,
            self.std_,
            winsorize_range=self.winsorize_range,
        )

    def state_dict(self) -> Dict[str, object]:
        """Serialise scaler parameters to a plain dict (for checkpoint)."""
        return {
            "mean": self.mean_.tolist() if self.mean_ is not None else None,
            "std": self.std_.tolist() if self.std_ is not None else None,
            "winsorize_range": list(self.winsorize_range),
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, object]) -> "NaNAwareStandardScaler":
        """Reconstruct a scaler from a serialised ``state_dict``."""
        scaler = cls(winsorize_range=tuple(d["winsorize_range"]))
        if d["mean"] is not None:
            scaler.mean_ = np.array(d["mean"], dtype=np.float64)
        if d["std"] is not None:
            scaler.std_ = np.array(d["std"], dtype=np.float64)
        return scaler
