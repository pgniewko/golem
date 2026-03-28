"""Mordred 2D descriptor computation and NaN-aware standardisation.

The pipeline:
1. Compute all Mordred 2D descriptors for a list of SMILES.
2. Drop descriptors that are all-NaN or non-numeric.
3. Build a boolean *validity_mask* (True = valid, False = was-NaN).
4. Replace NaN with 0.0 in the values array (NaN positions tracked by mask).

Scaling:
- ``NaNAwareStandardScaler`` fits mean/std on the **training split only**,
  using ``np.nanmean`` / ``np.nanstd`` so that NaN positions are ignored.
- ``transform()`` scales and then winsorises to a configurable range
  (default ``[-6, 6]``).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
from rdkit import Chem
from tqdm import tqdm

from golem.config import ConformerConfig, DescriptorConfig, Descriptor3DSettings
from golem.conformers import generate_conformer_ensemble

logger = logging.getLogger(__name__)

_BOLTZMANN_KT_KCAL = 0.593
_THREE_D_FAMILIES = ("rdkit3d", "usrcat", "electroshape")


# ---------------------------------------------------------------------------
# Mordred descriptor computation
# ---------------------------------------------------------------------------

def compute_mordred_descriptors(
    smiles_list: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Compute Mordred 2D descriptors for a list of SMILES.

    Args:
        smiles_list: List of SMILES strings.

    Returns:
        values:        ``np.float32`` array ``[N, D]`` — NaN stored as 0.0.
        validity_mask: ``np.bool_``  array ``[N, D]`` — True where original
                       value was numeric and finite.
        descriptor_names: list of ``D`` descriptor name strings.
    """
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

    # Force numeric — non-numeric entries become NaN
    df = df.apply(lambda col: col.map(lambda v: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else np.nan))

    # Drop descriptors that are all-NaN
    all_nan_cols = df.columns[df.isna().all()]
    if len(all_nan_cols) > 0:
        logger.info("Dropping %d all-NaN descriptor columns", len(all_nan_cols))
        df = df.drop(columns=all_nan_cols)

    descriptor_names = df.columns.tolist()
    raw = df.values.astype(np.float64)

    # Build validity mask BEFORE filling NaN
    validity_mask = np.isfinite(raw)

    # Replace NaN/inf with 0.0 for storage
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


def _aggregate_boltzmann_mean(
    descriptor_values: np.ndarray,
    validity_mask: np.ndarray,
    relative_energies_kcal: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    weights = np.exp(-relative_energies_kcal / _BOLTZMANN_KT_KCAL)
    weighted_mask = validity_mask * weights[:, None]
    weight_sum = weighted_mask.sum(axis=0)
    aggregated = np.divide(
        np.where(validity_mask, descriptor_values * weights[:, None], 0.0).sum(axis=0),
        weight_sum,
        out=np.zeros(descriptor_values.shape[1], dtype=np.float64),
        where=weight_sum > 0,
    )
    return aggregated.astype(np.float32), weight_sum > 0


def compute_aggregated_3d_descriptors(
    smiles_list: List[str],
    three_d_settings: Descriptor3DSettings,
    conformers: ConformerConfig,
    *,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """Compute and aggregate the fixed 3D descriptor pack."""
    calculators = _build_3d_calculators(three_d_settings)
    columns_by_family = {
        family: _calculator_columns(calculators[family]) for family in _THREE_D_FAMILIES
    }
    descriptor_names = [
        f"{family}:{column}"
        for family in _THREE_D_FAMILIES
        for column in columns_by_family[family]
    ]
    total_width = len(descriptor_names)
    values = np.zeros((len(smiles_list), total_width), dtype=np.float32)
    validity_mask = np.zeros((len(smiles_list), total_width), dtype=np.bool_)
    keep_mask = np.ones(len(smiles_list), dtype=np.bool_)
    dropped_timeout = 0
    dropped_conformer_failures = 0
    dropped_invalid_descriptors = 0

    for row_idx, smiles in enumerate(smiles_list):
        ensemble, failure_reason = generate_conformer_ensemble(
            smiles,
            conformers,
            seed=seed,
        )
        if ensemble is None:
            keep_mask[row_idx] = False
            if failure_reason == "timeout":
                dropped_timeout += 1
            else:
                dropped_conformer_failures += 1
            continue

        offset = 0
        row_is_valid = True
        for family in _THREE_D_FAMILIES:
            width = len(columns_by_family[family])
            try:
                rows = []
                masks = []
                for conformer_id in ensemble.conformer_ids:
                    row = np.asarray(
                        calculators[family](ensemble.mol, conformer_id=conformer_id),
                        dtype=np.float64,
                    )
                    if row.shape != (width,):
                        raise RuntimeError(
                            f"{family} returned shape {row.shape}, expected {(width,)}"
                        )
                    mask = np.isfinite(row)
                    rows.append(np.where(mask, row, 0.0))
                    masks.append(mask)
                family_values, family_mask = _aggregate_boltzmann_mean(
                    np.vstack(rows),
                    np.vstack(masks),
                    ensemble.relative_energies_kcal,
                )
            except Exception:
                logger.debug("3D descriptor family %s failed for %s", family, smiles, exc_info=True)
                family_values = np.zeros(width, dtype=np.float32)
                family_mask = np.zeros(width, dtype=np.bool_)

            values[row_idx, offset : offset + width] = family_values
            validity_mask[row_idx, offset : offset + width] = family_mask
            offset += width
            row_is_valid &= bool(family_mask.all())

        if not row_is_valid:
            keep_mask[row_idx] = False
            dropped_invalid_descriptors += 1

    logger.info(
        "3D descriptors: %d molecules × %d descriptors (%.1f%% valid entries)",
        values.shape[0],
        values.shape[1],
        validity_mask.mean() * 100 if validity_mask.size else 0.0,
    )
    if dropped_timeout or dropped_conformer_failures or dropped_invalid_descriptors:
        logger.info(
            "Dropped %d molecules during 3D target generation "
            "(%d timeouts, %d other conformer failures, %d invalid descriptor ensembles)",
            int((~keep_mask).sum()),
            dropped_timeout,
            dropped_conformer_failures,
            dropped_invalid_descriptors,
        )
    return values, validity_mask, descriptor_names, keep_mask


def compute_descriptor_targets(
    smiles_list: List[str],
    descriptors: DescriptorConfig,
    conformers: ConformerConfig,
    *,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """Compute the configured 2D/3D descriptor target matrix."""
    if not descriptors.include_2d_targets and not descriptors.include_3d_targets:
        raise ValueError("At least one descriptor target family must be enabled.")

    values_blocks = []
    mask_blocks = []
    name_blocks = []
    keep_mask = np.ones(len(smiles_list), dtype=np.bool_)

    if descriptors.include_2d_targets:
        values_2d, mask_2d, names_2d = compute_mordred_descriptors(smiles_list)
        values_blocks.append(values_2d)
        mask_blocks.append(mask_2d)
        name_blocks.append(names_2d)

    if descriptors.include_3d_targets:
        values_3d, mask_3d, names_3d, keep_3d = compute_aggregated_3d_descriptors(
            smiles_list,
            descriptors.three_d_settings,
            conformers,
            seed=seed,
        )
        values_blocks.append(values_3d)
        mask_blocks.append(mask_3d)
        name_blocks.append(names_3d)
        keep_mask &= keep_3d

    if len(values_blocks) == 1:
        return values_blocks[0], mask_blocks[0], name_blocks[0], keep_mask

    return (
        np.concatenate(values_blocks, axis=1),
        np.concatenate(mask_blocks, axis=1),
        [name for block in name_blocks for name in block],
        keep_mask,
    )


# ---------------------------------------------------------------------------
# NaN-aware standard scaler
# ---------------------------------------------------------------------------

class NaNAwareStandardScaler:
    """Per-feature zero-mean / unit-variance scaler that ignores NaN/invalid
    positions when computing statistics.

    Typical workflow::

        scaler = NaNAwareStandardScaler(winsorize_range=(-6.0, 6.0))
        scaler.fit(X_train, validity_mask_train)
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled   = scaler.transform(X_val)

    The scaler is fully serialisable via ``state_dict()`` / ``from_state_dict()``.
    """

    def __init__(self, winsorize_range: Tuple[float, float] = (-6.0, 6.0)) -> None:
        self.winsorize_range = winsorize_range
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        validity_mask: np.ndarray,
    ) -> "NaNAwareStandardScaler":
        """Compute per-feature mean and std from **valid** entries only.

        Args:
            X: Values array ``[N, D]`` (NaN positions may contain 0.0).
            validity_mask: Boolean array ``[N, D]`` — True = valid.

        Returns:
            self (for chaining).
        """
        # Work with float64 for numerical stability
        X_masked = np.where(validity_mask, X.astype(np.float64), np.nan)

        self.mean_ = np.nanmean(X_masked, axis=0)
        self.std_ = np.nanstd(X_masked, axis=0)

        # Guard against all-NaN columns (nanmean/nanstd return NaN)
        nan_mean = np.isnan(self.mean_)
        if nan_mean.any():
            logger.info(
                "Setting mean=0.0, std=1.0 for %d all-NaN descriptors in train",
                nan_mean.sum(),
            )
            self.mean_[nan_mean] = 0.0
            self.std_[nan_mean] = 1.0

        # Guard against zero-std (constant columns): replace with 1.0
        zero_std = self.std_ < 1e-12
        if zero_std.any():
            logger.info(
                "Setting std=1.0 for %d constant/near-constant descriptors",
                zero_std.sum(),
            )
            self.std_[zero_std] = 1.0

        return self

    # ------------------------------------------------------------------
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale *X* and winsorise.

        Invalid positions (which should be 0.0 in *X*) will be
        transformed to ``(0 - mean) / std`` and then clipped.  This is
        fine because the validity mask prevents these from entering the loss.

        Args:
            X: Values array ``[N, D]``.

        Returns:
            Scaled ``np.float32`` array ``[N, D]``.
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler has not been fit yet")

        scaled = (X.astype(np.float64) - self.mean_) / self.std_
        lo, hi = self.winsorize_range
        scaled = np.clip(scaled, lo, hi)
        return scaled.astype(np.float32)

    # ------------------------------------------------------------------
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
