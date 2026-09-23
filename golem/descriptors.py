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
from golem.conformers import generate_lowest_energy_conformer

logger = logging.getLogger(__name__)

_THREE_D_FAMILIES = ("rdkit3d", "usrcat", "electroshape")


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

    logger.info(f"Computing Mordred 2D descriptors for {len(mols)} molecules ...")
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
        logger.info(f"Dropping {len(all_nan_cols)} all-NaN descriptor columns")
        df = df.drop(columns=all_nan_cols)

    descriptor_names = df.columns.tolist()
    raw = df.values.astype(np.float64)

    # Build validity mask BEFORE filling NaN
    validity_mask = np.isfinite(raw)

    # Replace NaN/inf with 0.0 for storage
    values = np.where(validity_mask, raw, 0.0).astype(np.float32)

    logger.info(
        f"Mordred descriptors: {values.shape[0]} molecules x {values.shape[1]} descriptors ({validity_mask.mean() * 100:.1f}% valid entries)",
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


def compute_3d_descriptors(
    smiles_list: List[str],
    three_d_settings: Descriptor3DSettings,
    conformers: ConformerConfig,
    *,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Compute the fixed 3D descriptor pack on the lowest-energy conformer."""
    calculators = _build_3d_calculators(three_d_settings)
    columns_by_family = {
        family: _calculator_columns(calculators[family]) for family in _THREE_D_FAMILIES
    }
    descriptor_names = [
        f"{family}:{column}"
        for family in _THREE_D_FAMILIES
        for column in columns_by_family[family]
    ]
    values = np.zeros((len(smiles_list), len(descriptor_names)), dtype=np.float32)
    validity_mask = np.zeros((len(smiles_list), len(descriptor_names)), dtype=np.bool_)
    conformer_failures = 0
    descriptor_failures = 0

    for row_idx, smiles in enumerate(
        tqdm(smiles_list, desc="3D descriptors", unit="mol")
    ):
        conformer = generate_lowest_energy_conformer(
            smiles,
            conformers,
            seed=seed,
        )
        if conformer is None:
            conformer_failures += 1
            continue

        offset = 0
        for family in _THREE_D_FAMILIES:
            width = len(columns_by_family[family])
            try:
                row = np.asarray(
                    calculators[family](conformer.mol, conformer_id=conformer.conformer_id),
                    dtype=np.float64,
                )
                if row.shape != (width,):
                    raise RuntimeError(
                        f"{family} returned shape {row.shape}, expected {(width,)}"
                    )
                family_mask = np.isfinite(row)
                family_values = np.where(family_mask, row, 0.0).astype(np.float32)
            except Exception:
                logger.debug(f"3D descriptor family {family} failed for {smiles}", exc_info=True)
                family_values = np.zeros(width, dtype=np.float32)
                family_mask = np.zeros(width, dtype=np.bool_)
                descriptor_failures += 1

            values[row_idx, offset : offset + width] = family_values
            validity_mask[row_idx, offset : offset + width] = family_mask
            offset += width

    all_invalid = ~validity_mask.any(axis=0)
    if all_invalid.any():
        logger.info(f"Dropping {int(all_invalid.sum())} all-invalid 3D descriptor columns")
        keep_columns = ~all_invalid
        values = values[:, keep_columns]
        validity_mask = validity_mask[:, keep_columns]
        descriptor_names = [
            name
            for name, keep in zip(descriptor_names, keep_columns, strict=False)
            if keep
        ]

    logger.info(
        f"3D descriptors: {values.shape[0]} molecules x {values.shape[1]} "
        f"descriptors ({validity_mask.mean() * 100 if validity_mask.size else 0.0}% valid entries)"
    )
    if conformer_failures or descriptor_failures:
        logger.info(
            f"3D target generation kept all molecules in-place "
            f"({conformer_failures} conformer failures, "
            f"{descriptor_failures} descriptor-family failures; "
            "invalid entries were masked)"
        )
    return values, validity_mask, descriptor_names


def prepare_descriptor_targets(
    smiles_list: List[str],
    descriptors: DescriptorConfig,
    conformers: ConformerConfig,
    *,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
    """Compute the configured 2D/3D descriptor targets plus the 2D width."""
    if not descriptors.include_2d_targets and not descriptors.include_3d_targets:
        raise ValueError("At least one descriptor target family must be enabled.")

    blocks = []
    num_2d_descriptors = 0
    if descriptors.include_2d_targets:
        two_d_values, two_d_mask, two_d_names = compute_mordred_descriptors(smiles_list)
        num_2d_descriptors = len(two_d_names)
        blocks.append((two_d_values, two_d_mask, two_d_names))
    if descriptors.include_3d_targets:
        blocks.append(
            compute_3d_descriptors(
                smiles_list,
                descriptors.three_d_settings,
                conformers,
                seed=seed,
            )
        )

    if len(blocks) == 1:
        values, masks, names = blocks[0]
    else:
        values = np.concatenate([block[0] for block in blocks], axis=1)
        masks = np.concatenate([block[1] for block in blocks], axis=1)
        names = [name for _, _, block_names in blocks for name in block_names]
        assert values.shape[0] == masks.shape[0] and values.shape[1] == masks.shape[1]
        assert len(names) == values.shape[1]

    if values.shape[1] == 0:
        raise ValueError(
            "No valid descriptor targets remained after dropping all-invalid columns."
        )
    return values, masks, names, num_2d_descriptors


class NaNAwareStandardScaler:
    """Per-feature zero-mean / unit-variance scaler that ignores NaN/invalid
    positions when computing statistics.

    Usage:

        >>> scaler = NaNAwareStandardScaler(winsorize_range=(-6.0, 6.0))
        >>> scaler.fit(X_train, validity_mask_train)
        >>> X_train_scaled = scaler.transform(X_train)
        >>> X_val_scaled   = scaler.transform(X_val)

    The scaler is serialisable via ``state_dict()`` / ``from_state_dict()``.
    """

    def __init__(self, winsorize_range: Tuple[float, float] = (-6.0, 6.0)) -> None:
        self.winsorize_range = winsorize_range
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        validity_mask: np.ndarray,
    ) -> "NaNAwareStandardScaler":
        """Compute per-feature mean and std from **valid** entries only."""
        valid = validity_mask.astype(bool, copy=False)
        Xm = np.ma.masked_array(X.astype(np.float64, copy=False), mask=~valid)

        self.mean_ = Xm.mean(axis=0).filled(0.0)
        self.std_ = Xm.std(axis=0).filled(1.0)

        all_invalid = ~valid.any(axis=0)
        if all_invalid.any():
            logger.info(f"Setting mean=0.0, std=1.0 for {all_invalid.sum()} all-NaN descriptors in train.")
        
        zero_std = self.std_ < 1e-12
        if zero_std.any():
            logger.info(f"Setting std=1.0 for {zero_std.sum()} constant/near-constant descriptors")
            self.std_[zero_std] = 1.0

        return self

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
