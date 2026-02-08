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

logger = logging.getLogger(__name__)


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
    df = calc.pandas(mols, quiet=False)

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
