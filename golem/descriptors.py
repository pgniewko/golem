"""Mordred 2D descriptor computation and NaN-aware standardisation."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple
from importlib.metadata import version, packages_distributions, PackageNotFoundError

import numpy as np
from rdkit import Chem
from tqdm import tqdm
from scipy.special import ndtr, ndtri
from scipy.stats import skew

from golem.config import ConformerConfig, DescriptorConfig, Descriptor3DSettings
from golem.conformers import generate_lowest_energy_conformer

logger = logging.getLogger(__name__)

_THREE_D_FAMILIES = ("rdkit3d", "usrcat", "electroshape")
_VERSION_TRACKED_PACKAGES = ("rdkit", "mordred", "molfeat")
_MIN_VALID = 30
_N_QUANTILES = 101
_P_EPS = 1e-7


def compute_mordred_descriptors(
    smiles_list: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Compute Mordred 2D descriptors for a list of SMILES.

    Args:
        smiles_list: List of SMILES strings.

    Returns:
        values:        ``np.float64`` array ``[N, D]`` — NaN stored as 0.0.
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

    all_nan_cols = df.columns[df.isna().all()]
    if len(all_nan_cols) > 0:
        logger.info(f"Found {len(all_nan_cols)} all-NaN 2D descriptor columns")

    descriptor_names = df.columns.tolist()
    raw = df.values.astype(np.float64)

    # Build validity mask BEFORE filling NaN
    validity_mask = np.isfinite(raw)

    # Replace NaN/inf with 0.0 for storage
    values = np.where(validity_mask, raw, 0.0).astype(np.float64)

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
    values = np.zeros((len(smiles_list), len(descriptor_names)), dtype=np.float64)
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
                family_values = np.where(family_mask, row, 0.0).astype(np.float64)
            except Exception:
                logger.debug(f"3D descriptor family {family} failed for {smiles}", exc_info=True)
                family_values = np.zeros(width, dtype=np.float64)
                family_mask = np.zeros(width, dtype=np.bool_)
                descriptor_failures += 1

            values[row_idx, offset : offset + width] = family_values
            validity_mask[row_idx, offset : offset + width] = family_mask
            offset += width

    all_nan_cols = ~validity_mask.any(axis=0)
    if all_nan_cols.any():
        logger.info(f"Found {int(all_nan_cols.sum())} all-invalid 3D descriptor columns")

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


def _get_3rd_party_versions() -> Dict[str, str | None]:
    """Get the versions of the 3rd party libraries."""

    pkg_to_dist = packages_distributions()
    out: Dict[str, str | None] = {}

    for name in _VERSION_TRACKED_PACKAGES:
        dist = pkg_to_dist.get(name) or [name]  # list fallback; 'mordred' -> 'mordredcommunity'
        try:
            out[name] = version(dist[0])        # installed -> must have a version
        except PackageNotFoundError:
            out[name] = None                    # not installed -> but if required it'd already raise
    return out


class DescriptorTransformer:
    """Per-column transform (none/log/quantile) followed by z-score and winsorisation.

    The transform for each column is chosen in ``fit`` from its valid train entries.
    Statistics ignore invalid positions.

    Usage:

        >>> scaler = DescriptorTransformer(winsorize_range=(-6.0, 6.0))
        >>> scaler.fit(X_train, validity_mask_train)
        >>> X_train_scaled = scaler.transform(X_train)
        >>> X_val_scaled   = scaler.transform(X_val)

    The scaler is serialisable via ``state_dict()`` / ``from_state_dict()``.
    """

    def __init__(self, winsorize_range: Tuple[float, float] = (-6.0, 6.0)) -> None:
        self.winsorize_range = winsorize_range
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.names: list[str] | None = None
        self.keep_mask: list[bool] | np.ndarray | None = None
        self.versions: dict[str, str] | None = None
        self.kinds: list[str] | None = None
        self.params: list[Dict[str, object]] | None = None

    def fit(
        self,
        X: np.ndarray,
        validity_mask: np.ndarray,
        names: List[str],
        filter_low_variance: bool = False,
        filter_correlated: bool = False,
        correlation_threshold: float = 0.95,
    ) -> "DescriptorTransformer":
        """Choose a transform for each column, then compute mean and std of the transformed *valid* entries."""
        self.versions = _get_3rd_party_versions()

        self.names = list(names)
        self.keep_mask = np.ones(X.shape[1], dtype=bool)
        assert len(self.names) == X.shape[1]

        valid = validity_mask.astype(bool, copy=False)

        lo, hi = self.winsorize_range
        X = X.astype(np.float64, copy=True)
        self.kinds = []
        self.params = []
        for j in range(X.shape[1]):
            kind, params_ = _choose_transform(X[valid[:, j], j], lo, hi)
            self.kinds.append(kind)
            self.params.append(params_)
            X[:, j] = _forward(X[:, j], kind, params_)

        counts = {k: self.kinds.count(k) for k in ("none", "log", "quantile")}
        logger.info(f"Descriptor transforms: {counts}")

        Xm = np.ma.masked_array(X, mask=~valid)

        self.mean_ = Xm.mean(axis=0).filled(0.0)
        self.std_ = Xm.std(axis=0).filled(1.0)

        all_invalid = ~valid.any(axis=0)
        if all_invalid.any():
            logger.info(f"Setting mean=0.0, std=1.0 for {all_invalid.sum()} all-NaN descriptors in train.")
        self.keep_mask = ~all_invalid & self.keep_mask

        zero_std = self.std_ < 1e-12
        if zero_std.any():
            logger.info(f"Setting std=1.0 for {zero_std.sum()} constant/near-constant descriptors")
            self.std_[zero_std] = 1.0

        if filter_low_variance:
            self.keep_mask = ~zero_std & self.keep_mask

        if filter_correlated:
            before = int(self.keep_mask.sum())
            self.keep_mask = _prune_correlated_columns(
                X,
                validity_mask,
                self.keep_mask,
                threshold=correlation_threshold,
            )
            logger.info(
                f"Correlation filter: dropped {before - int(self.keep_mask.sum())} "
                f"columns (|rho| > {correlation_threshold})"
            )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the per-column transform, z-score, and winsorise.

        Invalid positions (which should be 0.0 in *X*) will be
        transformed to ``(0 - mean) / std`` and then clipped.  This is
        fine because the validity mask prevents these from entering the loss.

        Args:
            X: Values array ``[N, D]``.

        Returns:
            Scaled ``np.float64`` array ``[N, D]``.
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler has not been fit yet")

        X = X.astype(np.float64, copy=True)
        for j, (kind, params_) in enumerate(zip(self.kinds, self.params)):
            X[:, j] = _forward(X[:, j], kind, params_)

        scaled = (X - self.mean_) / self.std_
        lo, hi = self.winsorize_range
        scaled = np.clip(scaled, lo, hi)
        return scaled.astype(np.float64)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Undo ``transform``. Not exact for winsorised values or quantile columns."""
        Y = Z.astype(np.float64) * self.std_ + self.mean_
        for j, (kind, params_) in enumerate(zip(self.kinds, self.params)):
            Y[:, j] = _inverse(Y[:, j], kind, params_)
        return Y

    def state_dict(self) -> Dict[str, object]:
        """Serialise scaler parameters to a plain dict (for checkpoint)."""
        return {
            "mean": self.mean_.tolist() if self.mean_ is not None else None,
            "std": self.std_.tolist() if self.std_ is not None else None,
            "winsorize_range": list(self.winsorize_range),
            "names": list(self.names) if self.names is not None else None,
            "keep_mask": self.keep_mask.tolist() if self.keep_mask is not None else None,
            "versions": self.versions if self.versions is not None else None,
            "kinds": self.kinds if self.kinds is not None else None,
            "params": self.params if self.params is not None else None,
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, object]) -> "DescriptorTransformer":
        """Reconstruct a scaler from a serialised ``state_dict``."""
        scaler = cls(winsorize_range=tuple(d["winsorize_range"]))
        if d["mean"] is not None:
            scaler.mean_ = np.array(d["mean"], dtype=np.float64)
        if d["std"] is not None:
            scaler.std_ = np.array(d["std"], dtype=np.float64)
        if d["names"] is not None:
            scaler.names = list(d["names"])
        if d["keep_mask"] is not None:
            scaler.keep_mask = np.array(d["keep_mask"], dtype=bool)
        if d["versions"] is not None:
            scaler.versions = d["versions"]
        if d["kinds"] is not None:
            scaler.kinds = d["kinds"]
        if d["params"] is not None:
            scaler.params = d["params"]

        return scaler


def _prune_correlated_columns(
    X: np.ndarray,
    validity_mask: np.ndarray,
    keep_mask: np.ndarray,
    threshold: float = 0.95,
) -> np.ndarray:
    """Greedily drop highly correlated columns."""
    import pandas as pd

    kept_idx = np.flatnonzero(keep_mask)

    if kept_idx.size < 2:
        return keep_mask.copy()

    data = X[:, kept_idx].astype(np.float64, copy=True)
    valid = validity_mask[:, kept_idx].astype(bool, copy=False)
    data[~valid] = np.nan

    corr = pd.DataFrame(data).corr(method="spearman", min_periods=50).to_numpy()
    corr = np.abs(corr)
    np.fill_diagonal(corr, 0.0)
    corr = np.nan_to_num(corr, nan=0.0)  # too-few overlaps return nan, we treat them as uncorrelated (never prune)

    # Sort by the number of valid entries, deterministic order.
    valid_counts = valid.sum(axis=0)
    order = sorted(range(kept_idx.size), key=lambda j: (-int(valid_counts[j]), int(kept_idx[j])))

    kept_positions: list[int] = []
    drop_original_idx: list[int] = []
    for j in order:
        if kept_positions and corr[j, kept_positions].max() > threshold:
            drop_original_idx.append(int(kept_idx[j]))
        else:
            kept_positions.append(j)

    new_mask = keep_mask.copy()
    new_mask[drop_original_idx] = False
    return new_mask


def _signed_log(x: np.ndarray, c: float) -> np.ndarray:
    """Signed log: ``sign(x) * log1p(|x| / c)``."""
    return np.sign(x) * np.log1p(np.abs(x) / c)


def _signed_log_inv(y: np.ndarray, c: float) -> np.ndarray:
    """Inverse of ``_signed_log``."""
    return np.sign(y) * c * np.expm1(np.abs(y))


def _quantile_fwd(x: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Map to N(0, 1) via train quantiles ``q``; ties map to the middle of their block."""
    probs = np.linspace(0.0, 1.0, len(q))
    p = 0.5 * (np.interp(x, q, probs) - np.interp(-x, -q[::-1], -probs[::-1]))
    return ndtri(np.clip(p, _P_EPS, 1.0 - _P_EPS))


def _quantile_inv(y: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Inverse of ``_quantile_fwd``; saturates outside the train range."""
    probs = np.linspace(0.0, 1.0, len(q))
    return np.interp(ndtr(y), probs, q)


def _is_ok(x: np.ndarray, lo: float, hi: float) -> bool:
    """True if z-score alone suffices: |skew| < 1 and < 1% outside the winsorize range."""
    std = x.std()
    if std < 1e-12:
        return True
    z = (x - x.mean()) / std

    return abs(skew(x)) < 1.0 and np.mean((z < lo) | (z > hi)) < 0.01  # 1% outside the winsorization range


def _choose_transform(x: np.ndarray, lo: float, hi: float) -> Tuple[str, Dict[str, object]]:
    """Pick 'none', 'log' or 'quantile' for one column of valid values, plus its params."""
    assert np.isfinite(x).all()

    if x.size < _MIN_VALID or _is_ok(x, lo, hi) or np.unique(x).size < 10:
        return "none", {}

    nz = np.abs(x[x != 0])
    c = float(np.percentile(nz, 5)) if nz.size else 1.0
    c = c or 1.0
    if _is_ok(_signed_log(x, c), lo, hi):
        return "log", {"c": c}

    q = np.quantile(x, np.linspace(0.0, 1.0, _N_QUANTILES))
    return "quantile", {"q": q.tolist()}


def _forward(x: np.ndarray, kind: str, params: Dict[str, object]) -> np.ndarray:
    """Apply the pre-transform for one column."""
    if kind == "log":
        return _signed_log(x, params["c"])
    if kind == "quantile":
        return _quantile_fwd(x, np.asarray(params["q"]))
    return x


def _inverse(y: np.ndarray, kind: str, params: Dict[str, object]) -> np.ndarray:
    """Undo the pre-transform for one column."""
    if kind == "log":
        return _signed_log_inv(y, params["c"])
    if kind == "quantile":
        return _quantile_inv(y, np.asarray(params["q"]))
    return y
