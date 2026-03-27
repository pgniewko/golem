"""Rank-alignment regularizer helpers for pretraining."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from golem.config import RankAlignmentConfig

logger = logging.getLogger(__name__)


@dataclass
class RankAlignmentBatchResult:
    """Loss and sampled distances for one rank-alignment batch."""

    loss: torch.Tensor
    d_fp: torch.Tensor
    d_z: torch.Tensor


def _smiles_cache_key(smiles_list: List[str]) -> str:
    """Return a 16-char hex SHA-256 hash of the SMILES list."""
    import hashlib

    h = hashlib.sha256("\n".join(smiles_list).encode())
    return h.hexdigest()[:16]


def _sample_batch_pairs(
    batch_size: int,
    num_pairs: Optional[int],
    device: torch.device,
    *,
    deterministic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select unique unordered graph pairs from a batch.

    We use uniform sampling of unordered pairs within each batch. More
    sophisticated strategies (e.g., hard-pair mining or
    similarity-stratified sampling) could improve the sample efficiency,
    but are not explored at the moment.
    """
    if batch_size < 2 or (num_pairs is not None and num_pairs <= 0):
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty

    all_pairs = torch.triu_indices(batch_size, batch_size, offset=1, device=device)
    total_pairs = all_pairs.size(1)
    if num_pairs is None or total_pairs <= num_pairs:
        return all_pairs[0], all_pairs[1]

    if deterministic:
        # Spread validation coverage across the batch without introducing RNG noise.
        step = total_pairs / num_pairs
        indices = torch.floor(torch.arange(num_pairs, device=device) * step).long()
    else:
        indices = torch.randperm(total_pairs, device=device)[:num_pairs]
    return all_pairs[0, indices], all_pairs[1, indices]


def _tanimoto_distance_for_pairs(
    fp_bits: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
) -> torch.Tensor:
    """Compute Tanimoto distance for sampled fingerprint pairs."""
    fp_i = fp_bits[pair_i].bool()
    fp_j = fp_bits[pair_j].bool()
    intersection = (fp_i & fp_j).sum(dim=-1).float()
    union = (fp_i | fp_j).sum(dim=-1).float()
    similarity = torch.where(union > 0, intersection / union, torch.ones_like(union))
    return 1.0 - similarity


def _latent_distance_for_pairs(
    z: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    metric: str,
) -> torch.Tensor:
    """Compute latent distances for sampled graph-embedding pairs."""
    metric = metric.lower()
    z_norm = F.normalize(z, p=2, dim=-1, eps=1e-8)

    if metric == "cosine":
        cosine_sim = (z_norm[pair_i] * z_norm[pair_j]).sum(dim=-1).clamp(-1.0, 1.0)
        return 1.0 - cosine_sim
    if metric == "l2_norm":
        return (z_norm[pair_i] - z_norm[pair_j]).pow(2).sum(dim=-1)

    raise ValueError(f"Unsupported latent metric: {metric!r}")


def _pair_order_surrogate(
    d_fp: torch.Tensor,
    d_z: torch.Tensor,
    temperature: float,
    tie_epsilon: float,
) -> Tuple[torch.Tensor, int]:
    """Smooth pair-of-pairs ranking loss over sampled distances."""
    if d_fp.numel() < 2 or d_z.numel() < 2:
        return d_z.sum() * 0.0, 0

    delta_fp = d_fp[:, None] - d_fp[None, :]
    delta_z = d_z[:, None] - d_z[None, :]
    upper_mask = torch.triu(
        torch.ones_like(delta_fp, dtype=torch.bool), diagonal=1
    )
    informative = delta_fp.abs() > tie_epsilon
    mask = upper_mask & informative

    if not mask.any():
        return d_z.sum() * 0.0, 0

    direction = torch.sign(delta_fp[mask])
    scaled_margin = direction * delta_z[mask] / max(temperature, 1e-8)
    weights = delta_fp[mask].abs()
    loss = (F.softplus(-scaled_margin) * weights).sum() / weights.sum().clamp_min(1e-8)
    return loss, int(mask.sum().item())


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return 1-based average ranks with stable tie handling."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float64)

    start = 0
    while start < len(sorted_values):
        end = start + 1
        while end < len(sorted_values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = avg_rank
        start = end

    return ranks


def _spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rank correlation on 1-D arrays."""
    if x.size < 2 or y.size < 2:
        return math.nan

    rx = _average_ranks(x)
    ry = _average_ranks(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt(np.sum(rx ** 2) * np.sum(ry ** 2))
    if denom == 0.0:
        return math.nan
    return float(np.sum(rx * ry) / denom)


def _kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Kendall tau-b for 1-D arrays."""
    n = x.size
    if n < 2:
        return math.nan

    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0

    for i in range(n - 1):
        dx = x[i] - x[i + 1 :]
        dy = y[i] - y[i + 1 :]
        for dx_ij, dy_ij in zip(dx, dy, strict=False):
            sign_x = 0 if dx_ij == 0 else (1 if dx_ij > 0 else -1)
            sign_y = 0 if dy_ij == 0 else (1 if dy_ij > 0 else -1)
            if sign_x == 0 and sign_y == 0:
                continue
            if sign_x == 0:
                ties_x += 1
            elif sign_y == 0:
                ties_y += 1
            elif sign_x == sign_y:
                concordant += 1
            else:
                discordant += 1

    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    if denom == 0.0:
        return math.nan
    return (concordant - discordant) / denom


def _pair_rank_metrics(d_fp: torch.Tensor, d_z: torch.Tensor) -> Tuple[float, float]:
    """Compute exact rank-correlation metrics on sampled pair distances."""
    if d_fp.numel() < 2 or d_z.numel() < 2:
        return math.nan, math.nan

    fp_np = d_fp.detach().cpu().numpy().astype(np.float64, copy=False)
    z_np = d_z.detach().cpu().numpy().astype(np.float64, copy=False)
    return _spearman_correlation(fp_np, z_np), _kendall_tau(fp_np, z_np)


def _compute_rank_alignment_batch(
    batch,
    z: torch.Tensor,
    rank_alignment: RankAlignmentConfig,
    *,
    deterministic_pairs: bool = False,
) -> RankAlignmentBatchResult:
    """Return rank-alignment loss and sampled primal/latent pair distances."""
    if not hasattr(batch, "ecfp"):
        empty = z.new_empty(0)
        return RankAlignmentBatchResult(loss=z.sum() * 0.0, d_fp=empty, d_z=empty)

    pair_i, pair_j = _sample_batch_pairs(
        z.size(0),
        rank_alignment.num_pairs,
        z.device,
        deterministic=deterministic_pairs,
    )
    if pair_i.numel() == 0:
        empty = z.new_empty(0)
        return RankAlignmentBatchResult(loss=z.sum() * 0.0, d_fp=empty, d_z=empty)

    d_fp = _tanimoto_distance_for_pairs(batch.ecfp, pair_i, pair_j)
    d_z = _latent_distance_for_pairs(
        z, pair_i, pair_j, rank_alignment.latent_metric
    )
    loss, _num_comparisons = _pair_order_surrogate(
        d_fp,
        d_z,
        temperature=rank_alignment.temperature,
        tie_epsilon=rank_alignment.tie_epsilon,
    )
    return RankAlignmentBatchResult(loss=loss, d_fp=d_fp, d_z=d_z)


def _fingerprint_cache_path(
    output_dir: Path,
    cache_key: str,
    rank_alignment: RankAlignmentConfig,
) -> Path:
    """Return the fingerprint cache path for the current rank-alignment settings."""
    return output_dir / (
        f"ecfp_r{rank_alignment.fp_radius}_b{rank_alignment.fp_bits}_{cache_key}.npz"
    )


def _compute_ecfp_fingerprints(
    smiles_list: List[str],
    *,
    radius: int,
    fp_bits: int,
) -> np.ndarray:
    """Compute Morgan/ECFP bit vectors for a list of SMILES."""
    fps = np.zeros((len(smiles_list), fp_bits), dtype=np.bool_)
    for idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not parse SMILES for fingerprinting: {smiles!r}")
        bit_vect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=fp_bits)
        fp_array = np.zeros((fp_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(bit_vect, fp_array)
        fps[idx] = fp_array.astype(np.bool_)
    return fps


def _load_or_compute_fingerprints(
    output_dir: Path,
    smiles_list: List[str],
    rank_alignment: RankAlignmentConfig,
) -> np.ndarray:
    """Load cached ECFP bits or compute them from expanded isoform SMILES."""
    cache_key = _smiles_cache_key(smiles_list)
    cache_path = _fingerprint_cache_path(output_dir, cache_key, rank_alignment)

    if cache_path.exists():
        logger.info("Loading cached ECFP bits from %s", cache_path.name)
        cached = np.load(cache_path, allow_pickle=False)
        return cached["bits"].astype(np.bool_, copy=False)

    logger.info(
        "Computing ECFP fingerprints for %d molecules (radius=%d, bits=%d) …",
        len(smiles_list),
        rank_alignment.fp_radius,
        rank_alignment.fp_bits,
    )
    fp_bits = _compute_ecfp_fingerprints(
        smiles_list,
        radius=rank_alignment.fp_radius,
        fp_bits=rank_alignment.fp_bits,
    )
    np.savez(cache_path, bits=fp_bits)
    logger.info("Saved fingerprint cache to %s", cache_path.name)
    return fp_bits
