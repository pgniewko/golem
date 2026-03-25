"""Geometry regularizer helpers for pretraining."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from golem.config import GeometryConfig

logger = logging.getLogger(__name__)


@dataclass
class GeometryBatchResult:
    """Loss and sampled distances for one geometry-regularizer batch."""

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
    num_pairs: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample unique unordered graph pairs from a batch."""
    if batch_size < 2 or num_pairs <= 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty

    all_pairs = torch.triu_indices(batch_size, batch_size, offset=1, device=device)
    if all_pairs.size(1) <= num_pairs:
        return all_pairs[0], all_pairs[1]

    perm = torch.randperm(all_pairs.size(1), device=device)[:num_pairs]
    return all_pairs[0, perm], all_pairs[1, perm]


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


def _forward_with_latent(
    model: torch.nn.Module,
    batch,
    *,
    zero_var: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward pass that also returns the pre-dropout graph embedding."""
    h = model.node_emb(batch.x)
    h = model.input_norm(h)
    h = model.input_dropout(h)

    if model.edge_emb is not None:
        if batch.edge_attr is None:
            raise ValueError("Model expects edge_attr but batch.edge_attr is None")
        e = model.edge_emb(batch.edge_attr)
    else:
        e = None

    for gt_layer in model.gt_layers:
        h, e = gt_layer(x=h, edge_index=batch.edge_index, edge_attr=e)

    z = model.global_pool(h, batch.batch)
    z = model.readout_norm(z)
    head_input = model.readout_dropout(z)
    mu = model.mu_mlp(head_input)
    log_var = torch.clamp(model.log_var_mlp(head_input), min=-10.0, max=10.0)

    if model.training and not zero_var:
        std = torch.exp(0.5 * log_var)
        pred = mu + std * torch.randn_like(std)
    else:
        pred = mu

    return pred, log_var, z


def _compute_geometry_batch(
    batch,
    z: torch.Tensor,
    geometry: GeometryConfig,
) -> GeometryBatchResult:
    """Return geometry loss and sampled primal/latent pair distances."""
    if not hasattr(batch, "ecfp"):
        empty = z.new_empty(0)
        return GeometryBatchResult(loss=z.sum() * 0.0, d_fp=empty, d_z=empty)

    pair_i, pair_j = _sample_batch_pairs(z.size(0), geometry.num_pairs, z.device)
    if pair_i.numel() == 0:
        empty = z.new_empty(0)
        return GeometryBatchResult(loss=z.sum() * 0.0, d_fp=empty, d_z=empty)

    d_fp = _tanimoto_distance_for_pairs(batch.ecfp, pair_i, pair_j)
    d_z = _latent_distance_for_pairs(z, pair_i, pair_j, geometry.latent_metric)
    loss, _num_comparisons = _pair_order_surrogate(
        d_fp,
        d_z,
        temperature=geometry.temperature,
        tie_epsilon=geometry.tie_epsilon,
    )
    return GeometryBatchResult(loss=loss, d_fp=d_fp, d_z=d_z)


def _fingerprint_cache_path(
    output_dir: Path,
    cache_key: str,
    geometry: GeometryConfig,
) -> Path:
    """Return the fingerprint cache path for the current geometry settings."""
    return output_dir / (
        f"ecfp_r{geometry.fp_radius}_b{geometry.fp_bits}_{cache_key}.npz"
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
    geometry: GeometryConfig,
) -> np.ndarray:
    """Load cached ECFP bits or compute them from expanded isoform SMILES."""
    cache_key = _smiles_cache_key(smiles_list)
    cache_path = _fingerprint_cache_path(output_dir, cache_key, geometry)

    if cache_path.exists():
        logger.info("Loading cached ECFP bits from %s", cache_path.name)
        cached = np.load(cache_path, allow_pickle=False)
        return cached["bits"].astype(np.bool_, copy=False)

    logger.info(
        "Computing ECFP fingerprints for %d molecules (radius=%d, bits=%d) …",
        len(smiles_list),
        geometry.fp_radius,
        geometry.fp_bits,
    )
    fp_bits = _compute_ecfp_fingerprints(
        smiles_list,
        radius=geometry.fp_radius,
        fp_bits=geometry.fp_bits,
    )
    np.savez(cache_path, bits=fp_bits)
    logger.info("Saved fingerprint cache to %s", cache_path.name)
    return fp_bits
