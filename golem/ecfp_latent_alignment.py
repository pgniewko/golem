"""Helpers for aligning ECFP and latent metric spaces."""

from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from golem.cache import cache_dir
from golem.config import ECFPLatentAlignmentConfig

logger = logging.getLogger(__name__)


def _sample_pairs(
    batch_size: int,
    num_pairs: int,
    device: torch.device,
    *,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size < 2:
        logger.warning(
            "Skipping ECFP-latent alignment for batch_size=%d: need at least 2 samples.",
            batch_size,
        )
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty

    if num_pairs <= 0:
        logger.warning(
            "Skipping ECFP-latent alignment for num_pairs=%d: need a positive pair count.",
            num_pairs,
        )
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty

    all_pairs = torch.triu_indices(batch_size, batch_size, offset=1, device=device)
    total_pairs = all_pairs.size(1)
    if total_pairs <= num_pairs:
        return all_pairs[0], all_pairs[1]

    if deterministic:
        step = total_pairs / num_pairs
        indices = torch.floor(torch.arange(num_pairs, device=device) * step).long()
    else:
        indices = torch.randperm(total_pairs, device=device)[:num_pairs]
    return all_pairs[0, indices], all_pairs[1, indices]


def _tanimoto_distances(
    fp_bits: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
) -> torch.Tensor:
    fp_i = fp_bits[pair_i].bool()
    fp_j = fp_bits[pair_j].bool()
    intersection = (fp_i & fp_j).sum(dim=-1).float()
    union = (fp_i | fp_j).sum(dim=-1).float()
    similarity = torch.where(union > 0, intersection / union, torch.ones_like(union))
    return 1.0 - similarity


def _cosine_distances(
    z: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
) -> torch.Tensor:
    z_norm = F.normalize(z, p=2, dim=-1, eps=1e-8)
    cosine_sim = (z_norm[pair_i] * z_norm[pair_j]).sum(dim=-1).clamp(-1.0, 1.0)
    return 1.0 - cosine_sim


def _pair_order_loss(
    d_fp: torch.Tensor,
    d_z: torch.Tensor,
    temperature: float,
    tie_epsilon: float,
) -> torch.Tensor:
    if d_fp.numel() < 2 or d_z.numel() < 2:
        return d_z.sum() * 0.0

    delta_fp = d_fp[:, None] - d_fp[None, :]
    delta_z = d_z[:, None] - d_z[None, :]
    upper_mask = torch.triu(torch.ones_like(delta_fp, dtype=torch.bool), diagonal=1)
    informative = delta_fp.abs() > tie_epsilon
    mask = upper_mask & informative
    if not mask.any():
        return d_z.sum() * 0.0

    direction = torch.sign(delta_fp[mask])
    scaled_margin = direction * delta_z[mask] / max(temperature, 1e-8)
    weights = delta_fp[mask].abs()
    return (F.softplus(-scaled_margin) * weights).sum() / weights.sum().clamp_min(1e-8)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float64)

    start = 0
    while start < len(sorted_values):
        end = start + 1
        while end < len(sorted_values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end

    return ranks


def _kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
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

    denom = math.sqrt(
        (concordant + discordant + ties_x) * (concordant + discordant + ties_y)
    )
    if denom == 0.0:
        return math.nan
    return (concordant - discordant) / denom


def compute_alignment_batch(
    batch,
    z: torch.Tensor,
    config: ECFPLatentAlignmentConfig,
    *,
    deterministic_pairs: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pair_i, pair_j = _sample_pairs(
        z.size(0),
        config.num_pairs,
        z.device,
        deterministic=deterministic_pairs,
    )
    if pair_i.numel() == 0:
        empty = z.new_empty(0)
        return z.sum() * 0.0, empty, empty

    d_fp = _tanimoto_distances(batch.ecfp, pair_i, pair_j)
    d_z = _cosine_distances(z, pair_i, pair_j)
    return _pair_order_loss(d_fp, d_z, config.temperature, config.tie_epsilon), d_fp, d_z


def compute_alignment_metrics(
    d_fp: torch.Tensor,
    d_z: torch.Tensor,
) -> tuple[float, float]:
    if d_fp.numel() < 2 or d_z.numel() < 2:
        return math.nan, math.nan

    fp_np = d_fp.detach().cpu().numpy().astype(np.float64, copy=False)
    z_np = d_z.detach().cpu().numpy().astype(np.float64, copy=False)
    fp_ranks = _average_ranks(fp_np)
    z_ranks = _average_ranks(z_np)
    spearman = float(np.corrcoef(fp_ranks, z_ranks)[0, 1])
    if not math.isfinite(spearman):
        spearman = math.nan
    return spearman, _kendall_tau(fp_np, z_np)


def _fingerprint_cache_path(
    _output_dir: Path,
    smiles_list: List[str],
    config: ECFPLatentAlignmentConfig,
) -> Path:
    cache_key = hashlib.sha256("\n".join(smiles_list).encode()).hexdigest()[:16]
    return cache_dir("fingerprints") / f"ecfp_r{config.fp_radius}_b{config.fp_bits}_{cache_key}.npz"


def load_or_compute_fingerprints(
    output_dir: Path,
    smiles_list: List[str],
    config: ECFPLatentAlignmentConfig,
) -> np.ndarray:
    cache_path = _fingerprint_cache_path(output_dir, smiles_list, config)
    if cache_path.exists():
        logger.info("Loading shared fingerprint cache from %s", cache_path.name)
        cached = np.load(cache_path, allow_pickle=False)
        return cached["bits"].astype(np.bool_, copy=False)

    logger.info(
        "Shared fingerprint cache miss for %d molecules; computing ECFP bits",
        len(smiles_list),
    )
    fps = np.zeros((len(smiles_list), config.fp_bits), dtype=np.bool_)
    for idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not parse SMILES for fingerprinting: {smiles!r}")
        bit_vect = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            config.fp_radius,
            nBits=config.fp_bits,
        )
        fp_array = np.zeros((config.fp_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(bit_vect, fp_array)
        fps[idx] = fp_array.astype(np.bool_)

    np.savez(cache_path, bits=fps)
    logger.info("Saved shared fingerprint cache to %s", cache_path.name)
    return fps
