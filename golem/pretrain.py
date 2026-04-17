"""Pretraining loop for Graph Transformers on Mordred descriptors.

Uses ``GraphTransformerNet(num_tasks=num_descriptors)`` directly — no wrapper
model. The built-in ``mu_mlp`` serves as the descriptor prediction head.
"""

from __future__ import annotations

import copy
import csv
import inspect
import json
import logging
import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from golem.config import (
    ECFPLatentAlignmentConfig,
    PretrainConfig,
    validate_pretrain_config,
)
from golem.descriptors import (
    Conformer3DPool,
    NaNAwareStandardScaler,
    PreparedDescriptorTargets,
    compute_boltzmann_weighted_3d_statistics,
    materialize_boltzmann_mean_targets,
    prepare_descriptor_targets,
    scale_boltzmann_3d_pools,
)
from golem.ecfp_latent_alignment import (
    compute_alignment_batch,
    compute_alignment_metrics,
    compute_fingerprints,
)
from golem.isoforms import enumerate_isoforms_batch
from golem.report import generate_report
from golem.utils import load_smiles, make_loader, resolve_torch_device, seed_everything

logger = logging.getLogger(__name__)

_BATCH_NORM_NAMES = {"bn", "batchnorm", "batch_norm"}
_SPLIT_NAMES = ("train", "val", "test")
METRICS_FIELDNAMES = [
    "epoch",
    "train_loss",
    "val_loss",
    "train_descriptor_loss",
    "val_descriptor_loss",
    "val_rmse",
    "learning_rate",
    "elapsed_seconds",
    "train_alignment_loss",
    "val_alignment_loss",
    "val_alignment_spearman",
    "val_alignment_kendall",
]
SplitIndices = dict[str, np.ndarray | None]


@dataclass
class EpochMetrics:
    objective_loss: float = math.nan
    descriptor_loss: float = math.nan
    rmse: float = math.nan
    alignment_loss: float = math.nan
    alignment_spearman: float = math.nan
    alignment_kendall: float = math.nan


class _BoltzmannTrainingDataset:
    """Dataset wrapper that samples one cached 3D conformer target per fetch."""

    def __init__(
        self,
        dataset: Sequence,
        conformer_pools: Sequence[Conformer3DPool],
        three_d_slice: slice,
        *,
        seed: int,
    ) -> None:
        if len(dataset) != len(conformer_pools):
            raise ValueError("Training dataset and Boltzmann conformer pools must align.")
        self._dataset = dataset
        self._conformer_pools = list(conformer_pools)
        self._three_d_slice = three_d_slice
        self._sampling_probabilities = [
            self._normalize_sampling_probabilities(pool) for pool in self._conformer_pools
        ]
        self._seed = int(seed)
        self._epoch = 0

    def __len__(self) -> int:
        return len(self._dataset)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    @staticmethod
    def _normalize_sampling_probabilities(pool: Conformer3DPool) -> np.ndarray:
        probabilities = pool.boltzmann_weights.astype(np.float64, copy=False)
        if pool.values.shape[0] == 0:
            if probabilities.ndim != 1 or probabilities.shape[0] != 0:
                raise ValueError(
                    "Boltzmann conformer weights must be a 1D vector aligned with the pool rows."
                )
            return np.zeros(0, dtype=np.float64)
        if probabilities.ndim != 1 or probabilities.shape[0] != pool.values.shape[0]:
            raise ValueError(
                "Boltzmann conformer weights must be a 1D vector aligned with the pool rows."
            )
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("Boltzmann conformer weights must be finite.")
        if np.any(probabilities < 0.0):
            raise ValueError("Boltzmann conformer weights must be non-negative.")

        total = float(probabilities.sum())
        if total <= 0.0:
            raise ValueError("Boltzmann conformer weights must sum to a positive value.")
        return probabilities / total

    def _sample_conformer_index(self, index: int) -> int:
        rng = np.random.default_rng(
            np.random.SeedSequence([self._seed, self._epoch, int(index)])
        )
        return int(
            rng.choice(
                self._conformer_pools[index].values.shape[0],
                p=self._sampling_probabilities[index],
            )
        )

    def __getitem__(self, index: int):
        sample = _clone_data_object(self._dataset[index])
        pool = self._conformer_pools[index]
        if pool.values.shape[0] == 0 or pool.values.shape[1] == 0:
            return sample

        conformer_index = self._sample_conformer_index(index)
        sample.y[:, self._three_d_slice] = torch.from_numpy(
            pool.values[conformer_index]
        ).to(dtype=sample.y.dtype).unsqueeze(0)
        sample.y_mask[:, self._three_d_slice] = torch.from_numpy(
            pool.validity_mask[conformer_index].astype(np.float32, copy=False)
        ).to(dtype=sample.y_mask.dtype).unsqueeze(0)
        return sample


def _checkpoint_library_versions() -> dict[str, str]:
    """Return library versions recorded in generated checkpoints."""
    import golem
    import gt_pyg

    return {"golem": golem.__version__, "gt_pyg": gt_pyg.__version__}


def _format_optional_metric(value: float) -> str:
    """Format finite optional metrics and leave inactive ones blank."""
    return f"{value:.6f}" if math.isfinite(value) else ""


def _setup_logging(output_dir: Path, verbose: bool = False) -> None:
    """Configure logging to both stdout and file."""
    golem_logger = logging.getLogger("golem")
    golem_logger.setLevel(logging.DEBUG)
    golem_logger.propagate = False

    for handler in golem_logger.handlers[:]:
        golem_logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s — %(message)s", datefmt="%H:%M:%S"
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    golem_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(output_dir / "pretrain.log", mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    golem_logger.addHandler(file_handler)


def _make_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    max_epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Return a LambdaLR with linear warmup then cosine decay."""

    def _lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(max_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)


def _forward_batch(
    model: torch.nn.Module,
    batch,
    alignment_cfg: ECFPLatentAlignmentConfig | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    common_kwargs = {
        "batch": batch.batch,
        "zero_var": True,
    }
    if alignment_cfg is None:
        pred, _ = model(batch.x, batch.edge_index, batch.edge_attr, **common_kwargs)
        return pred, None

    pred, _, latent = model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        return_latent=True,
        **common_kwargs,
    )
    return pred, latent


def _run_epoch(
    model: torch.nn.Module,
    loader,
    descriptor_loss_weight: float,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    masking_ratio: float | None = None,
) -> EpochMetrics:
    """Run one training or evaluation epoch."""
    is_training = optimizer is not None
    alignment_cfg: ECFPLatentAlignmentConfig | None = getattr(
        loader, "ecfp_latent_alignment", None
    )
    descriptor_se = 0.0
    descriptor_count = 0
    alignment_losses: list[float] = []
    spearmans: list[float] = []
    kendalls: list[float] = []
    saw_optimization_step = False

    model.train(is_training)
    context = nullcontext() if is_training else torch.no_grad()
    with context:
        for batch in loader:
            batch = batch.to(device)
            targets = batch.y
            valid_mask = batch.y_mask.bool()
            pred, latent = _forward_batch(model, batch, alignment_cfg)

            if is_training:
                descriptor_mask = (torch.rand_like(targets) < masking_ratio).bool()
                descriptor_mask &= valid_mask
                masked_count = int(descriptor_mask.sum().item())
                if masked_count == 0 and valid_mask.sum().item() > 0:
                    descriptor_mask = valid_mask
                    masked_count = int(descriptor_mask.sum().item())
            else:
                descriptor_mask = valid_mask
                masked_count = int(valid_mask.sum().item())

            descriptor_loss = pred.sum() * 0.0
            if masked_count:
                masked_diff = pred[descriptor_mask] - targets[descriptor_mask]
                descriptor_loss = F.mse_loss(
                    pred[descriptor_mask], targets[descriptor_mask]
                )
                descriptor_se += masked_diff.pow(2).sum().item()
                descriptor_count += masked_count

            alignment_loss = pred.sum() * 0.0
            has_alignment_pairs = False
            if latent is not None and alignment_cfg is not None:
                alignment_loss, d_fp, d_z = compute_alignment_batch(
                    batch,
                    latent,
                    alignment_cfg,
                    deterministic_pairs=not is_training,
                )
                has_alignment_pairs = isinstance(d_fp, torch.Tensor) and d_fp.numel() > 0
                if has_alignment_pairs:
                    alignment_losses.append(alignment_loss.item())
                if (
                    not is_training
                    and has_alignment_pairs
                    and alignment_cfg.log_rank_metrics
                ):
                    spearman, kendall = compute_alignment_metrics(d_fp, d_z)
                    if math.isfinite(spearman):
                        spearmans.append(spearman)
                    if math.isfinite(kendall):
                        kendalls.append(kendall)

            if not is_training:
                continue
            if masked_count == 0 and not has_alignment_pairs:
                continue

            saw_optimization_step = True
            total_step_loss = descriptor_loss * descriptor_loss_weight
            if has_alignment_pairs and alignment_cfg is not None:
                total_step_loss = total_step_loss + alignment_loss * alignment_cfg.weight

            optimizer.zero_grad()
            total_step_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

    descriptor_loss_value = math.nan
    if descriptor_count:
        descriptor_loss_value = descriptor_se / descriptor_count
    elif saw_optimization_step:
        descriptor_loss_value = 0.0

    alignment_loss_value = float(np.mean(alignment_losses)) if alignment_losses else math.nan
    objective_terms: list[float] = []
    if math.isfinite(descriptor_loss_value):
        objective_terms.append(descriptor_loss_value * descriptor_loss_weight)
    if math.isfinite(alignment_loss_value) and alignment_cfg is not None:
        objective_terms.append(alignment_loss_value * alignment_cfg.weight)

    return EpochMetrics(
        objective_loss=float(sum(objective_terms)) if objective_terms else math.nan,
        descriptor_loss=descriptor_loss_value,
        rmse=math.sqrt(descriptor_loss_value)
        if not is_training and math.isfinite(descriptor_loss_value)
        else math.nan,
        alignment_loss=alignment_loss_value,
        alignment_spearman=float(np.mean(spearmans)) if spearmans else math.nan,
        alignment_kendall=float(np.mean(kendalls)) if kendalls else math.nan,
    )


def _build_pyg_dataset(
    smiles_list: list[str],
    descriptor_values: np.ndarray,
    validity_mask: np.ndarray,
    fingerprint_bits: np.ndarray | None = None,
) -> list:
    """Build PyG Data objects with graph features + descriptor targets."""
    from gt_pyg import get_tensor_data

    logger.info("Building PyG graph features for %d molecules …", len(smiles_list))
    data_list = get_tensor_data(smiles_list, y=None)
    for index, data in enumerate(data_list):
        data.y = torch.tensor(descriptor_values[index], dtype=torch.float32).unsqueeze(0)
        data.y_mask = torch.tensor(validity_mask[index], dtype=torch.float32).unsqueeze(0)
        if fingerprint_bits is not None:
            data.ecfp = torch.tensor(fingerprint_bits[index], dtype=torch.bool).unsqueeze(0)
    return data_list


def _clone_data_object(data):
    if hasattr(data, "clone") and callable(data.clone):
        return data.clone()
    return copy.deepcopy(data)


def _prepare_scaled_descriptor_targets(
    prepared_targets: PreparedDescriptorTargets,
    train_indices: np.ndarray,
    config: PretrainConfig,
) -> tuple[
    NaNAwareStandardScaler,
    np.ndarray,
    np.ndarray,
    list[Conformer3DPool] | None,
]:
    winsorize_range = config.winsorize_range
    total_width = prepared_targets.values.shape[1]
    mean = np.zeros(total_width, dtype=np.float64)
    std = np.ones(total_width, dtype=np.float64)
    scaled_values = prepared_targets.values.copy()
    scaled_validity = prepared_targets.validity_mask.copy()
    scaled_pools = prepared_targets.boltzmann_3d_pools
    used_boltzmann_3d_stats = False

    two_d_slice = slice(0, prepared_targets.num_2d_descriptors)
    if prepared_targets.num_2d_descriptors > 0:
        two_d_scaler = NaNAwareStandardScaler(winsorize_range=winsorize_range)
        two_d_scaler.fit(
            prepared_targets.values[train_indices, two_d_slice],
            prepared_targets.validity_mask[train_indices, two_d_slice],
        )
        mean[two_d_slice] = two_d_scaler.mean_
        std[two_d_slice] = two_d_scaler.std_
        scaled_values[:, two_d_slice] = two_d_scaler.transform(
            prepared_targets.values[:, two_d_slice]
        )

    three_d_slice = prepared_targets.three_d_slice
    if prepared_targets.num_3d_descriptors > 0:
        if prepared_targets.has_boltzmann_3d and scaled_pools is not None:
            train_pools = [scaled_pools[index] for index in train_indices]
            three_d_mean, three_d_std = compute_boltzmann_weighted_3d_statistics(
                train_pools
            )
            mean[three_d_slice] = three_d_mean
            std[three_d_slice] = three_d_std
            scaled_pools = scale_boltzmann_3d_pools(
                scaled_pools,
                three_d_mean,
                three_d_std,
                winsorize_range,
            )
            mean_targets, mean_validity = materialize_boltzmann_mean_targets(scaled_pools)
            scaled_values[:, three_d_slice] = mean_targets
            scaled_validity[:, three_d_slice] = mean_validity
            used_boltzmann_3d_stats = True
        else:
            three_d_scaler = NaNAwareStandardScaler(winsorize_range=winsorize_range)
            three_d_scaler.fit(
                prepared_targets.values[train_indices, three_d_slice],
                prepared_targets.validity_mask[train_indices, three_d_slice],
            )
            mean[three_d_slice] = three_d_scaler.mean_
            std[three_d_slice] = three_d_scaler.std_
            scaled_values[:, three_d_slice] = three_d_scaler.transform(
                prepared_targets.values[:, three_d_slice]
            )

    scaler = NaNAwareStandardScaler(winsorize_range=winsorize_range)
    scaler.mean_ = mean
    scaler.std_ = std
    if used_boltzmann_3d_stats:
        logger.info(
            "Scaler fit on train split (%d samples) with Boltzmann-weighted 3D statistics",
            len(train_indices),
        )
    else:
        logger.info("Scaler fit on train split (%d samples)", len(train_indices))
    return scaler, scaled_values, scaled_validity, scaled_pools


def _filter_valid_smiles(original_smiles: list[str]) -> list[str]:
    """Drop invalid SMILES before split planning while preserving valid originals."""
    valid_smiles: list[str] = []
    invalid_examples: list[str] = []
    parse_failures = 0
    sanitize_failures = 0

    for smiles in original_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            parse_failures += 1
            if len(invalid_examples) < 5:
                invalid_examples.append(smiles)
            continue

        try:
            Chem.SanitizeMol(mol)
        except Exception:
            sanitize_failures += 1
            if len(invalid_examples) < 5:
                invalid_examples.append(smiles)
            continue

        valid_smiles.append(smiles)

    filtered_count = parse_failures + sanitize_failures
    if filtered_count:
        logger.info(
            "Filtered %d invalid SMILES before splitting (%d parse failures, %d sanitize failures). "
            "Examples: %s",
            filtered_count,
            parse_failures,
            sanitize_failures,
            invalid_examples,
        )
    return valid_smiles


def _derive_core_smiles(smiles: str) -> str:
    """Return the neutralized non-isomeric canonical core used for split grouping."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning(
            "Could not parse SMILES for core grouping; using raw input as its own core: %s",
            smiles,
        )
        return smiles

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        logger.warning(
            "Could not sanitize SMILES for core grouping; using raw input as its own core: %s",
            smiles,
        )
        return smiles

    canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    try:
        uncharged = rdMolStandardize.Uncharger().uncharge(mol)
        if uncharged is None:
            return canonical_smiles
        return Chem.MolToSmiles(uncharged, canonical=True, isomericSmiles=False)
    except Exception:
        logger.debug(
            "Neutralization failed while deriving core SMILES; falling back to canonical sanitized SMILES for %s",
            smiles,
            exc_info=True,
        )
        return canonical_smiles


def _split_counts(total: int, fractions: list[float]) -> list[int]:
    """Convert fractional split ratios into item counts using the existing rounding rule."""
    counts: list[int] = []
    start = 0
    for index, fraction in enumerate(fractions):
        if index == len(fractions) - 1:
            counts.append(total - start)
            continue
        end = start + int(total * fraction)
        counts.append(end - start)
        start = end
    return counts


def _choose_split_boundaries(
    group_sizes: list[int],
    fractions: list[float],
) -> list[int]:
    """Snap ideal split cuts to feasible core-group boundaries.

    ``group_sizes`` describes the shuffled contiguous core blocks that must stay
    intact. We first compute the ideal item counts implied by ``fractions`` and
    convert those counts into cumulative cut positions. Each ideal cut is then
    moved to the nearest group boundary that still leaves at least one whole
    core group available for every remaining split. When two boundaries are
    equally close, we prefer the earlier one so we do not overfill the current
    split.
    """
    n_groups = len(group_sizes)
    n_splits = len(fractions)
    if n_groups < n_splits:
        raise ValueError(
            "Need at least as many unique core groups as requested splits. "
            f"Got {n_groups} unique cores for {n_splits} splits."
        )

    cumulative_sizes = np.cumsum(group_sizes).tolist()
    ideal_counts = _split_counts(sum(group_sizes), fractions)
    ideal_cuts = np.cumsum(ideal_counts[:-1]).tolist()

    boundary_group_counts = list(range(1, n_groups))
    chosen_boundaries: list[int] = []
    previous_group_count = 0
    for cut_index, ideal_cut in enumerate(ideal_cuts):
        # Keep boundaries increasing and reserve at least one group for each
        # later split, since groups cannot be broken apart.
        min_group_count = previous_group_count + 1
        max_group_count = n_groups - (len(ideal_cuts) - cut_index)
        feasible_group_counts = [
            group_count
            for group_count in boundary_group_counts
            if min_group_count <= group_count <= max_group_count
        ]
        if not feasible_group_counts:
            raise ValueError("Could not place split boundaries without breaking core groups.")

        chosen_group_count = min(
            feasible_group_counts,
            key=lambda group_count: (
                abs(cumulative_sizes[group_count - 1] - ideal_cut),
                cumulative_sizes[group_count - 1],
            ),
        )
        chosen_boundaries.append(cumulative_sizes[chosen_group_count - 1])
        previous_group_count = chosen_group_count

    return chosen_boundaries


def _expand_smiles_within_split(
    original_smiles: list[str],
    config: PretrainConfig,
    seen_smiles: set[str],
) -> list[str]:
    """Expand original SMILES within one split and deduplicate across splits."""
    if not original_smiles:
        return []

    expanded = (
        enumerate_isoforms_batch(original_smiles, config.isoforms)
        if config.isoforms.enabled
        else {smiles: [smiles] for smiles in original_smiles}
    )
    split_smiles: list[str] = []
    for original_smiles_value in original_smiles:
        for smiles in expanded[original_smiles_value]:
            if smiles in seen_smiles:
                continue
            seen_smiles.add(smiles)
            split_smiles.append(smiles)
    return split_smiles


def _log_split_sizes(label: str, splits: dict[str, list[str]], *, has_test: bool) -> None:
    logger.info(
        "%s split: train=%d  val=%d%s",
        label,
        len(splits["train"]),
        len(splits["val"]),
        f"  test={len(splits['test'])}" if has_test else "  (no test)",
    )


def _prepare_split_smiles(
    original_smiles: list[str],
    config: PretrainConfig,
) -> tuple[list[str], SplitIndices]:
    """Split shuffled core blocks first, then expand isoforms within each split."""
    original_smiles = _filter_valid_smiles(original_smiles)
    core_to_originals: dict[str, list[str]] = {}
    for smiles in original_smiles:
        core_smiles = _derive_core_smiles(smiles)
        core_to_originals.setdefault(core_smiles, []).append(smiles)

    core_items = list(core_to_originals.items())
    rng = np.random.RandomState(config.seed)
    rng.shuffle(core_items)

    shuffled_originals: list[str] = []
    core_counts = {name: 0 for name in _SPLIT_NAMES}
    for _, core_originals in core_items:
        shuffled_originals.extend(core_originals)

    group_sizes = [len(core_originals) for _, core_originals in core_items]
    cumulative_group_sizes = np.cumsum(group_sizes).tolist()
    boundaries = _choose_split_boundaries(group_sizes, config.split_ratios)
    split_offsets = [0, *boundaries, len(shuffled_originals)]

    split_originals: dict[str, list[str]] = {}
    enabled_split_names = _SPLIT_NAMES[: len(config.split_ratios)]
    for split_index, name in enumerate(enabled_split_names):
        start = split_offsets[split_index]
        end = split_offsets[split_index + 1]
        split_originals[name] = shuffled_originals[start:end]
    for name in _SPLIT_NAMES[len(enabled_split_names):]:
        split_originals[name] = []

    offset = 0
    for split_index, name in enumerate(enabled_split_names):
        split_end = split_offsets[split_index + 1]
        while offset < len(core_items) and split_end >= cumulative_group_sizes[offset]:
            core_counts[name] += 1
            offset += 1

    has_test = len(config.split_ratios) == 3
    logger.info(
        "Core split: train=%d  val=%d%s",
        core_counts["train"],
        core_counts["val"],
        f"  test={core_counts['test']}" if has_test else "  (no test)",
    )
    _log_split_sizes("Original", split_originals, has_test=has_test)

    seen_smiles: set[str] = set()
    split_smiles = {
        name: _expand_smiles_within_split(split_originals[name], config, seen_smiles)
        for name in _SPLIT_NAMES
    }
    if not split_smiles["train"] or not split_smiles["val"] or (
        has_test and not split_smiles["test"]
    ):
        raise ValueError(
            "One or more data splits became empty after core-group splitting and "
            "within-split isoform deduplication. Increase dataset size, adjust "
            "split ratios, or disable aggressive subsampling."
        )
    _log_split_sizes("Expanded", split_smiles, has_test=has_test)

    all_smiles: list[str] = []
    split_indices: SplitIndices = {}
    for name in _SPLIT_NAMES:
        if name not in enabled_split_names:
            split_indices[name] = None
            continue
        start = len(all_smiles)
        all_smiles.extend(split_smiles[name])
        split_indices[name] = np.arange(start, len(all_smiles))
    return all_smiles, split_indices


def _build_split_datasets(
    smiles_list: list[str],
    descriptor_values: np.ndarray,
    validity_mask: np.ndarray,
    split_indices: SplitIndices,
    fingerprint_bits: np.ndarray | None = None,
) -> dict[str, list]:
    datasets: dict[str, list] = {}
    for name, indices in split_indices.items():
        if indices is None:
            continue
        datasets[name] = _build_pyg_dataset(
            [smiles_list[index] for index in indices],
            descriptor_values[indices],
            validity_mask[indices],
            fingerprint_bits=fingerprint_bits[indices] if fingerprint_bits is not None else None,
        )
    return datasets


def _make_split_loaders(
    datasets: dict[str, list],
    config: PretrainConfig,
    alignment_cfg: ECFPLatentAlignmentConfig,
) -> dict[str, object]:
    loaders: dict[str, object] = {}
    for name, dataset in datasets.items():
        loader = make_loader(
            dataset,
            config.batch_size,
            shuffle=name == "train",
            num_workers=config.num_workers,
            persistent_workers=not isinstance(dataset, _BoltzmannTrainingDataset)
            and config.num_workers > 0,
        )
        loader.ecfp_latent_alignment = alignment_cfg if alignment_cfg.enabled else None
        loaders[name] = loader
    return loaders


def _validate_training_batch_configuration(
    train_size: int,
    config: PretrainConfig,
) -> None:
    """Reject batch-norm training setups that would create singleton batches."""
    if config.model.norm.lower() not in _BATCH_NORM_NAMES or train_size == 0:
        return

    singleton_final_batch = config.batch_size == 1 or train_size % config.batch_size == 1
    if not singleton_final_batch:
        return

    suggested_batch_sizes: list[int] = []
    for candidate in (config.batch_size - 1, config.batch_size + 1, train_size):
        if 1 < candidate <= train_size and train_size % candidate != 1:
            if candidate not in suggested_batch_sizes:
                suggested_batch_sizes.append(candidate)

    suggestion = ""
    if suggested_batch_sizes:
        suggestion = " Try batch_size=" + " or ".join(
            str(candidate) for candidate in suggested_batch_sizes
        ) + "."

    raise ValueError(
        "batch_size="
        f"{config.batch_size} with {train_size} training samples would create a "
        "singleton final training batch, which is incompatible with "
        f"model.norm={config.model.norm!r}. Choose a batch size where "
        "train_size % batch_size != 1, adjust the split/subsample, or switch "
        f"away from batch norm.{suggestion}"
    )


def _checkpoint_extra(
    config: PretrainConfig,
    scaler: NaNAwareStandardScaler,
    descriptor_names: list[str],
    num_descriptors: int,
    split_indices: SplitIndices,
) -> dict[str, object]:
    return {
        "scaler_state": scaler.state_dict(),
        "descriptor_names": descriptor_names,
        "descriptor_count": num_descriptors,
        "config": asdict(config),
        "library_versions": _checkpoint_library_versions(),
        "split_indices": {
            name: indices.tolist() if indices is not None else None
            for name, indices in split_indices.items()
        },
    }


def pretrain(
    smiles_path: str,
    config: PretrainConfig,
    output_dir: str,
    subsample: float | None = None,
    verbose: bool = False,
) -> Path:
    """Full pretraining pipeline."""
    effective_subsample = subsample if subsample is not None else config.subsample
    resolved_config = replace(config, subsample=effective_subsample)
    validate_pretrain_config(resolved_config)
    device = resolve_torch_device(resolved_config.device)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for artifact_name in (
        "best_checkpoint.pt",
        "last_checkpoint.pt",
        "metrics.csv",
        "pretrain_report.html",
    ):
        artifact_path = output_dir / artifact_name
        if artifact_path.exists():
            artifact_path.unlink()
    _setup_logging(output_dir, verbose=verbose)

    if verbose:
        try:
            import gypsum_dl
            import rdkit

            logger.debug(
                "rdkit %s, gypsum_dl %s", rdkit.__version__, gypsum_dl.__version__
            )
        except Exception:
            pass

    logger.info("Configured device: %s", resolved_config.device)
    logger.info("Resolved device: %s", device)

    seed_everything(resolved_config.seed, enable_cuda=device.type == "cuda")

    if resolved_config.warmup_epochs >= resolved_config.max_epochs:
        logger.warning(
            "warmup_epochs (%d) >= max_epochs (%d): model will only warm up, never decay",
            resolved_config.warmup_epochs,
            resolved_config.max_epochs,
        )

    with open(output_dir / "resolved_config.yaml", "w") as handle:
        yaml.dump(
            json.loads(json.dumps(asdict(resolved_config))),
            handle,
            default_flow_style=False,
            sort_keys=False,
        )

    smiles_list = load_smiles(smiles_path)
    logger.info("Loaded %d SMILES from %s", len(smiles_list), smiles_path)
    if effective_subsample is not None and 0 < effective_subsample < 1:
        rng = np.random.RandomState(config.seed)
        n_subsampled = max(1, int(len(smiles_list) * effective_subsample))
        indices = rng.choice(len(smiles_list), size=n_subsampled, replace=False)
        smiles_list = [smiles_list[index] for index in sorted(indices)]
        logger.info(
            "Subsampled to %d SMILES (%.1f%%)",
            len(smiles_list),
            effective_subsample * 100,
        )

    smiles_list, split_indices = _prepare_split_smiles(smiles_list, config)
    train_idx = split_indices["train"]
    val_idx = split_indices["val"]
    assert train_idx is not None and val_idx is not None

    logger.info(
        "Computing descriptor targets (2D=%s, 3D=%s) …",
        config.descriptors.include_2d_targets,
        config.descriptors.include_3d_targets,
    )
    prepared_targets = prepare_descriptor_targets(
        smiles_list,
        config.descriptors,
        config.conformers,
        seed=config.seed,
    )
    descriptor_values = prepared_targets.values
    descriptor_validity = prepared_targets.validity_mask
    descriptor_names = prepared_targets.descriptor_names
    num_descriptors = descriptor_values.shape[1]
    logger.info(
        "Descriptor matrix: %d molecules × %d descriptors",
        descriptor_values.shape[0],
        num_descriptors,
    )

    alignment_cfg = config.ecfp_latent_alignment
    fingerprint_bits = None
    if alignment_cfg.enabled:
        fingerprint_bits = compute_fingerprints(smiles_list, alignment_cfg)
        logger.info(
            "Fingerprint matrix: %d molecules × %d bits",
            fingerprint_bits.shape[0],
            fingerprint_bits.shape[1],
        )

    scaler, descriptor_values, descriptor_validity, boltzmann_pools = (
        _prepare_scaled_descriptor_targets(
            prepared_targets,
            train_idx,
            config,
        )
    )

    datasets = _build_split_datasets(
        smiles_list,
        descriptor_values,
        descriptor_validity,
        split_indices,
        fingerprint_bits=fingerprint_bits,
    )
    if (
        prepared_targets.has_boltzmann_3d
        and boltzmann_pools is not None
        and prepared_targets.num_3d_descriptors > 0
    ):
        train_pool_slice = [boltzmann_pools[index] for index in train_idx]
        datasets["train"] = _BoltzmannTrainingDataset(
            datasets["train"],
            train_pool_slice,
            prepared_targets.three_d_slice,
            seed=config.seed,
        )
    _validate_training_batch_configuration(len(datasets["train"]), config)
    loaders = _make_split_loaders(datasets, config, alignment_cfg)

    from gt_pyg import GraphTransformerNet
    from gt_pyg.data import get_atom_feature_dim, get_bond_feature_dim

    model_config = config.model
    gt_params = inspect.signature(GraphTransformerNet.__init__).parameters
    model_kwargs = {
        "node_dim_in": get_atom_feature_dim(),
        "edge_dim_in": get_bond_feature_dim(),
        "num_gt_layers": model_config.num_gt_layers,
        "hidden_dim": model_config.hidden_dim,
        "num_heads": model_config.num_heads,
        "norm": model_config.norm,
        "gt_aggregators": model_config.gt_aggregators,
        "aggregators": model_config.aggregators,
        "dropout": model_config.dropout,
        "act": model_config.act,
        "gate": model_config.gate,
        "qkv_bias": model_config.qkv_bias,
        "num_tasks": num_descriptors,
        "num_head_layers": model_config.num_head_layers,
        "head_norm": model_config.head_norm,
        "head_residual": model_config.head_residual,
    }
    if model_config.head_dropout is not None:
        if "head_dropout" not in gt_params:
            raise RuntimeError(
                "model.head_dropout is set, but the installed gt-pyg "
                "GraphTransformerNet does not accept head_dropout."
            )
        model_kwargs["head_dropout"] = model_config.head_dropout

    model = GraphTransformerNet(**model_kwargs).to(device)
    if alignment_cfg.enabled and "return_latent" not in inspect.signature(
        model.forward
    ).parameters:
        raise RuntimeError(
            "ECFP-latent alignment requires gt-pyg with "
            "GraphTransformerNet.forward(..., return_latent=True)."
        )
    logger.info(
        "Model: %d trainable parameters, num_tasks=%d",
        model.num_parameters(),
        num_descriptors,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = _make_warmup_cosine_scheduler(
        optimizer, config.warmup_epochs, config.max_epochs
    )

    best_val_objective = float("inf")
    best_epoch = -1
    patience_counter = 0
    start_time = time.time()
    best_ckpt_path = output_dir / "best_checkpoint.pt"
    last_ckpt_path = output_dir / "last_checkpoint.pt"
    metrics_path = output_dir / "metrics.csv"
    last_val_loss = math.nan
    last_completed_epoch: int | None = None
    last_saved_epoch: int | None = None

    def _save_checkpoint(
        path: Path,
        *,
        epoch: int,
        best_metric: float | None,
    ) -> None:
        model.save_checkpoint(
            path=path,
            epoch=epoch,
            best_metric=best_metric,
            extra=_checkpoint_extra(
                resolved_config,
                scaler,
                descriptor_names,
                num_descriptors,
                split_indices,
            ),
        )

    def _save_last_checkpoint(epoch_to_save: int) -> None:
        nonlocal last_saved_epoch
        _save_checkpoint(
            last_ckpt_path,
            epoch=epoch_to_save,
            best_metric=best_val_objective if math.isfinite(best_val_objective) else None,
        )
        last_saved_epoch = epoch_to_save

    logger.info(
        "Starting training: max_epochs=%d  patience=%d  masking_ratio=%.2f",
        config.max_epochs,
        config.patience,
        config.masking_ratio,
    )

    epoch = 0
    try:
        with open(metrics_path, "w", newline="") as metrics_file:
            metrics_writer = csv.writer(metrics_file)
            metrics_writer.writerow(METRICS_FIELDNAMES)
            metrics_file.flush()

            for epoch in range(config.max_epochs):
                train_dataset = getattr(loaders["train"], "dataset", None)
                if isinstance(train_dataset, _BoltzmannTrainingDataset):
                    train_dataset.set_epoch(epoch)
                lr = scheduler.get_last_lr()[0]
                train_metrics = _run_epoch(
                    model,
                    loaders["train"],
                    config.descriptors.loss_weight,
                    device,
                    optimizer=optimizer,
                    masking_ratio=config.masking_ratio,
                )
                val_metrics = _run_epoch(
                    model,
                    loaders["val"],
                    config.descriptors.loss_weight,
                    device,
                )
                last_val_loss = val_metrics.objective_loss
                last_completed_epoch = epoch

                metrics_writer.writerow(
                    [
                        epoch,
                        f"{train_metrics.objective_loss:.6f}",
                        f"{val_metrics.objective_loss:.6f}",
                        f"{train_metrics.descriptor_loss:.6f}",
                        f"{val_metrics.descriptor_loss:.6f}",
                        f"{val_metrics.rmse:.6f}",
                        f"{lr:.2e}",
                        f"{time.time() - start_time:.1f}",
                        _format_optional_metric(train_metrics.alignment_loss),
                        _format_optional_metric(val_metrics.alignment_loss),
                        _format_optional_metric(val_metrics.alignment_spearman),
                        _format_optional_metric(val_metrics.alignment_kendall),
                    ]
                )
                metrics_file.flush()

                summary_parts = [
                    f"train_loss={train_metrics.objective_loss:.4f}",
                    f"val_loss={val_metrics.objective_loss:.4f}",
                    f"train_desc={train_metrics.descriptor_loss:.4f}",
                    f"val_desc={val_metrics.descriptor_loss:.4f}",
                    f"val_rmse={val_metrics.rmse:.4f}",
                    f"lr={lr:.2e}",
                ]
                if alignment_cfg.enabled and math.isfinite(train_metrics.alignment_loss):
                    summary_parts.append(f"train_align={train_metrics.alignment_loss:.4f}")
                if alignment_cfg.enabled and math.isfinite(val_metrics.alignment_loss):
                    summary_parts.append(f"val_align={val_metrics.alignment_loss:.4f}")
                logger.info(
                    "Epoch %3d/%d — %s",
                    epoch + 1,
                    config.max_epochs,
                    "  ".join(summary_parts),
                )
                if math.isfinite(val_metrics.alignment_spearman) or math.isfinite(
                    val_metrics.alignment_kendall
                ):
                    logger.info(
                        "           val_alignment_spearman=%.4f  val_alignment_kendall=%.4f",
                        val_metrics.alignment_spearman,
                        val_metrics.alignment_kendall,
                    )

                if math.isfinite(val_metrics.objective_loss) and (
                    val_metrics.objective_loss < best_val_objective
                ):
                    best_val_objective = val_metrics.objective_loss
                    best_epoch = epoch
                    patience_counter = 0
                    _save_checkpoint(
                        best_ckpt_path,
                        epoch=epoch,
                        best_metric=best_val_objective,
                    )
                    logger.info("  ↳ New best — saved %s", best_ckpt_path.name)
                else:
                    if not math.isfinite(val_metrics.objective_loss):
                        logger.warning(
                            "Validation objective is non-finite at epoch %d; skipping best-checkpoint update",
                            epoch + 1,
                        )
                    patience_counter += 1

                _save_last_checkpoint(epoch)
                if patience_counter >= config.patience:
                    logger.info(
                        "Early stopping at epoch %d (patience=%d)",
                        epoch + 1,
                        config.patience,
                    )
                    break
                scheduler.step()
    finally:
        if last_completed_epoch is not None and last_saved_epoch != last_completed_epoch:
            _save_last_checkpoint(last_completed_epoch)

    if not best_ckpt_path.exists():
        logger.warning(
            "No finite validation improvement was recorded; saving the final model as best_checkpoint.pt"
        )
        best_epoch = epoch
        fallback_metric = (
            best_val_objective
            if math.isfinite(best_val_objective)
            else last_val_loss
            if math.isfinite(last_val_loss)
            else math.nan
        )
        _save_checkpoint(best_ckpt_path, epoch=epoch, best_metric=fallback_metric)

    logger.info(
        "Training complete.  Best val_objective=%.4f at epoch %d",
        best_val_objective,
        best_epoch + 1 if best_epoch >= 0 else epoch + 1,
    )

    test_loader = loaders.get("test")
    if test_loader is not None and best_ckpt_path.exists():
        model.load_weights(best_ckpt_path, map_location=device)
        test_metrics = _run_epoch(
            model,
            test_loader,
            config.descriptors.loss_weight,
            device,
        )
        logger.info(
            "Test objective loss=%.4f  descriptor_loss=%.4f  rmse=%.4f",
            test_metrics.objective_loss,
            test_metrics.descriptor_loss,
            test_metrics.rmse,
        )
        if alignment_cfg.enabled:
            logger.info("Test alignment loss=%.4f", test_metrics.alignment_loss)
            if math.isfinite(test_metrics.alignment_spearman) or math.isfinite(
                test_metrics.alignment_kendall
            ):
                logger.info(
                    "Test alignment rank metrics: spearman=%.4f  kendall=%.4f",
                    test_metrics.alignment_spearman,
                    test_metrics.alignment_kendall,
                )

    try:
        generate_report(output_dir)
    except Exception:
        logger.warning("Could not generate HTML report", exc_info=True)

    logger.info("Outputs saved to %s", output_dir)
    return best_ckpt_path
