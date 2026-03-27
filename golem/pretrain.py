"""Pretraining loop for Graph Transformers on Mordred descriptors.

Uses ``GraphTransformerNet(num_tasks=num_descriptors)`` directly — no wrapper
model.  The built-in ``mu_mlp`` serves as the descriptor prediction head.

Pipeline:
1. Load SMILES
2. Split into train / val / (test) at the parent-molecule level
3. (Optional) Enumerate isoforms within each split only
4. Fit ``NaNAwareStandardScaler`` on **train only**
5. Compute Mordred 2D descriptors + validity masks
6. Transform all splits + winsorise
7. Build PyG datasets via ``get_tensor_data``
8. Train with masked MSE (15% random mask ∩ validity mask)
9. Early stopping on validation loss
10. Save checkpoint with scaler, descriptor names, config
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from rdkit import Chem
import torch.nn.functional as F
import yaml
from golem.config import PretrainConfig, RankAlignmentConfig
from golem.descriptors import NaNAwareStandardScaler, compute_mordred_descriptors
from golem.rank_alignment import (
    _compute_rank_alignment_batch,
    _load_or_compute_fingerprints,
    _pair_rank_metrics,
)
from golem.isoforms import enumerate_isoforms_batch
from golem.report import generate_report
from golem.utils import load_smiles, make_loader, seed_everything, split_data

logger = logging.getLogger(__name__)

METRICS_FIELDNAMES = [
    "epoch",
    "train_loss",
    "val_loss",
    "val_rmse",
    "learning_rate",
    "elapsed_seconds",
    "train_rank_loss",
    "train_total_loss",
    "val_rank_loss",
    "val_spearman",
    "val_kendall",
]


@dataclass
class TrainEpochMetrics:
    """Aggregated training losses for a single epoch."""

    main_loss: float
    rank_loss: float
    total_loss: float


@dataclass
class ValidationMetrics:
    """Aggregated validation metrics for a single epoch."""

    loss: float
    rmse: float
    rank_loss: float = math.nan
    spearman: float = math.nan
    kendall: float = math.nan


# ---------------------------------------------------------------------------
# Checkpoint metadata
# ---------------------------------------------------------------------------

def _checkpoint_library_versions() -> dict[str, str]:
    """Return library versions recorded in generated checkpoints."""
    import golem
    import gt_pyg

    return {
        "golem": golem.__version__,
        "gt_pyg": gt_pyg.__version__,
    }


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(output_dir: Path, verbose: bool = False) -> None:
    """Configure logging to both stdout and file.

    Uses a named ``"golem"`` logger with ``propagate=False`` so that
    existing root-logger handlers (e.g. Jupyter, application loggers)
    are not destroyed.

    Args:
        output_dir: Directory for the log file.
        verbose: If True, set console handler to DEBUG level.
    """
    golem_logger = logging.getLogger("golem")
    golem_logger.setLevel(logging.DEBUG)
    golem_logger.propagate = False

    # Remove existing golem handlers (avoid duplicates on re-runs)
    for h in golem_logger.handlers[:]:
        golem_logger.removeHandler(h)

    # Console handler (INFO by default, DEBUG when verbose)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s", datefmt="%H:%M:%S"))
    golem_logger.addHandler(ch)

    # File handler (DEBUG)
    fh = logging.FileHandler(output_dir / "pretrain.log", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s"))
    golem_logger.addHandler(fh)


# ---------------------------------------------------------------------------
# LR schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Single-epoch routines
# ---------------------------------------------------------------------------

def _train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    masking_ratio: float,
    device: torch.device,
    *,
    epoch: int,
    rank_alignment: RankAlignmentConfig,
) -> TrainEpochMetrics:
    """Run one training epoch and return aggregated main/rank/total losses."""
    model.train()
    total_main_loss = 0.0
    total_rank_loss = 0.0
    total_total_loss = 0.0
    n_batches = 0
    rank_alignment_active = (
        rank_alignment.enabled and epoch >= rank_alignment.warmup_epochs
    )

    for batch in loader:
        batch = batch.to(device)
        targets = batch.y  # [B, D]
        valid_mask = batch.y_mask.bool()  # [B, D]

        if rank_alignment_active:
            pred, _log_var, z = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch=batch.batch,
                zero_var=True,
                return_latent=True,
            )
        else:
            pred, _log_var = model(
                batch.x, batch.edge_index, batch.edge_attr,
                batch=batch.batch, zero_var=True,
            )
            z = None

        # Random 15% mask ∩ validity mask
        rand_mask = (torch.rand_like(targets) < masking_ratio).bool()
        final_mask = rand_mask & valid_mask

        if final_mask.sum() == 0:
            # Extremely rare fallback: use all valid positions
            final_mask = valid_mask

        if final_mask.sum() == 0:
            continue  # skip batch if no valid positions at all

        main_loss = F.mse_loss(pred[final_mask], targets[final_mask])
        rank_loss = main_loss.new_zeros(())
        if z is not None:
            rank_loss = _compute_rank_alignment_batch(batch, z, rank_alignment).loss
        total_loss = main_loss + rank_alignment.weight * rank_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_main_loss += main_loss.item()
        total_rank_loss += rank_loss.item()
        total_total_loss += total_loss.item()
        n_batches += 1

    denom = max(n_batches, 1)
    return TrainEpochMetrics(
        main_loss=total_main_loss / denom,
        rank_loss=total_rank_loss / denom,
        total_loss=total_total_loss / denom,
    )


@torch.no_grad()
def _validate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    rank_alignment: RankAlignmentConfig,
    epoch: Optional[int] = None,
) -> ValidationMetrics:
    """Validate on valid positions and optionally track rank-alignment metrics."""
    model.eval()
    total_se = 0.0
    total_count = 0
    rank_losses: List[float] = []
    spearmans: List[float] = []
    kendalls: List[float] = []
    rank_alignment_active = rank_alignment.enabled and (
        epoch is None or epoch >= rank_alignment.warmup_epochs
    )

    for batch in loader:
        batch = batch.to(device)
        targets = batch.y
        valid_mask = batch.y_mask.bool()

        if rank_alignment_active:
            pred, _log_var, z = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch=batch.batch,
                zero_var=True,
                return_latent=True,
            )
            rank_alignment_batch = _compute_rank_alignment_batch(
                batch,
                z,
                rank_alignment,
                deterministic_pairs=True,
            )
            rank_losses.append(rank_alignment_batch.loss.item())
            if rank_alignment.log_rank_metrics:
                spearman, kendall = _pair_rank_metrics(
                    rank_alignment_batch.d_fp, rank_alignment_batch.d_z
                )
                if math.isfinite(spearman):
                    spearmans.append(spearman)
                if math.isfinite(kendall):
                    kendalls.append(kendall)
        else:
            pred, _ = model(
                batch.x, batch.edge_index, batch.edge_attr,
                batch=batch.batch, zero_var=True,
            )

        if valid_mask.sum() == 0:
            continue

        se = (pred[valid_mask] - targets[valid_mask]).pow(2).sum().item()
        total_se += se
        total_count += valid_mask.sum().item()

    mse = total_se / max(total_count, 1)
    rmse = math.sqrt(mse)
    return ValidationMetrics(
        loss=mse,
        rmse=rmse,
        rank_loss=float(np.mean(rank_losses)) if rank_losses else math.nan,
        spearman=float(np.mean(spearmans)) if spearmans else math.nan,
        kendall=float(np.mean(kendalls)) if kendalls else math.nan,
    )


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def _build_pyg_dataset(
    smiles_list: List[str],
    descriptor_values: np.ndarray,
    validity_mask: np.ndarray,
    fingerprint_bits: Optional[np.ndarray] = None,
) -> list:
    """Build PyG Data objects with graph features + descriptor targets.

    Uses ``get_tensor_data`` from gt-pyg for graph featurisation, then
    overwrites ``data.y`` and ``data.y_mask`` with descriptor targets.
    """
    from gt_pyg import get_tensor_data

    logger.info("Building PyG graph features for %d molecules …", len(smiles_list))
    data_list = get_tensor_data(smiles_list, y=None)

    # Overwrite y and y_mask with actual descriptor data.
    # Store as [1, D] so that PyG batching concatenates to [B, D]
    # (matching the stacking behavior of data.y).
    for i, data in enumerate(data_list):
        data.y = torch.tensor(descriptor_values[i], dtype=torch.float32).unsqueeze(0)
        data.y_mask = torch.tensor(validity_mask[i], dtype=torch.float32).unsqueeze(0)
        if fingerprint_bits is not None:
            data.ecfp = torch.tensor(fingerprint_bits[i], dtype=torch.bool).unsqueeze(0)

    return data_list


# ---------------------------------------------------------------------------
# Main pretrain function
# ---------------------------------------------------------------------------

def _smiles_cache_key(smiles_list: List[str]) -> str:
    """Return a 16-char hex SHA-256 hash of the SMILES list (order-sensitive)."""
    h = hashlib.sha256("\n".join(smiles_list).encode())
    return h.hexdigest()[:16]


def _prepare_metrics_file(metrics_path: Path, fieldnames: List[str]) -> None:
    """Upgrade an existing metrics.csv to the current header when resuming."""
    if not metrics_path.exists():
        return

    with open(metrics_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        existing_fieldnames = reader.fieldnames or []

    if existing_fieldnames == fieldnames:
        return

    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _expand_smiles_within_split(
    parent_smiles: List[str],
    config: PretrainConfig,
    seen_smiles: set[str],
) -> List[str]:
    """Expand parent SMILES within a single split and deduplicate across splits."""
    if not parent_smiles:
        return []

    if config.isoforms.enabled:
        iso_map = enumerate_isoforms_batch(parent_smiles, config.isoforms)
        split_smiles: List[str] = []
        for parent_smi in parent_smiles:
            for smi in iso_map[parent_smi]:
                if smi not in seen_smiles:
                    seen_smiles.add(smi)
                    split_smiles.append(smi)
        return split_smiles

    split_smiles = []
    for smi in parent_smiles:
        if smi not in seen_smiles:
            seen_smiles.add(smi)
            split_smiles.append(smi)
    return split_smiles


def _prepare_split_smiles(
    parent_smiles: List[str],
    config: PretrainConfig,
) -> Tuple[List[str], np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Split parent molecules first, then expand isoforms within each split."""
    parent_splits = split_data(len(parent_smiles), config.split_ratios, seed=config.seed)

    if len(parent_splits) == 3:
        parent_train_idx, parent_val_idx, parent_test_idx = parent_splits
    else:
        parent_train_idx, parent_val_idx = parent_splits
        parent_test_idx = None

    train_parents = [parent_smiles[i] for i in parent_train_idx]
    val_parents = [parent_smiles[i] for i in parent_val_idx]
    test_parents = [parent_smiles[i] for i in parent_test_idx] if parent_test_idx is not None else []

    logger.info(
        "Parent split: train=%d  val=%d%s",
        len(train_parents),
        len(val_parents),
        f"  test={len(test_parents)}" if parent_test_idx is not None else "  (no test)",
    )

    seen_smiles: set[str] = set()
    train_smiles = _expand_smiles_within_split(train_parents, config, seen_smiles)
    val_smiles = _expand_smiles_within_split(val_parents, config, seen_smiles)
    test_smiles = _expand_smiles_within_split(test_parents, config, seen_smiles)

    if not train_smiles or not val_smiles or (parent_test_idx is not None and not test_smiles):
        raise ValueError(
            "One or more data splits became empty after parent-level splitting and "
            "within-split isoform deduplication. Increase dataset size, adjust "
            "split ratios, or disable aggressive subsampling."
        )

    all_smiles = train_smiles + val_smiles + test_smiles
    train_idx = np.arange(0, len(train_smiles))
    val_idx = np.arange(len(train_smiles), len(train_smiles) + len(val_smiles))
    test_idx = (
        np.arange(len(train_smiles) + len(val_smiles), len(all_smiles))
        if parent_test_idx is not None
        else None
    )

    logger.info(
        "Expanded split: train=%d  val=%d%s",
        len(train_smiles),
        len(val_smiles),
        f"  test={len(test_smiles)}" if test_idx is not None else "  (no test)",
    )

    return all_smiles, train_idx, val_idx, test_idx


def pretrain(
    smiles_path: str,
    config: PretrainConfig,
    output_dir: str,
    subsample: Optional[float] = None,
    verbose: bool = False,
    resume_from: Optional[str] = None,
) -> Path:
    """Full pretraining pipeline.

    Args:
        smiles_path: Path to SMILES file (``.smi`` or ``.csv``).
        config: Resolved ``PretrainConfig``.
        output_dir: Directory for checkpoints, logs, metrics.
        subsample: If set, randomly subsample this fraction of SMILES
            before processing (e.g. 0.1 for 10%).
        verbose: If True, show DEBUG-level logs on console.
        resume_from: Path to a checkpoint file to resume training from.

    Returns:
        Path to the best checkpoint file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(output_dir, verbose=verbose)

    # Log library versions when verbose
    if verbose:
        try:
            import rdkit
            import gypsum_dl
            logger.debug("rdkit %s, gypsum_dl %s", rdkit.__version__, gypsum_dl.__version__)
        except Exception:
            pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    seed_everything(config.seed)

    if config.max_epochs > 0 and config.warmup_epochs >= config.max_epochs:
        logger.warning(
            "warmup_epochs (%d) >= max_epochs (%d): model will only warm up, never decay",
            config.warmup_epochs, config.max_epochs,
        )

    # Save resolved config (convert tuples to lists for safe_load compatibility)
    config_dict = json.loads(json.dumps(asdict(config)))
    with open(output_dir / "resolved_config.yaml", "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    # ------------------------------------------------------------------
    # 1. Load SMILES
    # ------------------------------------------------------------------
    smiles_list = load_smiles(smiles_path)
    logger.info("Loaded %d SMILES from %s", len(smiles_list), smiles_path)

    if subsample is not None and 0 < subsample < 1:
        rng = np.random.RandomState(config.seed)
        n_sub = max(1, int(len(smiles_list) * subsample))
        indices = rng.choice(len(smiles_list), size=n_sub, replace=False)
        smiles_list = [smiles_list[i] for i in sorted(indices)]
        logger.info("Subsampled to %d SMILES (%.1f%%)", len(smiles_list), subsample * 100)

    # ------------------------------------------------------------------
    # 2. Filter invalid parent SMILES before splitting
    # ------------------------------------------------------------------
    valid_smiles = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
                valid_smiles.append(smi)
            except Exception:
                logger.debug("Filtered invalid SMILES: %s", smi)
        else:
            logger.debug("Filtered unparseable SMILES: %s", smi)
    n_filtered = len(smiles_list) - len(valid_smiles)
    if n_filtered > 0:
        logger.info("Filtered %d invalid SMILES (%d remain)", n_filtered, len(valid_smiles))
    smiles_list = valid_smiles

    # ------------------------------------------------------------------
    # 3. Split parents first, then expand isoforms within each split
    # ------------------------------------------------------------------
    smiles_list, train_idx, val_idx, test_idx = _prepare_split_smiles(smiles_list, config)

    # ------------------------------------------------------------------
    # 4. Compute Mordred descriptors (with disk cache)
    # ------------------------------------------------------------------
    cache_key = _smiles_cache_key(smiles_list)
    cache_path = output_dir / f"descriptors_{cache_key}.npz"

    if cache_path.exists():
        logger.info("Loading cached descriptors from %s", cache_path.name)
        cached = np.load(cache_path, allow_pickle=True)
        desc_values = cached["values"]
        desc_valid = cached["valid"]
        descriptor_names = cached["names"].tolist()
    else:
        logger.info("Computing Mordred descriptors …")
        desc_values, desc_valid, descriptor_names = compute_mordred_descriptors(smiles_list)
        np.savez(cache_path, values=desc_values, valid=desc_valid, names=np.array(descriptor_names))
        logger.info("Saved descriptor cache to %s", cache_path.name)

    num_descriptors = desc_values.shape[1]
    logger.info("Descriptor matrix: %d molecules × %d descriptors", desc_values.shape[0], num_descriptors)

    fp_bits = None
    if config.rank_alignment.enabled:
        fp_bits = _load_or_compute_fingerprints(
            output_dir, smiles_list, config.rank_alignment
        )
        logger.info(
            "Fingerprint matrix: %d molecules × %d bits",
            fp_bits.shape[0],
            fp_bits.shape[1],
        )

    # ------------------------------------------------------------------
    # 5. Fit scaler on TRAIN only
    # ------------------------------------------------------------------
    scaler = NaNAwareStandardScaler(winsorize_range=config.winsorize_range)
    scaler.fit(desc_values[train_idx], desc_valid[train_idx])
    logger.info("Scaler fit on train split (%d samples)", len(train_idx))

    # ------------------------------------------------------------------
    # 6. Transform all splits
    # ------------------------------------------------------------------
    desc_values = scaler.transform(desc_values)

    # ------------------------------------------------------------------
    # 7. Build PyG datasets
    # ------------------------------------------------------------------
    train_smiles = [smiles_list[i] for i in train_idx]
    val_smiles = [smiles_list[i] for i in val_idx]

    train_fp = fp_bits[train_idx] if fp_bits is not None else None
    val_fp = fp_bits[val_idx] if fp_bits is not None else None

    train_data = _build_pyg_dataset(
        train_smiles,
        desc_values[train_idx],
        desc_valid[train_idx],
        fingerprint_bits=train_fp,
    )
    val_data = _build_pyg_dataset(
        val_smiles,
        desc_values[val_idx],
        desc_valid[val_idx],
        fingerprint_bits=val_fp,
    )

    test_data = None
    if test_idx is not None:
        test_smiles = [smiles_list[i] for i in test_idx]
        test_fp = fp_bits[test_idx] if fp_bits is not None else None
        test_data = _build_pyg_dataset(
            test_smiles,
            desc_values[test_idx],
            desc_valid[test_idx],
            fingerprint_bits=test_fp,
        )

    train_loader = make_loader(train_data, config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = make_loader(val_data, config.batch_size, shuffle=False, num_workers=config.num_workers)

    # ------------------------------------------------------------------
    # 8. Create model
    # ------------------------------------------------------------------
    from gt_pyg import GraphTransformerNet
    from gt_pyg.data import get_atom_feature_dim, get_bond_feature_dim

    mc = config.model
    gt_params = inspect.signature(GraphTransformerNet.__init__).parameters
    model_kwargs = dict(
        node_dim_in=get_atom_feature_dim(),
        edge_dim_in=get_bond_feature_dim(),
        num_gt_layers=mc.num_gt_layers,
        hidden_dim=mc.hidden_dim,
        num_heads=mc.num_heads,
        norm=mc.norm,
        gt_aggregators=mc.gt_aggregators,
        aggregators=mc.aggregators,
        dropout=mc.dropout,
        act=mc.act,
        gate=mc.gate,
        qkv_bias=mc.qkv_bias,
        num_tasks=num_descriptors,
        num_head_layers=mc.num_head_layers,
        head_norm=mc.head_norm,
        head_residual=mc.head_residual,
    )
    if "head_dropout" in gt_params and mc.head_dropout is not None:
        model_kwargs["head_dropout"] = mc.head_dropout
    model = GraphTransformerNet(**model_kwargs).to(device)

    if config.rank_alignment.enabled:
        forward_params = inspect.signature(model.forward).parameters
        if "return_latent" not in forward_params:
            raise RuntimeError(
                "Rank-alignment loss requires gt-pyg with "
                "GraphTransformerNet.forward(..., return_latent=True)."
            )

    n_params = model.num_parameters()
    logger.info("Model: %d trainable parameters, num_tasks=%d", n_params, num_descriptors)

    # ------------------------------------------------------------------
    # 9. Optimizer + LR scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )
    scheduler = _make_warmup_cosine_scheduler(optimizer, config.warmup_epochs, config.max_epochs)

    # ------------------------------------------------------------------
    # 9b. Resume from checkpoint (optional)
    # ------------------------------------------------------------------
    start_epoch = 0
    best_val_loss = float("inf")
    best_epoch = 0

    if resume_from is not None:
        resume_path = Path(resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        logger.info("Resuming from checkpoint: %s", resume_path)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "epoch" in ckpt and ckpt["epoch"] is not None:
            start_epoch = ckpt["epoch"] + 1
        if "best_metric" in ckpt and ckpt["best_metric"] is not None:
            best_val_loss = ckpt["best_metric"]
            best_epoch = ckpt.get("epoch", 0)
        logger.info("Resumed at epoch %d (best_val_loss=%.4f)", start_epoch, best_val_loss)

    # ------------------------------------------------------------------
    # 10. Metrics CSV + Training loop
    # ------------------------------------------------------------------
    patience_counter = 0
    start_time = time.time()

    best_ckpt_path = output_dir / "best_checkpoint.pt"
    last_ckpt_path = output_dir / "last_checkpoint.pt"
    metrics_path = output_dir / "metrics.csv"

    logger.info("Starting training: max_epochs=%d  patience=%d  masking_ratio=%.2f",
                config.max_epochs, config.patience, config.masking_ratio)

    epoch = start_epoch
    csv_mode = "a" if resume_from is not None and metrics_path.exists() else "w"
    if csv_mode == "a":
        _prepare_metrics_file(metrics_path, METRICS_FIELDNAMES)

    with open(metrics_path, csv_mode, newline="") as metrics_file:
        metrics_writer = csv.DictWriter(metrics_file, fieldnames=METRICS_FIELDNAMES)
        if csv_mode == "w":
            metrics_writer.writeheader()

        for epoch in range(start_epoch, config.max_epochs):
            lr = scheduler.get_last_lr()[0]

            train_metrics = _train_one_epoch(
                model,
                train_loader,
                optimizer,
                config.masking_ratio,
                device,
                epoch=epoch,
                rank_alignment=config.rank_alignment,
            )
            val_metrics = _validate(
                model,
                val_loader,
                device,
                rank_alignment=config.rank_alignment,
                epoch=epoch,
            )

            elapsed = time.time() - start_time
            metrics_writer.writerow({
                "epoch": epoch,
                "train_loss": f"{train_metrics.main_loss:.6f}",
                "val_loss": f"{val_metrics.loss:.6f}",
                "val_rmse": f"{val_metrics.rmse:.6f}",
                "learning_rate": f"{lr:.2e}",
                "elapsed_seconds": f"{elapsed:.1f}",
                "train_rank_loss": f"{train_metrics.rank_loss:.6f}",
                "train_total_loss": f"{train_metrics.total_loss:.6f}",
                "val_rank_loss": f"{val_metrics.rank_loss:.6f}",
                "val_spearman": f"{val_metrics.spearman:.6f}",
                "val_kendall": f"{val_metrics.kendall:.6f}",
            })
            metrics_file.flush()

            logger.info(
                "Epoch %3d/%d — train_loss=%.4f  train_rank=%.4f  train_total=%.4f  "
                "val_loss=%.4f  val_rmse=%.4f  val_rank=%.4f  lr=%.2e",
                epoch + 1,
                config.max_epochs,
                train_metrics.main_loss,
                train_metrics.rank_loss,
                train_metrics.total_loss,
                val_metrics.loss,
                val_metrics.rmse,
                val_metrics.rank_loss,
                lr,
            )
            if math.isfinite(val_metrics.spearman) or math.isfinite(val_metrics.kendall):
                logger.info(
                    "           rank-alignment metrics — val_spearman=%.4f  val_kendall=%.4f",
                    val_metrics.spearman,
                    val_metrics.kendall,
                )

            # Early stopping
            if val_metrics.loss < best_val_loss:
                best_val_loss = val_metrics.loss
                best_epoch = epoch
                patience_counter = 0

                # Save best checkpoint
                _save_checkpoint(
                    model, optimizer, best_ckpt_path,
                    epoch=epoch,
                    best_metric=best_val_loss,
                    config=config,
                    scaler=scaler,
                    descriptor_names=descriptor_names,
                    num_descriptors=num_descriptors,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    test_idx=test_idx,
                    scheduler=scheduler,
                )
                logger.info("  ↳ New best — saved %s", best_ckpt_path.name)
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    logger.info("Early stopping at epoch %d (patience=%d)", epoch + 1, config.patience)
                    break

            scheduler.step()

    # Save last checkpoint
    _save_checkpoint(
        model, optimizer, last_ckpt_path,
        epoch=epoch,
        best_metric=best_val_loss,
        config=config,
        scaler=scaler,
        descriptor_names=descriptor_names,
        num_descriptors=num_descriptors,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        scheduler=scheduler,
    )

    logger.info(
        "Training complete.  Best val_loss=%.4f at epoch %d",
        best_val_loss, best_epoch + 1,
    )

    # ------------------------------------------------------------------
    # 12. Test RMSE (if test split exists)
    # ------------------------------------------------------------------
    if test_data is not None and best_ckpt_path.exists():
        test_loader = make_loader(test_data, config.batch_size, shuffle=False, num_workers=config.num_workers)

        # Load best checkpoint for test evaluation
        model.load_weights(best_ckpt_path, map_location=device)

        test_metrics = _validate(
            model,
            test_loader,
            device,
            rank_alignment=config.rank_alignment,
        )
        logger.info("Test RMSE (best model): %.4f", test_metrics.rmse)

    # ------------------------------------------------------------------
    # 13. Generate HTML report
    # ------------------------------------------------------------------
    try:
        generate_report(output_dir)
    except Exception:
        logger.warning("Could not generate HTML report", exc_info=True)

    logger.info("Outputs saved to %s", output_dir)
    return best_ckpt_path


# ---------------------------------------------------------------------------
# Checkpoint helper
# ---------------------------------------------------------------------------

def _save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: Path,
    *,
    epoch: int,
    best_metric: float,
    config: PretrainConfig,
    scaler: NaNAwareStandardScaler,
    descriptor_names: List[str],
    num_descriptors: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: Optional[np.ndarray],
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> None:
    """Save checkpoint via gt-pyg's model.save_checkpoint()."""
    model.save_checkpoint(
        path=path,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        best_metric=best_metric,
        extra={
            "scaler_state": scaler.state_dict(),
            "descriptor_names": descriptor_names,
            "descriptor_count": num_descriptors,
            "config": asdict(config),
            "library_versions": _checkpoint_library_versions(),
            "split_indices": {
                "train": train_idx.tolist(),
                "val": val_idx.tolist(),
                "test": test_idx.tolist() if test_idx is not None else None,
            },
        },
    )
