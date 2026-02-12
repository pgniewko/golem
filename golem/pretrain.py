"""MDAE pretraining loop for Graph Transformers on Mordred descriptors.

Uses ``GraphTransformerNet(num_tasks=num_descriptors)`` directly — no wrapper
model.  The built-in ``mu_mlp`` serves as the descriptor prediction head.

Pipeline:
1. Load SMILES
2. (Optional) Enumerate isoforms
3. Compute Mordred 2D descriptors + validity masks
4. Split into train / val / (test) — **before** fitting scaler
5. Fit ``NaNAwareStandardScaler`` on **train only**
6. Transform all splits + winsorise
7. Build PyG datasets via ``get_tensor_data``
8. Train with masked MSE (15% random mask ∩ validity mask)
9. Early stopping on validation loss
10. Save checkpoint with scaler, descriptor names, config
"""

from __future__ import annotations

import csv
import logging
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from golem.config import PretrainConfig
from golem.descriptors import NaNAwareStandardScaler, compute_mordred_descriptors
from golem.isoforms import enumerate_isoforms_batch
from golem.report import generate_report
from golem.utils import load_smiles, make_loader, seed_everything, split_data

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(output_dir: Path) -> None:
    """Configure logging to both stdout (INFO) and file (DEBUG)."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Remove existing handlers (avoid duplicates on re-runs)
    for h in root.handlers[:]:
        root.removeHandler(h)

    # Console handler (INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s", datefmt="%H:%M:%S"))
    root.addHandler(ch)

    # File handler (DEBUG)
    fh = logging.FileHandler(output_dir / "pretrain.log", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s"))
    root.addHandler(fh)


# ---------------------------------------------------------------------------
# LR schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def _warmup_cosine_lr(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    warmup_epochs: int,
    max_epochs: int,
    base_lr: float,
) -> float:
    """Apply linear warmup then cosine decay; returns current LR."""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(max_epochs - warmup_epochs, 1)
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ---------------------------------------------------------------------------
# Single-epoch routines
# ---------------------------------------------------------------------------

def _train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    masking_ratio: float,
    device: torch.device,
) -> float:
    """Run one training epoch.  Returns mean loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        targets = batch.y  # [B, D]
        valid_mask = batch.y_mask.bool()  # [B, D]

        pred, _log_var = model(
            batch.x, batch.edge_index, batch.edge_attr,
            batch=batch.batch, zero_var=True,
        )

        # Random 15% mask ∩ validity mask
        rand_mask = (torch.rand_like(targets) < masking_ratio).bool()
        final_mask = rand_mask & valid_mask

        if final_mask.sum() == 0:
            # Extremely rare fallback: use all valid positions
            final_mask = valid_mask

        if final_mask.sum() == 0:
            continue  # skip batch if no valid positions at all

        loss = F.mse_loss(pred[final_mask], targets[final_mask])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def _validate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate.  Returns (mean_mse_loss, rmse) on valid positions (no random mask)."""
    model.eval()
    total_se = 0.0
    total_count = 0

    for batch in loader:
        batch = batch.to(device)
        targets = batch.y
        valid_mask = batch.y_mask.bool()

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
    return mse, rmse


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def _build_pyg_dataset(
    smiles_list: List[str],
    descriptor_values: np.ndarray,
    validity_mask: np.ndarray,
) -> list:
    """Build PyG Data objects with graph features + descriptor targets.

    Uses ``get_tensor_data`` from gt-pyg for graph featurisation, then
    overwrites ``data.y`` and ``data.y_mask`` with descriptor targets.
    """
    from gt_pyg import get_tensor_data

    num_descriptors = descriptor_values.shape[1]

    logger.info("Building PyG graph features for %d molecules …", len(smiles_list))
    data_list = get_tensor_data(smiles_list, y=None, gnm=True)

    # Overwrite y and y_mask with actual descriptor data.
    # Store as [1, D] so that PyG batching concatenates to [B, D]
    # (matching the stacking behavior of data.y).
    for i, data in enumerate(data_list):
        data.y = torch.tensor(descriptor_values[i], dtype=torch.float32).unsqueeze(0)
        data.y_mask = torch.tensor(validity_mask[i], dtype=torch.float32).unsqueeze(0)

    return data_list


# ---------------------------------------------------------------------------
# Main pretrain function
# ---------------------------------------------------------------------------

def pretrain(
    smiles_path: str,
    config: PretrainConfig,
    output_dir: str,
    subsample: Optional[float] = None,
) -> Path:
    """Full MDAE pretraining pipeline.

    Args:
        smiles_path: Path to SMILES file (``.smi`` or ``.csv``).
        config: Resolved ``PretrainConfig``.
        output_dir: Directory for checkpoints, logs, metrics.
        subsample: If set, randomly subsample this fraction of SMILES
            before processing (e.g. 0.1 for 10%).

    Returns:
        Path to the best checkpoint file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    seed_everything(config.seed)

    # Save resolved config
    with open(output_dir / "resolved_config.yaml", "w") as f:
        yaml.dump(asdict(config), f, default_flow_style=False, sort_keys=False)

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
    # 2. Enumerate isoforms (optional)
    # ------------------------------------------------------------------
    if config.isoforms.enabled:
        logger.info("Enumerating isoforms …")
        iso_map = enumerate_isoforms_batch(smiles_list, config.isoforms)
        # Flatten: one entry per isoform
        all_smiles: List[str] = []
        for parent_smi in smiles_list:
            all_smiles.extend(iso_map[parent_smi])
        # Deduplicate globally (keep order)
        seen = set()
        unique_smiles: List[str] = []
        for smi in all_smiles:
            if smi not in seen:
                seen.add(smi)
                unique_smiles.append(smi)
        logger.info(
            "After isoform enumeration + dedup: %d → %d unique SMILES",
            len(smiles_list), len(unique_smiles),
        )
        smiles_list = unique_smiles
    else:
        logger.info("Isoform enumeration disabled")

    # ------------------------------------------------------------------
    # 3. Compute Mordred descriptors
    # ------------------------------------------------------------------
    logger.info("Computing Mordred descriptors …")
    desc_values, desc_valid, descriptor_names = compute_mordred_descriptors(smiles_list)
    num_descriptors = desc_values.shape[1]
    logger.info("Descriptor matrix: %d molecules × %d descriptors", desc_values.shape[0], num_descriptors)

    # ------------------------------------------------------------------
    # 4. Split BEFORE fitting scaler
    # ------------------------------------------------------------------
    n = len(smiles_list)
    splits = split_data(n, config.split_ratios, seed=config.seed)

    if len(splits) == 3:
        train_idx, val_idx, test_idx = splits
        logger.info("Split: train=%d  val=%d  test=%d", len(train_idx), len(val_idx), len(test_idx))
    else:
        train_idx, val_idx = splits
        test_idx = None
        logger.info("Split: train=%d  val=%d  (no test)", len(train_idx), len(val_idx))

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

    train_data = _build_pyg_dataset(train_smiles, desc_values[train_idx], desc_valid[train_idx])
    val_data = _build_pyg_dataset(val_smiles, desc_values[val_idx], desc_valid[val_idx])

    test_data = None
    if test_idx is not None:
        test_smiles = [smiles_list[i] for i in test_idx]
        test_data = _build_pyg_dataset(test_smiles, desc_values[test_idx], desc_valid[test_idx])

    train_loader = make_loader(train_data, config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = make_loader(val_data, config.batch_size, shuffle=False, num_workers=config.num_workers)

    # ------------------------------------------------------------------
    # 8. Create model
    # ------------------------------------------------------------------
    from gt_pyg import GraphTransformerNet
    from gt_pyg.data import get_atom_feature_dim, get_bond_feature_dim

    mc = config.model
    model = GraphTransformerNet(
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
    ).to(device)

    n_params = model.num_parameters()
    logger.info("Model: %d trainable parameters, num_tasks=%d", n_params, num_descriptors)

    # ------------------------------------------------------------------
    # 9. Optimizer
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )

    # ------------------------------------------------------------------
    # 10. Metrics CSV
    # ------------------------------------------------------------------
    metrics_path = output_dir / "metrics.csv"
    metrics_file = open(metrics_path, "w", newline="")
    metrics_writer = csv.writer(metrics_file)
    metrics_writer.writerow(["epoch", "train_loss", "val_loss", "val_rmse", "learning_rate", "elapsed_seconds"])

    # ------------------------------------------------------------------
    # 11. Training loop
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()

    best_ckpt_path = output_dir / "best_checkpoint.pt"
    last_ckpt_path = output_dir / "last_checkpoint.pt"

    logger.info("Starting training: max_epochs=%d  patience=%d  masking_ratio=%.2f",
                config.max_epochs, config.patience, config.masking_ratio)

    for epoch in range(config.max_epochs):
        lr = _warmup_cosine_lr(optimizer, epoch, config.warmup_epochs, config.max_epochs, config.lr)

        train_loss = _train_one_epoch(model, train_loader, optimizer, config.masking_ratio, device)
        val_loss, val_rmse = _validate(model, val_loader, device)

        elapsed = time.time() - start_time
        metrics_writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_rmse:.6f}", f"{lr:.2e}", f"{elapsed:.1f}"])
        metrics_file.flush()

        logger.info(
            "Epoch %3d/%d — train_loss=%.4f  val_loss=%.4f  val_rmse=%.4f  lr=%.2e",
            epoch + 1, config.max_epochs, train_loss, val_loss, val_rmse, lr,
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
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
            )
            logger.info("  ↳ New best — saved %s", best_ckpt_path.name)
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch + 1, config.patience)
                break

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
    )

    metrics_file.close()

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

        test_mse, test_rmse = _validate(model, test_loader, device)
        logger.info("Test RMSE (best model): %.4f", test_rmse)

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
) -> None:
    """Save checkpoint via gt-pyg's model.save_checkpoint()."""
    model.save_checkpoint(
        path=path,
        optimizer=optimizer,
        epoch=epoch,
        best_metric=best_metric,
        extra={
            "scaler_state": scaler.state_dict(),
            "descriptor_names": descriptor_names,
            "descriptor_count": num_descriptors,
            "config": asdict(config),
            "split_indices": {
                "train": train_idx.tolist(),
                "val": val_idx.tolist(),
                "test": test_idx.tolist() if test_idx is not None else None,
            },
        },
    )
