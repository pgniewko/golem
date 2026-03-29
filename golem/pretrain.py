"""Pretraining loop for Graph Transformers on Mordred descriptors.

Uses ``GraphTransformerNet(num_tasks=num_descriptors)`` directly — no wrapper
model.  The built-in ``mu_mlp`` serves as the descriptor prediction head.

Pipeline:
1. Load SMILES
2. Split into train / val / (test) at the parent-molecule level
3. (Optional) Enumerate isoforms within each split only
4. Compute descriptor targets + validity masks
5. Fit ``NaNAwareStandardScaler`` on **train only**
6. Transform all splits + winsorise
7. Build PyG datasets via ``get_tensor_data``
8. Train with masked MSE (15% random mask ∩ validity mask)
9. Early stopping on validation objective
10. Save checkpoint with scaler, descriptor names, config
"""

from __future__ import annotations

import csv
import inspect
import json
import logging
import math
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from rdkit import Chem
import torch.nn.functional as F
import yaml
from golem.config import (
    ECFPLatentAlignmentConfig,
    PretrainConfig,
    validate_pretrain_config,
)
from golem.descriptors import (
    NaNAwareStandardScaler,
    compute_descriptor_targets,
)
from golem.ecfp_latent_alignment import (
    compute_alignment_batch,
    compute_alignment_metrics,
    compute_fingerprints,
)
from golem.isoforms import enumerate_isoforms_batch
from golem.report import generate_report
from golem.utils import (
    load_smiles,
    make_loader,
    seed_everything,
    split_data,
)

logger = logging.getLogger(__name__)

_BATCH_NORM_NAMES = {"bn", "batchnorm", "batch_norm"}

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


def _format_optional_metric(value: float) -> str:
    """Format finite optional metrics and leave inactive ones blank."""
    return f"{value:.6f}" if math.isfinite(value) else ""


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
    descriptor_loss_weight: float,
    device: torch.device,
) -> tuple[float, float, float]:
    """Run one training epoch. Returns objective, descriptor, and alignment losses."""
    model.train()
    total_descriptor_se = 0.0
    total_descriptor_count = 0
    total_alignment_loss = 0.0
    n_alignment_batches = 0
    saw_optimization_step = False
    alignment_cfg: ECFPLatentAlignmentConfig | None = getattr(
        loader, "ecfp_latent_alignment", None
    )
    for batch in loader:
        batch = batch.to(device)
        targets = batch.y  # [B, D]
        valid_mask = batch.y_mask.bool()  # [B, D]

        if alignment_cfg is not None:
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
        final_mask = (torch.rand_like(targets) < masking_ratio).bool() & valid_mask

        masked_count = int(final_mask.sum().item())
        if masked_count == 0 and valid_mask.sum() > 0:
            final_mask = valid_mask
            masked_count = int(final_mask.sum().item())

        descriptor_loss = pred.sum() * 0.0
        if masked_count > 0:
            masked_diff = pred[final_mask] - targets[final_mask]
            descriptor_loss = F.mse_loss(pred[final_mask], targets[final_mask])
            total_descriptor_se += masked_diff.pow(2).sum().item()
            total_descriptor_count += masked_count

        alignment_loss = pred.sum() * 0.0
        has_alignment_pairs = False
        if z is not None and alignment_cfg is not None:
            alignment_loss, d_fp, _ = compute_alignment_batch(batch, z, alignment_cfg)
            has_alignment_pairs = isinstance(d_fp, torch.Tensor) and d_fp.numel() > 0
            if has_alignment_pairs:
                total_alignment_loss += alignment_loss.item()
                n_alignment_batches += 1

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
    if total_descriptor_count > 0:
        descriptor_loss_value = total_descriptor_se / total_descriptor_count
    elif saw_optimization_step:
        descriptor_loss_value = 0.0

    alignment_loss_value = (
        total_alignment_loss / n_alignment_batches if n_alignment_batches else math.nan
    )

    objective_terms: list[float] = []
    if math.isfinite(descriptor_loss_value):
        objective_terms.append(descriptor_loss_value * descriptor_loss_weight)
    if math.isfinite(alignment_loss_value) and alignment_cfg is not None:
        objective_terms.append(alignment_loss_value * alignment_cfg.weight)
    objective_loss = float(sum(objective_terms)) if objective_terms else math.nan

    return objective_loss, descriptor_loss_value, alignment_loss_value


@torch.no_grad()
def _validate(
    model: torch.nn.Module,
    loader,
    descriptor_loss_weight: float,
    device: torch.device,
) -> tuple[float, float, float, float, float, float]:
    """Validate and optionally compute alignment metrics."""
    model.eval()
    total_se = 0.0
    total_count = 0
    alignment_losses: List[float] = []
    spearmans: List[float] = []
    kendalls: List[float] = []
    alignment_cfg: ECFPLatentAlignmentConfig | None = getattr(
        loader, "ecfp_latent_alignment", None
    )

    for batch in loader:
        batch = batch.to(device)
        targets = batch.y
        valid_mask = batch.y_mask.bool()

        if alignment_cfg is not None:
            pred, _, z = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch=batch.batch,
                zero_var=True,
                return_latent=True,
            )
            alignment_loss, d_fp, d_z = compute_alignment_batch(
                batch,
                z,
                alignment_cfg,
                deterministic_pairs=True,
            )
            has_alignment_pairs = isinstance(d_fp, torch.Tensor) and d_fp.numel() > 0
            if has_alignment_pairs:
                alignment_losses.append(alignment_loss.item())
            if has_alignment_pairs and alignment_cfg.log_rank_metrics:
                spearman, kendall = compute_alignment_metrics(d_fp, d_z)
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

    descriptor_loss = math.nan
    rmse = math.nan
    if total_count > 0:
        descriptor_loss = total_se / total_count
        rmse = math.sqrt(descriptor_loss)

    alignment_loss = float(np.mean(alignment_losses)) if alignment_losses else math.nan
    objective_terms: list[float] = []
    if math.isfinite(descriptor_loss):
        objective_terms.append(descriptor_loss * descriptor_loss_weight)
    if math.isfinite(alignment_loss) and alignment_cfg is not None:
        objective_terms.append(alignment_loss * alignment_cfg.weight)
    objective_loss = float(sum(objective_terms)) if objective_terms else math.nan

    return (
        objective_loss,
        descriptor_loss,
        rmse,
        alignment_loss,
        float(np.mean(spearmans)) if spearmans else math.nan,
        float(np.mean(kendalls)) if kendalls else math.nan,
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


def _canonicalize_parent_smiles(parent_smiles: List[str]) -> List[str]:
    """Canonicalize and deduplicate parent SMILES before splitting."""
    canonical_smiles: List[str] = []
    seen_smiles: set[str] = set()
    n_collapsed = 0

    for smi in parent_smiles:
        canonical_smi = smi
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
                canonical_smi = Chem.MolToSmiles(mol, canonical=True)
            except Exception:
                canonical_smi = smi

        if canonical_smi in seen_smiles:
            n_collapsed += 1
            continue

        seen_smiles.add(canonical_smi)
        canonical_smiles.append(canonical_smi)

    if n_collapsed:
        logger.info(
            "Collapsed %d duplicate/synonymous parent SMILES before splitting",
            n_collapsed,
        )

    return canonical_smiles


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
    parent_smiles = _canonicalize_parent_smiles(parent_smiles)
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


def _validate_training_batch_configuration(
    train_size: int,
    config: PretrainConfig,
) -> None:
    """Reject batch-norm training setups that would create singleton batches."""
    if config.model.norm.lower() not in _BATCH_NORM_NAMES:
        return
    if train_size == 0:
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


def pretrain(
    smiles_path: str,
    config: PretrainConfig,
    output_dir: str,
    subsample: Optional[float] = None,
    verbose: bool = False,
) -> Path:
    """Full pretraining pipeline.

    Args:
        smiles_path: Path to SMILES file (``.smi`` or ``.csv``).
        config: Resolved ``PretrainConfig``.
        output_dir: Directory for checkpoints, logs, metrics.
        subsample: If set, randomly subsample this fraction of SMILES
            before processing (e.g. 0.1 for 10%). Explicit function
            arguments override ``config.subsample`` and must be in ``(0, 1]``.
        verbose: If True, show DEBUG-level logs on console.

    Returns:
        Path to the best checkpoint file.
    """
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
    effective_subsample = subsample if subsample is not None else config.subsample
    resolved_config = replace(config, subsample=effective_subsample)
    validate_pretrain_config(resolved_config)

    if config.max_epochs > 0 and config.warmup_epochs >= config.max_epochs:
        logger.warning(
            "warmup_epochs (%d) >= max_epochs (%d): model will only warm up, never decay",
            config.warmup_epochs, config.max_epochs,
        )

    # Save resolved config (convert tuples to lists for safe_load compatibility)
    config_dict = json.loads(json.dumps(asdict(resolved_config)))
    with open(output_dir / "resolved_config.yaml", "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    # ------------------------------------------------------------------
    # 1. Load SMILES
    # ------------------------------------------------------------------
    smiles_list = load_smiles(smiles_path)
    logger.info("Loaded %d SMILES from %s", len(smiles_list), smiles_path)

    if effective_subsample is not None and 0 < effective_subsample < 1:
        rng = np.random.RandomState(config.seed)
        n_sub = max(1, int(len(smiles_list) * effective_subsample))
        indices = rng.choice(len(smiles_list), size=n_sub, replace=False)
        smiles_list = [smiles_list[i] for i in sorted(indices)]
        logger.info(
            "Subsampled to %d SMILES (%.1f%%)",
            len(smiles_list),
            effective_subsample * 100,
        )

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
    # 4. Compute descriptor targets
    # ------------------------------------------------------------------
    logger.info(
        "Computing descriptor targets (2D=%s, 3D=%s) …",
        config.descriptors.include_2d_targets,
        config.descriptors.include_3d_targets,
    )
    desc_values, desc_valid, descriptor_names = compute_descriptor_targets(
        smiles_list,
        config.descriptors,
        config.conformers,
        seed=config.seed,
    )

    num_descriptors = desc_values.shape[1]
    logger.info("Descriptor matrix: %d molecules × %d descriptors", desc_values.shape[0], num_descriptors)

    fp_bits = None
    alignment_cfg = config.ecfp_latent_alignment
    if alignment_cfg.enabled:
        fp_bits = compute_fingerprints(smiles_list, alignment_cfg)
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
    _validate_training_batch_configuration(len(train_data), config)
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
    if mc.head_dropout is not None:
        if "head_dropout" not in gt_params:
            raise RuntimeError(
                "model.head_dropout is set, but the installed gt-pyg "
                "GraphTransformerNet does not accept head_dropout."
            )
        model_kwargs["head_dropout"] = mc.head_dropout
    model = GraphTransformerNet(**model_kwargs).to(device)
    if alignment_cfg.enabled and "return_latent" not in inspect.signature(model.forward).parameters:
        raise RuntimeError(
            "ECFP-latent alignment requires gt-pyg with "
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
    # 9b. Tracking state
    # ------------------------------------------------------------------
    best_val_objective = float("inf")
    best_epoch = -1

    train_loader = make_loader(
        train_data,
        config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = make_loader(
        val_data,
        config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    train_loader.ecfp_latent_alignment = alignment_cfg if alignment_cfg.enabled else None
    val_loader.ecfp_latent_alignment = alignment_cfg if alignment_cfg.enabled else None

    # ------------------------------------------------------------------
    # 10. Metrics CSV + Training loop
    # ------------------------------------------------------------------
    patience_counter = 0
    start_time = time.time()

    best_ckpt_path = output_dir / "best_checkpoint.pt"
    last_ckpt_path = output_dir / "last_checkpoint.pt"
    metrics_path = output_dir / "metrics.csv"
    last_val_loss = math.nan
    last_completed_epoch: int | None = None
    last_saved_epoch: int | None = None

    def _save_last_checkpoint(epoch_to_save: int) -> None:
        nonlocal last_saved_epoch
        _save_checkpoint(
            model,
            last_ckpt_path,
            epoch=epoch_to_save,
            best_metric=best_val_objective if math.isfinite(best_val_objective) else None,
            config=resolved_config,
            scaler=scaler,
            descriptor_names=descriptor_names,
            num_descriptors=num_descriptors,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
        )
        last_saved_epoch = epoch_to_save

    logger.info("Starting training: max_epochs=%d  patience=%d  masking_ratio=%.2f",
                config.max_epochs, config.patience, config.masking_ratio)

    epoch = 0
    try:
        with open(metrics_path, "w", newline="") as metrics_file:
            metrics_writer = csv.writer(metrics_file)
            metrics_writer.writerow(METRICS_FIELDNAMES)
            metrics_file.flush()

            for epoch in range(config.max_epochs):
                lr = scheduler.get_last_lr()[0]
                (
                    train_loss,
                    train_descriptor_loss,
                    train_alignment_loss,
                ) = _train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    config.masking_ratio,
                    config.descriptors.loss_weight,
                    device,
                )
                (
                    val_loss,
                    val_descriptor_loss,
                    val_rmse,
                    val_alignment_loss,
                    val_alignment_spearman,
                    val_alignment_kendall,
                ) = _validate(
                    model,
                    val_loader,
                    config.descriptors.loss_weight,
                    device,
                )
                last_val_loss = val_loss
                last_completed_epoch = epoch

                elapsed = time.time() - start_time
                row = [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{val_loss:.6f}",
                    f"{train_descriptor_loss:.6f}",
                    f"{val_descriptor_loss:.6f}",
                    f"{val_rmse:.6f}",
                    f"{lr:.2e}",
                    f"{elapsed:.1f}",
                    _format_optional_metric(train_alignment_loss),
                    _format_optional_metric(val_alignment_loss),
                    _format_optional_metric(val_alignment_spearman),
                    _format_optional_metric(val_alignment_kendall),
                ]
                metrics_writer.writerow(row)
                metrics_file.flush()

                summary_parts = [
                    f"train_loss={train_loss:.4f}",
                    f"val_loss={val_loss:.4f}",
                    f"train_desc={train_descriptor_loss:.4f}",
                    f"val_desc={val_descriptor_loss:.4f}",
                    f"val_rmse={val_rmse:.4f}",
                    f"lr={lr:.2e}",
                ]
                if alignment_cfg.enabled and math.isfinite(train_alignment_loss):
                    summary_parts.append(f"train_align={train_alignment_loss:.4f}")
                if alignment_cfg.enabled and math.isfinite(val_alignment_loss):
                    summary_parts.append(f"val_align={val_alignment_loss:.4f}")
                logger.info(
                    "Epoch %3d/%d — %s",
                    epoch + 1,
                    config.max_epochs,
                    "  ".join(summary_parts),
                )
                if math.isfinite(val_alignment_spearman) or math.isfinite(val_alignment_kendall):
                    logger.info(
                        "           val_alignment_spearman=%.4f  val_alignment_kendall=%.4f",
                        val_alignment_spearman,
                        val_alignment_kendall,
                    )

                # Early stopping
                if math.isfinite(val_loss) and val_loss < best_val_objective:
                    best_val_objective = val_loss
                    best_epoch = epoch
                    patience_counter = 0

                    # Save best checkpoint
                    _save_checkpoint(
                        model, best_ckpt_path,
                        epoch=epoch,
                        best_metric=best_val_objective,
                        config=resolved_config,
                        scaler=scaler,
                        descriptor_names=descriptor_names,
                        num_descriptors=num_descriptors,
                        train_idx=train_idx,
                        val_idx=val_idx,
                        test_idx=test_idx,
                    )
                    logger.info("  ↳ New best — saved %s", best_ckpt_path.name)
                else:
                    if not math.isfinite(val_loss):
                        logger.warning(
                            "Validation objective is non-finite at epoch %d; skipping best-checkpoint update",
                            epoch + 1,
                        )
                    patience_counter += 1

                _save_last_checkpoint(epoch)

                if patience_counter >= config.patience:
                    logger.info("Early stopping at epoch %d (patience=%d)", epoch + 1, config.patience)
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
        _save_checkpoint(
            model, best_ckpt_path,
            epoch=epoch,
            best_metric=(
                best_val_objective
                if math.isfinite(best_val_objective)
                else last_val_loss if math.isfinite(last_val_loss) else math.nan
            ),
            config=resolved_config,
            scaler=scaler,
            descriptor_names=descriptor_names,
            num_descriptors=num_descriptors,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
        )

    logger.info(
        "Training complete.  Best val_objective=%.4f at epoch %d",
        best_val_objective,
        best_epoch + 1 if best_epoch >= 0 else epoch + 1,
    )

    # ------------------------------------------------------------------
    # 12. Test RMSE (if test split exists)
    # ------------------------------------------------------------------
    if test_data is not None and best_ckpt_path.exists():
        test_loader = make_loader(
            test_data,
            config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
        test_loader.ecfp_latent_alignment = alignment_cfg if alignment_cfg.enabled else None

        # Load best checkpoint for test evaluation
        model.load_weights(best_ckpt_path, map_location=device)

        (
            test_loss,
            test_descriptor_loss,
            test_rmse,
            test_alignment_loss,
            test_alignment_spearman,
            test_alignment_kendall,
        ) = _validate(
            model,
            test_loader,
            config.descriptors.loss_weight,
            device,
        )
        logger.info(
            "Test objective loss=%.4f  descriptor_loss=%.4f  rmse=%.4f",
            test_loss,
            test_descriptor_loss,
            test_rmse,
        )
        if alignment_cfg.enabled:
            logger.info(
                "Test alignment loss=%.4f",
                test_alignment_loss,
            )
            if math.isfinite(test_alignment_spearman) or math.isfinite(test_alignment_kendall):
                logger.info(
                    "Test alignment rank metrics: spearman=%.4f  kendall=%.4f",
                    test_alignment_spearman,
                    test_alignment_kendall,
                )

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
    path: Path,
    *,
    epoch: int,
    best_metric: float | None,
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
