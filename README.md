# Golem

Descriptor pretraining for Graph Transformers on molecular descriptors. Inspired by [CheMeleon](https://github.com/JacksonBurns/chemeleon), with improvements including NaN-aware validity masking and scaling, and isoform enumeration for data augmentation.

Golem pretrains a [gt-pyg](https://github.com/pgniewko/gt-pyg) `GraphTransformerNet` backbone to predict Mordred 2D molecular descriptors, then the pretrained weights transfer to downstream property-prediction tasks via fine-tuning notebooks.

## Installation

### Prerequisites

- Python 3.10+
- pip (included with Python)
- A checkout of the [gt-pyg](https://github.com/pgniewko/gt-pyg) package

`golem` imports `gt_pyg` at runtime, so `gt-pyg` must be installed in the same environment before running `golem pretrain`.

### Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install gt-pyg first.
# Local development checkout:
python -m pip install -e /path/to/gt-pyg

# Or install gt-pyg from GitHub instead:
# python -m pip install "gt-pyg @ git+https://github.com/pgniewko/gt-pyg.git"

# Install golem (editable)
python -m pip install -e .

# (Optional) Install dev dependencies
python -m pip install -e ".[dev]"
```

If you are working in this sibling-checkout layout:

```bash
cd /Users/pawelgniewek/projects/golem
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ../gt-pyg
# Optional: install dev dependencies
python -m pip install -e ".[dev]"
```

### Verify installation

```bash
golem --help
# Should show 'pretrain' and 'report' commands

python -c "from gt_pyg import GraphTransformerNet; print('gt-pyg OK')"
python -c "from golem.config import PretrainConfig; print('golem OK')"
```

## Running Pretraining

### Quick smoke run (~1 minute)

```bash
golem pretrain \
  --smiles data/openadmet/expansion_rx/train_test_smiles.smi \
  --output experiments/test_pretrain \
  --max-epochs 10 \
  --subsample 0.1 \
  --no-isoforms
```

### Production run

```bash
golem pretrain \
  --smiles data/openadmet/expansion_rx/train_test_smiles.smi \
  --config configs/golem-2d.yaml \
  --output experiments/pretrain
```

Config files in `configs/` are intended to contain overrides over the defaults in
`golem.config.PretrainConfig`, not a full copy of every setting.

Optional ECFP-latent alignment can be enabled in YAML:

```yaml
ecfp_latent_alignment:
  enabled: true
```

Optional 3D descriptor targets can be enabled in YAML:

```yaml
descriptors:
  include_3d_targets: true
```

Set `descriptors.include_2d_targets: false` together with `descriptors.include_3d_targets: true` to train on 3D descriptors only. If you want the run to optimize only the ECFP-latent alignment objective while still keeping descriptor heads active, set `descriptors.loss_weight: 0.0` and enable `ecfp_latent_alignment`. ElectroShape uses fixed `gasteiger` charges, conformer embedding is fixed to `ETKDGv3`, conformer optimization uses fixed `MMFF` with `UFF` fallback, and the single lowest-energy conformer from `conformers.n_generate` attempts is used for 3D descriptors. If conformer generation or a 3D descriptor family fails, the molecule is kept and the affected 3D targets are masked the same way invalid 2D descriptor entries are masked. 3D descriptor columns that are invalid for every molecule are dropped across the dataset, and the run fails if no descriptor columns remain.

### CLI options

| Flag | Description | Default |
|------|-------------|---------|
| `--smiles` | Path to SMILES file (`.smi` or `.csv`) | Required |
| `--config` | Path to YAML config file | Built-in defaults |
| `--output` | Output directory for checkpoints and logs | Required |
| `--max-epochs` | Override max training epochs | 500 |
| `--batch-size` | Override batch size | 128 |
| `--lr` | Override learning rate | 1e-4 |
| `--num-workers` | Override PyG data loading workers | 0 |
| `--subsample` | Subsample fraction (e.g. 0.1 for 10%) | None (use all) |
| `--seed` | Override random seed | 42 |
| `--no-isoforms` | Disable isoform enumeration | Enabled |
| `--verbose` | Show DEBUG-level logs on console | Disabled |

### What pretraining produces

After a run completes, the output directory contains:

```
experiments/pretrain/
  best_checkpoint.pt        # Best model by validation objective
  last_checkpoint.pt        # Most recent completed-epoch weights
  resolved_config.yaml      # Full resolved config used for the run
  pretrain_report.html      # HTML dashboard with training curves and metrics (not tracked)
  metrics.csv               # Per-epoch objective, descriptor, RMSE, LR, and optional alignment metrics (not tracked)
  pretrain.log              # Full log output (not tracked)
```

## Generating Reports

After a pretraining run completes, an HTML report with training curves is automatically generated in the output directory. You can also regenerate or create a report from any existing experiment directory:

```bash
golem report experiments/pretrain
```

This reads `metrics.csv` and `resolved_config.yaml` from the experiment directory and produces a single-file HTML dashboard (`pretrain_report.html`) with:

- Training & validation objective curves
- Training & validation descriptor-loss curves
- Validation RMSE curve
- Learning rate schedule
- Optional ECFP-latent alignment chart when those metrics are present
- Summary cards (best epoch, best val loss, elapsed time, architecture)
- Epoch-by-epoch table with the best row highlighted

Note: the generated HTML references Chart.js from a CDN, so it is not fully offline/self-contained.

To write the report to a custom path:

```bash
golem report experiments/pretrain --output path/to/report.html
```

### Key module responsibilities

| Module | What it does |
|--------|-------------|
| `cli.py` | Parses CLI args, merges config, calls `pretrain()` |
| `config.py` | Defines `PretrainConfig` dataclass tree; merges defaults / YAML overrides / CLI |
| `conformers.py` | Builds the lowest-energy RDKit conformer used for optional 3D descriptor targets |
| `isoforms.py` | Enumerates tautomers, protonation states, and neutralized forms per molecule |
| `descriptors.py` | Computes 2D/3D descriptor targets; provides `NaNAwareStandardScaler` |
| `pretrain.py` | Orchestrates the full pipeline: load SMILES &rarr; isoforms &rarr; descriptors &rarr; split &rarr; scale &rarr; train &rarr; checkpoint |
| `utils.py` | Shared utilities: seeding, train/val/test splitting, PyG DataLoader creation, SMILES file loading |

### Where things live

| Looking for... | Go to |
|----------------|-------|
| How the model is constructed | `pretrain.py` model creation section |
| The masked MSE pretraining loss | `pretrain.py:_train_one_epoch()` |
| NaN handling / validity masking | `descriptors.py:compute_mordred_descriptors()` |
| Scaler fit (train-only, no leakage) | `pretrain.py:pretrain()` step 5 |
| Config defaults | `config.py` dataclass definitions |
| Production config overrides | `configs/golem-2d.yaml` |
| Pretraining pipeline flow | `pretrain.py` module docstring |
