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
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install gt-pyg first.
# Local development checkout:
python -m pip install -e /path/to/gt-pyg

# Or install gt-pyg from GitHub instead:
# python -m pip install "gt-pyg @ git+https://github.com/pgniewko/gt-pyg.git"

# Install golem (editable)
python -m pip install -e .

# (Optional) Install dev dependencies for tests and notebooks
python -m pip install -e ".[dev]"
```

If you are working in this sibling-checkout layout:

```bash
cd /Users/pawelgniewek/projects/golem
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ../gt-pyg
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

### Quick test run (~1 minute)

```bash
golem pretrain \
  --smiles data/openadmet/train_test_smiles.smi \
  --output experiments/test_pretrain \
  --max-epochs 10 \
  --subsample 0.1 \
  --no-isoforms
```

### Production run

```bash
golem pretrain \
  --smiles data/openadmet/train_test_smiles.smi \
  --config configs/pretrain_openadmet.yaml \
  --output experiments/pretrain
```

### CLI options

| Flag | Description | Default |
|------|-------------|---------|
| `--smiles` | Path to SMILES file (`.smi` or `.csv`) | Required |
| `--config` | Path to YAML config file | Built-in defaults |
| `--output` | Output directory for checkpoints and logs | Required |
| `--max-epochs` | Override max training epochs | 500 |
| `--batch-size` | Override batch size | 128 |
| `--lr` | Override learning rate | 1e-4 |
| `--subsample` | Subsample fraction (e.g. 0.1 for 10%) | None (use all) |
| `--seed` | Override random seed | 42 |
| `--no-isoforms` | Disable isoform enumeration | Enabled |
| `--verbose` | Show DEBUG-level logs on console | Disabled |
| `--resume` | Path to checkpoint to resume training from | None |

### What pretraining produces

After a run completes, the output directory contains:

```
experiments/pretrain/
  best_checkpoint.pt        # Best model by validation loss
  last_checkpoint.pt        # Last epoch (for potential resuming)
  resolved_config.yaml      # Full resolved config used for the run
  metrics.csv               # Per-epoch: train_loss, val_loss, val_rmse, lr
  pretrain.log              # Full log output
```

## Generating Reports

After a pretraining run completes, an HTML report with training curves is automatically generated in the output directory. You can also regenerate or create a report from any existing experiment directory:

```bash
golem report experiments/pretrain
```

This reads `metrics.csv` and `resolved_config.yaml` from the experiment directory and produces a single-file HTML dashboard (`pretrain_report.html`) with:

- Training & validation loss curves
- Validation RMSE curve
- Learning rate schedule
- Train vs val loss gap
- Summary cards (best epoch, best val loss, elapsed time, architecture)
- Epoch-by-epoch table with the best row highlighted

Note: the generated HTML references Chart.js from a CDN, so it is not fully offline/self-contained.

To write the report to a custom path:

```bash
golem report experiments/pretrain --output path/to/report.html
```

## Running Tests

```bash
pytest tests/ -v
```

The tests cover isoform enumeration, Mordred descriptor computation, the NaN-aware scaler, config loading, data splitting, and SMILES file loading. Note: descriptor tests require `mordredcommunity` and may take a few seconds.

### Key module responsibilities

| Module | What it does |
|--------|-------------|
| `cli.py` | Parses CLI args, merges config, calls `pretrain()` |
| `config.py` | Defines `PretrainConfig` dataclass tree; merges defaults / YAML / CLI |
| `isoforms.py` | Enumerates tautomers, protonation states, and neutralized forms per molecule |
| `descriptors.py` | Computes Mordred 2D descriptors; provides `NaNAwareStandardScaler` |
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
| Production config | `configs/pretrain_openadmet.yaml` |
| Pretraining pipeline flow | `pretrain.py` module docstring |
