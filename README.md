# Golem

Masked Descriptor Autoencoding pretraining for Graph Transformers on molecular descriptors. Inspired by [CheMeleon](https://github.com/JacksonBurns/chemeleon), with improvements including NaN-aware validity masking, isoform enumeration for data augmentation, and a clean train-only scaler.

Golem pretrains a [gt-pyg](https://github.com/pgniewko/gt-pyg) `GraphTransformerNet` backbone to predict Mordred 2D molecular descriptors, then the pretrained weights transfer to downstream property-prediction tasks via fine-tuning notebooks.

## Installation

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- The [gt-pyg](https://github.com/pgniewko/gt-pyg) package

### Setup

```bash
cd golem

# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install golem (editable)
uv pip install -e .

# Install gt-pyg from GitHub
pip install git+https://github.com/pgniewko/gt-pyg.git

# (Optional) Install dev dependencies for tests and notebooks
uv pip install -e ".[dev]"
```

### Verify installation

```bash
golem --help
# Should show the 'pretrain' command

python -c "from gt_pyg import GraphTransformerNet; print('gt-pyg OK')"
python -c "from golem.config import PretrainConfig; print('golem OK')"
```

## Running Pretraining

### Quick test run (~1 minute)

```bash
golem pretrain \
  --smiles data/all_openadmet_smiles.smi \
  --output experiments/test_pretrain \
  --max-epochs 10 \
  --subsample 0.1 \
  --no-isoforms
```

### Production run

```bash
golem pretrain \
  --smiles data/all_openadmet_smiles.smi \
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

This reads `metrics.csv` and `resolved_config.yaml` from the experiment directory and produces a self-contained Chart.js dashboard (`pretrain_report.html`) with:

- Training & validation loss curves
- Validation RMSE curve
- Learning rate schedule
- Train vs val loss gap
- Summary cards (best epoch, best val loss, elapsed time, architecture)
- Epoch-by-epoch table with the best row highlighted

To write the report to a custom path:

```bash
golem report experiments/pretrain --output path/to/report.html
```

## Running Fine-Tuning Notebooks

The four fine-tuning notebooks in `notebooks/` compare random-init vs pretrained models on single-task (LogD) and multi-task (9 endpoints) settings.

### Setup

```bash
# Install notebook dependencies (scipy, scikit-learn not in package deps)
uv pip install jupyter scipy scikit-learn matplotlib

# Launch Jupyter
cd notebooks
jupyter notebook
```

### The four experiments

| Notebook | Tasks | Init | What it tests |
|----------|-------|------|---------------|
| `finetune_st_random.ipynb` | 1 (LogD) | Random | Baseline single-task |
| `finetune_st_pretrained.ipynb` | 1 (LogD) | Golem | Pretrained single-task |
| `finetune_mt_random.ipynb` | 9 (all) | Random | Baseline multi-task |
| `finetune_mt_pretrained.ipynb` | 9 (all) | Golem | Pretrained multi-task |

All four notebooks are **structurally identical** &mdash; they differ only in the first cell (config). The pretrained notebooks expect a checkpoint at `../../experiments/pretrain/best_checkpoint.pt`.

## Running Tests

```bash
pytest tests/ -v
```

The tests cover isoform enumeration, Mordred descriptor computation, the NaN-aware scaler, config loading, data splitting, and SMILES file loading. Note: descriptor tests require `mordredcommunity` and may take a few seconds.

## Project Structure

```
golem/
├── pyproject.toml                  # Package metadata + dependencies
├── docs/
│   └── audit.html                  # Comprehensive project audit
│
├── golem/                          # The Python package (pretraining only)
│   ├── __init__.py
│   ├── cli.py                      # Click CLI: `golem pretrain`
│   ├── config.py                   # Dataclasses (ModelConfig, IsoformConfig,
│   │                               #   PretrainConfig) + YAML loading
│   ├── isoforms.py                 # Tautomer / protonation / neutralization
│   │                               #   enumeration (RDKit + Dimorphite-DL)
│   ├── descriptors.py              # Mordred 2D computation + NaNAwareStandardScaler
│   ├── pretrain.py                 # Full MDAE pretraining loop
│   ├── utils.py                    # Seeding, data splitting, DataLoader, SMILES I/O
│   └── _vendor/                    # Local copies of gt-pyg code if fixes are needed
│       └── __init__.py             #   (currently empty — no fixes needed)
│
├── configs/
│   └── pretrain_openadmet.yaml     # Production pretraining config
│
├── data/                           # Training and test data
│   ├── all_openadmet_smiles.smi    # 7,608 SMILES for pretraining
│   ├── train-set/
│   │   ├── expansion_data_train.csv          # Original-scale training data
│   │   └── expansion_log_data_train.csv      # Log-transformed (used by notebooks)
│   └── test-set/
│       ├── expansion_data_test_blinded.csv
│       ├── expansion_data_test_full.csv
│       └── expansion_data_test_full_lb_flag.csv  # With leaderboard flag
│
├── notebooks/                      # Fine-tuning & analysis notebooks
│   ├── finetune_st_random.ipynb
│   ├── finetune_st_pretrained.ipynb
│   ├── finetune_mt_random.ipynb
│   ├── finetune_mt_pretrained.ipynb
│   ├── inspect_isoforms.ipynb      # Isoform enumeration analysis
│   └── compare_experiments.ipynb   # Cross-experiment comparison
│
├── experiments/                    # Output directory for all runs
│   ├── test_pretrain/              # Test run outputs
│   ├── pretrain/                   # Production pretrain outputs (when run)
│   ├── st_random/                  # Notebook experiment outputs
│   ├── st_pretrained/
│   ├── mt_random/
│   └── mt_pretrained/
│
└── tests/
    ├── test_isoforms.py            # Isoform enumeration tests
    ├── test_descriptors.py         # Mordred + scaler tests
    └── test_pretrain.py            # Config, splitting, SMILES loading tests
```

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
| How the model is constructed | `pretrain.py:339-353` and notebook cell 5 |
| The masked MSE pretraining loss | `pretrain.py:117-128` |
| The 5-component fine-tuning loss | Notebook cell 7 |
| NaN handling / validity masking | `descriptors.py:72-76` |
| Scaler fit (train-only, no leakage) | `pretrain.py:306-308` |
| Pretrained weight loading (shape-filter) | Notebook cell 6 |
| Config defaults | `config.py` dataclass definitions |
| Production config | `configs/pretrain_openadmet.yaml` |
| Pretraining pipeline flow | `pretrain.py:pretrain()` docstring (lines 1-16) |
