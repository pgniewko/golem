# Golem Pretraining Review: Comparison with CheMeleon, Errors, and Specifications for New Project

## Context

The `golem` branch of gt-pyg implements a CheMeleon-inspired pretraining pipeline (MDAE — Masked Descriptor Autoencoding) that trains a Graph Transformer backbone to predict Mordred molecular descriptors. This review compares the implementation against the actual CheMeleon code (`JacksonBurns/chemeleon`), identifies errors and improvements, and provides specifications for a new clean-start project.

The user's intent is to:
1. Pretrain on a **smaller** dataset (not 1M molecules) but with **isoform enumeration** for data augmentation
2. Eventually write specs/instructions for a new standalone project based on these notes
3. Get the NaN/validity masking right — save as 0 but never train on those positions
4. Use best-practice isoform enumeration

**Key decisions made:**
- **New separate repository** (gt-pyg stays a focused model library, new project depends on it)
- **Protonation**: Dimorphite-DL as primary (pH-aware, chemically accurate), RDKit Uncharger as fallback for failures
- **NaN handling**: Validity mask approach (Golem's, which is actually better than CheMeleon's)

---

## File Reference — Absolute Paths

### Golem project (target)
- **Project root**: `/Users/pawelgniewek/projects/golem/`
- **This plan**: `/Users/pawelgniewek/projects/golem/PLAN.md`

### gt-pyg repository
- **Repo root**: `/Users/pawelgniewek/projects/gt-pyg/`
- **Main branch**: current gt-pyg v1.5.7 code
- **`golem` branch**: original in-repo Golem implementation (reference only)
  - Pretrain code: `git show golem:gt_pyg/pretrain/{__init__.py,config.py,dataset.py,descriptors.py,model.py,trainer.py}`
  - Tests: `git show golem:gt_pyg/pretrain/tests/`
- **`feature/multi-isoform-v2` branch**: data files + notebooks
  - Training data: `git show feature/multi-isoform-v2:notebooks/v2.0.0/data/train-set/expansion_data_train.csv`
  - Log-transformed training data: `git show feature/multi-isoform-v2:notebooks/v2.0.0/data/train-set/expansion_log_data_train.csv`
  - Test data (blinded): `git show feature/multi-isoform-v2:notebooks/v2.0.0/data/test-set/expansion_data_test_blinded.csv`
  - Test data (full): `git show feature/multi-isoform-v2:notebooks/v2.0.0/data/test-set/expansion_data_test_full.csv`
  - Test data (with leaderboard flag): `git show feature/multi-isoform-v2:notebooks/v2.0.0/data/test-set/expansion_data_test_full_lb_flag.csv`
  - All SMILES: `git show feature/multi-isoform-v2:notebooks/v2.0.0/data/all_openadmet_smiles.smi`
  - Log transform functions: `git show feature/multi-isoform-v2:notebooks/v2.0.0/src/utils.py`
  - Training notebooks (reference): `git show feature/multi-isoform-v2:notebooks/v2.0.0/train_without_isoforms.ipynb` (and others)
  - Comparison notebooks (reference): `git show feature/multi-isoform-v2:notebooks/v2.0.0/compare_isoform_single_ep.ipynb`

### External reference files
- **CALIBRATE notebook**: `/Users/pawelgniewek/projects/expansionrx-openadmet/v33/CALIBRATE-MTME-0.ipynb`

### CheMeleon (external repo, for reference only)
- GitHub: `JacksonBurns/chemeleon`

---

## Part A: Side-by-Side Comparison — Golem vs CheMeleon

| Aspect | CheMeleon (actual code) | Golem (golem branch) | Assessment |
|--------|------------------------|---------------------|------------|
| **Backbone** | D-MPNN (Chemprop), 6 layers, d=2048 | Graph Transformer (PyG), 4 layers, d=128 | Different architectures; GT is fine, but much smaller default capacity |
| **Descriptor set** | All Mordred 2D (~1613 raw, filtered to numeric) | All Mordred 2D (filtered to numeric, no all-NaN cols) | Equivalent |
| **Descriptor NaN handling** | `fill_missing()` then `to_numpy(float32)` — NaN becomes 0.0 via fill_missing; **no validity mask** | `fill_missing()` → `dropna(axis=1, how='all')` → NaN preserved → `descriptor_valid_mask` built → NaN filled with 0.0 for storage | **Golem is better** — tracks validity per element |
| **Scaling** | Feature means/vars computed via **Welford online algorithm on train split only**; NaN→0 via `nan_to_num` before stats | `StandardScaler` with `np.nanmean/nanstd` on **entire dataset** (before train/val split) | **Golem has data leakage** — scaler fit on val data too |
| **Winsorization** | `torch.clamp(scaled, -6, 6)` | `np.clip(scaled, -6, 6)` | Equivalent |
| **Masking strategy** | **Loss masking**: `mask = (torch.rand_like(targets) < 0.15).bool()`, passed to `criterion(preds, targets, mask, ...)` | **Loss masking**: `rand_mask & valid_mask`, then `F.mse_loss(preds[mask], targets[mask])` | Conceptually equivalent; Golem adds validity awareness |
| **Masking ratio** | 0.15 | 0.15 | Same |
| **NaN in loss** | Not explicitly handled — `fill_missing()` turns NaN→0, so model trains on 0.0 targets for failed descriptors | `descriptor_valid_mask` excludes NaN positions from loss | **Golem is better** — CheMeleon silently trains on imputed zeros |
| **Validation loss** | Full MSE (no masking) on all descriptors including imputed zeros | MSE on valid positions only (no masking) | Golem is more principled |
| **Optimizer** | Lightning default (Adam?) | AdamW (lr=1e-4, wd=1e-5) | Both reasonable |
| **LR schedule** | Not visible in code (Lightning default) | Linear warmup (25 epochs) + cosine decay | Golem is more explicit |
| **Early stopping** | patience=50, monitor val_loss | patience=50, monitor val_loss | Same |
| **Batch size** | 128 | 128 | Same |
| **Max epochs** | 500 | 500 | Same |
| **Prediction head** | `RegressionFFN(n_tasks=n_features, input_dim=2048, hidden_dim=1024)` (3-layer FFN) | `Linear(head_in_dim, 512) → GELU → Linear(512, D)` (2-layer) | CheMeleon head is larger (matches backbone); Golem head is modest |
| **Train/val/test split** | 70/20/10 | 80/20/0 | CheMeleon holds out a test set |
| **Pretraining data** | ~1M molecules from PubChem | User-provided SMILES (smaller), with isoform expansion | Different strategy; Golem uses isoforms as augmentation |
| **Checkpoint** | `torch.save(model, 'best.pt')` (full model) | Saves backbone only + scaler + descriptor names + head state in extras | Golem is more structured |
| **Isoform support** | None | Tautomers, protonation states (Dimorphite-DL), neutralization | Golem's unique addition |

### A.2 — CheMeleon Deep Dive: Architecture and Training Details

**D-MPNN (the "CheMeleon" model):**
- Chemprop's `BondMessagePassing` with `d_h=2048`, `depth=6` (6 message-passing layers)
- `MeanAggregation` for graph-level pooling
- Predictor: `RegressionFFN(n_tasks=n_features, input_dim=2048, hidden_dim=1024)` — 3-layer FFN
- Total parameters: ~9.3M
- Loss: `BoundedMSELoss` with masking — loss only computed on randomly masked 15% of descriptor positions
- Optimizer: AdamW (via Lightning), specific lr not explicit in config (Lightning default or `1e-3`)
- Split: 70/20/10 using Chemprop's `make_split_indices()`
- Data: ~1M PubChem molecules, Mordred 2D descriptors stored in Zarr format

**MLP-PLR (descriptor autoencoder — separate model, not the GNN):**
- Input: raw Mordred descriptors → `PeriodicEmbeddings` (learned Fourier features, 48 frequencies per descriptor)
- Architecture: 4-layer MLP autoencoder `[n_features*48 → 2048 → 2048 → 2048 → 2048 → n_features]`
- Key difference from D-MPNN: **input descriptors are zeroed at masked positions** (input masking) AND loss only on masked positions (loss masking). The D-MPNN version does NOT zero inputs.
- Welford online algorithm computes feature means/vars on train split only, NaN-aware (ignores NaN during accumulation, per-feature non-NaN count tracking)
- `torch.nan_to_num(w.mean, posinf=0.0, neginf=0.0)` handles all-NaN columns by setting mean to 0
- AdamW lr=1e-5, batch_size=1024, max_epochs=500, patience=50

**CheMeleon fine-tuning protocol:**
- **No frozen layers** — full fine-tuning of all 9.3M D-MPNN parameters
- Pretrained weights loaded via `model.load_state_dict()` (full model, not backbone-only)
- Task-specific heads are randomly initialized (part of Chemprop's `RegressionFFN`)
- This is important: CheMeleon does NOT freeze the backbone during downstream task training

**CheMeleon fingerprint extraction:**
- 2048-dimensional embeddings extracted from the pretrained D-MPNN's message-passing output
- Checkpoint downloaded from Zenodo (public release)
- Uses `MeanAggregation` on node features → 2048D vector per molecule

---

## Part B: Errors and Issues in the Golem Implementation

### B.1 — Critical Issues

**B.1.1 Scaler Data Leakage** (High)
- `build_pretrain_dataset()` fits the `StandardScaler` on ALL molecules (line 122 of dataset.py), then the CLI splits into train/val afterward (golem.py lines 107-113).
- Fix: Split first, fit scaler on train only, transform val separately. CheMeleon does this correctly — Welford stats are computed on `train_dset` only.

**B.1.2 CheMeleon does NOT mask NaN positions — it trains on imputed zeros** (Design divergence)
- CheMeleon's `fill_missing()` replaces NaN with 0.0, and there is **no validity mask**. The model trains on these zero-valued targets.
- Golem's approach (validity mask excluding NaN from loss) is actually **more principled** than CheMeleon's.
- However, this means Golem's results won't directly reproduce CheMeleon's. This is fine — it's an improvement.

**B.1.3 `descriptor_head_hidden` config key is ignored** (Medium)
- `config.py` defines `descriptor_head_hidden: 512`, but `golem.py` passes it correctly as `hidden_dim=config["pretrain"]["descriptor_head_hidden"]` (line 141).
- Wait — actually checking the code again, the CLI does thread it. This was flagged in ANALYSIS.md but appears to have been fixed.

### B.2 — Medium Issues

**B.2.1 No test split** (Medium)
- CheMeleon uses 70/20/10 (train/val/test). Golem uses 80/20 with no held-out test set.
- A test set is important for reporting final pretrain RMSE (CheMeleon reports 0.14 RMSE).

**B.2.2 Welford vs np.nanmean for statistics** (Low-Medium)
- CheMeleon uses `nan_to_num(mean, posinf=0, neginf=0)` which turns NaN means to 0.
- Golem uses `np.nanmean/nanstd` which skips NaN — more principled.
- But: the `nan_to_num` on means suggests CheMeleon has columns where ALL values are NaN (should have been filtered). Golem drops all-NaN columns before scaling, which is cleaner.

**B.2.3 No `num_workers` in DataLoader** (Medium for performance)
- `trainer.py` creates DataLoaders with no `num_workers`. CheMeleon uses `num_workers=4, persistent_workers=True`.

**B.2.4 Prediction head is undersized relative to backbone** (Design note)
- CheMeleon: backbone=2048 → FFN hidden=1024 → n_features output
- Golem: backbone pool output=128 → head hidden=512 → n_features output
- The 4:1 expansion from 128→512 is reasonable, but the absolute capacity is much lower.

### B.3 — Low Issues

**B.3.1 `_TAUTOMER_ENUMERATOR` module-level singleton with mutable state** (Low)
- `isoforms.py`: `SetMaxTautomers()` mutates global state. Thread-unsafe.

**B.3.2 Masking guarantee is batch-global, not per-sample** (Negligible)
- Statistically impossible to matter at 15% masking on ~500+ descriptors.

### B.4 — Training Notebook Issues (golem branch)

The `notebooks/v2.0.0/` directory contains 4 training notebooks that reveal additional issues:

**B.4.1 Massive code duplication** (Medium)
- 70%+ code overlap across `train_with_k_isoforms.ipynb`, `train_without_isoforms.ipynb`, `train_logd_golem.ipynb`, `train_logd_single.ipynb`
- Shared boilerplate: data loading, model construction, training loop, evaluation
- A shared `src/utils.py` exists but is a stale copy from another project (`expansionrx-openadmet`)
- New project should factor shared logic into importable modules

**B.4.2 Sophisticated custom loss function** (Design note)
- Notebooks use a 5-component loss: RAE (relative absolute error) + Huber + Correlation + Kendall Rank + R²
- This is **not used in pretraining** (pretraining uses simple MSE), only in fine-tuning
- Worth noting for the new project's fine-tuning capabilities
- The 5-component loss with per-component weighting is more advanced than CheMeleon's simple MSE fine-tuning

**B.4.3 Non-functional curriculum learning** (Medium)
- `train_without_isoforms.ipynb` accepts a `curriculum_learning` parameter but it is never used in the training loop
- The parameter is accepted, stored, but has no effect on batch construction or loss weighting
- Dead code — remove or implement properly in new project

**B.4.4 Silent NaN handling in evaluation** (Low-Medium)
- Evaluation code silently skips batches that produce NaN predictions
- Should log warnings when this occurs

**B.4.5 Pretrained weight loading strategy** (Design note)
- Notebooks load Golem pretrained weights with `strict=False`
- Task-head keys (`mu_mlp.*`, `log_var_mlp.*`) from checkpoint are filtered out
- Only backbone weights (embeddings, GT layers, norms) are loaded
- This is correct and should be preserved in the new project

---

## Part C: Isoform Enumeration — Best Practices and Recommendations

### C.1 — Recommended Isoform Enumeration Techniques

**Tier 1 (most reliable, always include):**

1. **Tautomers** — RDKit `TautomerEnumerator`
   - Well-tested, deterministic
   - Set `max_tautomers` conservatively (e.g., 25)
   - Always include the canonical tautomer as isoform 0

2. **Neutralized forms** — RDKit `Uncharger` or SMARTS-based neutralization
   - Deterministic, fast
   - Generates the uncharged form of ionizable molecules

**Tier 2 (valuable but requires care):**

3. **Protonation states** — Dimorphite-DL (pH-dependent)
   - Generates physiologically relevant forms at target pH (e.g., 7.4)
   - **Caveat**: Dimorphite-DL is an external dependency, can be slow, and occasionally crashes on unusual molecules. Needs robust error handling.
   - Alternative: `chembl_structure_pipeline` or `MolStandardize` protonation

**Tier 3 (optional, for specific use cases):**

4. **Stereoisomers** — enumerate undefined stereocenters
   - Only when stereochemistry is unspecified in input
   - Can cause combinatorial explosion

### C.2 — Where Should Isoform Enumeration Code Live?

**Decision: In the NEW separate repository, not gt-pyg.**

Rationale:
- gt-pyg is a focused Graph Transformer library (architecture + featurization)
- Isoform enumeration is a **data preparation** concern, not a model concern
- The new project will own the pretraining pipeline end-to-end
- Isoform code adds dependencies (Dimorphite-DL) that gt-pyg users don't need
- The new project imports gt-pyg's `GraphTransformerNet` as a backbone dependency

The **featurization functions** (`get_atom_features`, `get_bond_features`, `get_tensor_data`) stay in gt-pyg since they're tied to the model's expected input format.

### C.4 — Protonation Tool Recommendation

**Primary: Dimorphite-DL** (pH-aware, empirically-derived pKa rules, handles common drug-like functional groups at physiological pH 7.4)

**Fallback: RDKit Uncharger** (for when Dimorphite-DL fails on unusual molecules)

RDKit's Uncharger alone only produces the neutralized form — it can't enumerate protonation states (e.g., histidine ~50% protonated at pH 7.4). Dimorphite-DL is the gold standard for this. The golem branch already has this pattern working with robust error handling.

### C.3 — Isoform Deduplication Strategy

- Canonicalize all isoform SMILES → deduplicate by string equality
- Track parent-to-isoform mapping for analysis
- Always keep the original molecule as isoform index 0

---

## Part D: Issues Inherited from gt-pyg Main Branch

These apply to the backbone code used by the pretrain pipeline (from the audit):

1. **`_eij` thread-safety** (#1) — affects DataParallel pretraining
2. **Shape mismatch model output vs labels** (#3) — doesn't affect pretrain (uses `encode()` not `forward()`)
3. **Silent data truncation in `get_tensor_data`** (#29) — affects isoform dataset building
4. **No `edge_attr` validation in GTConv** (#38) — could silently degrade pretrained backbone
5. **`pandas` phantom dependency** (#39) — unnecessary at install time

---

## Part E: Implementation Specification for the Golem Project

> **Audience**: This specification is written for a Claude Code agent to implement from scratch.
> All code from the golem branch, CALIBRATE notebook, and comparison notebooks is **reference only** — none of it is correct or complete. The implementing agent must synthesize clean, self-contained code.

### E.0 — Project Location, Setup, and Git Workflow

**Location**: `/Users/pawelgniewek/projects/golem`
**Environment**: uv venv, gt-pyg installed from `/Users/pawelgniewek/projects/gt-pyg` (editable)
**CLI**: All commands must be callable from the command line

```bash
# Setup sequence
cd /Users/pawelgniewek/projects/golem
uv venv
source .venv/bin/activate
uv pip install -e .                            # install golem
uv pip install -e /Users/pawelgniewek/projects/gt-pyg  # install gt-pyg
```

**Git workflow:**
- **Golem repo** (`/Users/pawelgniewek/projects/golem`): New project, full autonomy. Work directly on main.
- **gt-pyg repo** (`/Users/pawelgniewek/projects/gt-pyg`): Create a **feature branch** for any modifications (e.g., `feature/golem-support`). Do NOT modify gt-pyg main directly. Changes to gt-pyg require user approval.
- If gt-pyg changes are needed (e.g., adding `encode()` or fixing APIs), branch off main, make changes, test, then note for user review.

### E.1 — Critical gt-pyg API Facts (MUST follow these exactly)

**The CALIBRATE-MTME-0.ipynb notebook was written for gt-pyg v1.3.0. We are now on v1.5.7.** The following changes happened between those versions and BREAK the notebook code:

| PR | Change | Impact on CALIBRATE notebook |
|----|--------|------------------------------|
| #22 | **Removed positional encoding code** | `pe_in_dim=None` constructor arg REMOVED; `pe=None` forward() arg REMOVED |
| #18 | **Refactored data utils** | `gnn=True, pe=False, pe_dim=0` → now just `gnm=True` |
| #30 | **Fixed node feature dimension** | Feature dim may differ from hardcoded 139 |
| #39 | **Fixed pharmacophore SMARTS** | Feature values may differ |
| #35 | **Renamed use_chirality** | Now `use_stereochemistry` |
| #25 | **Added checkpointing** | New save/load_checkpoint methods available |

The implementing agent MUST use the gt-pyg v1.5.7 API, NOT the notebook code:

| What | gt-pyg v1.5.7 (CORRECT) | CALIBRATE notebook v1.3.0 (BROKEN) |
|------|----------------------|--------------------------------------|
| `get_tensor_data` | `get_tensor_data(x_smiles, y, gnm=True)` | `get_tensor_data(smiles, y, gnn=True, pe=False, pe_dim=0)` |
| `model.forward()` | `model(x, edge_index, edge_attr, batch=batch, zero_var=False)` | `model(x, edge_index, edge_attr, pe=None, batch=batch)` |
| Constructor | No `pe_in_dim` parameter | `pe_in_dim=None` |
| Feature dims | `get_atom_feature_dim()` and `get_bond_feature_dim()` (dynamic) | Hardcoded 139 and 39 |

**Available gt-pyg imports:**
```python
from gt_pyg import GraphTransformerNet, get_tensor_data
from gt_pyg.data import get_atom_feature_dim, get_bond_feature_dim, canonicalize_smiles
from gt_pyg.nn import save_checkpoint, load_checkpoint
```

**GraphTransformerNet forward() signature on main:**
```python
def forward(self, x, edge_index, edge_attr, batch, zero_var=False) -> Tuple[Tensor, Tensor]:
    # Returns (pred, log_var) where pred = mu (or reparameterized sample if training and not zero_var)
```

**get_tensor_data() on main returns Data objects with:**
- `data.x` — node features `[N, F]`
- `data.edge_index` — edge indices `[2, E]`
- `data.edge_attr` — edge features `[E, D]`
- `data.y` — targets `[T]` (NaN for missing)
- `data.y_mask` — mask `[T]` (1.0=present, 0.0=missing)

### E.2 — NO Changes Needed to gt-pyg

**Simplification: We do NOT need an `encode()` method.** Instead, for pretraining, we use `GraphTransformerNet(num_tasks=num_descriptors)` directly. The built-in `mu_mlp` serves as the descriptor prediction head. No wrapper model, no gt-pyg modifications.

**How this works:**
- Pretraining: `GraphTransformerNet(num_tasks=D)` where D = number of valid Mordred descriptors (~500-600)
- Forward returns `(pred, log_var)` — use `pred` for masked MSE against descriptor targets
- Save full model checkpoint via `model.save_checkpoint()`
- Fine-tuning: `GraphTransformerNet(num_tasks=1 or 9)` — different `num_tasks` → head weights have different shapes
- Load pretrained weights with **shape-filtering** (see E.14)

**Why this is better than the encode() approach:**
- No changes to gt-pyg
- The first hidden layer of mu_mlp (`Linear(512, 128)`) is shape-compatible and TRANSFERS as a bonus — more pretrained parameters than encode() would give
- Only the output layers of mu_mlp/log_var_mlp are re-initialized (different num_tasks)

### E.3 — File Structure

```
/Users/pawelgniewek/projects/golem/
├── pyproject.toml
├── golem/
│   ├── __init__.py
│   ├── cli.py                 # Click CLI: golem pretrain, golem finetune, golem compare
│   ├── config.py              # Dataclasses + optional YAML loading/merging
│   ├── isoforms.py            # Tautomer/protonation/neutralization enumeration
│   ├── descriptors.py         # Mordred computation + NaN-aware StandardScaler
│   ├── pretrain.py            # Pretraining loop (uses GraphTransformerNet directly, no wrapper)
│   ├── finetune.py            # Fine-tuning loop + prediction at end (all 4 models)
│   ├── predict.py             # Utility: generate_predictions(), fit_calibration()
│   ├── loss.py                # MSE for pretrain; 5-component for fine-tune
│   ├── metrics.py             # Official OpenADMET metrics (MAE, RAE, R², Spearman, Kendall)
│   ├── transforms.py          # Log transform / inverse log transform (from OpenADMET)
│   └── utils.py               # Seeding, data splitting, DataLoader creation
├── tests/
│   ├── test_isoforms.py
│   ├── test_pretrain.py
│   ├── test_finetune.py
│   └── test_e2e.py
├── data/                      # Copied from gt-pyg feature/multi-isoform-v2 branch
│   ├── train-set/
│   │   ├── expansion_data_train.csv
│   │   └── expansion_log_data_train.csv
│   ├── test-set/
│   │   ├── expansion_data_test_blinded.csv
│   │   ├── expansion_data_test_full.csv
│   │   └── expansion_data_test_full_lb_flag.csv
│   └── all_openadmet_smiles.smi
├── notebooks/
│   └── compare_predictions.ipynb   # Optional interactive comparison
├── configs/
│   ├── pretrain_openadmet.yaml
│   └── finetune_openadmet.yaml
└── experiments/               # Output directory for experiment results
    └── .gitkeep
```

**Eliminated vs previous plan:** `pretrain_model.py` (no wrapper needed). `predict.py` is a utility module (not a CLI command). `compare` is a CLI command that outputs tables + optional plots.

### E.4 — Data Files (copy from gt-pyg branch)

Copy from the `feature/multi-isoform-v2` branch of gt-pyg:

```bash
cd /Users/pawelgniewek/projects/gt-pyg
git show feature/multi-isoform-v2:notebooks/v2.0.0/data/train-set/expansion_data_train.csv > /tmp/expansion_data_train.csv
git show feature/multi-isoform-v2:notebooks/v2.0.0/data/train-set/expansion_log_data_train.csv > /tmp/expansion_log_data_train.csv
git show feature/multi-isoform-v2:notebooks/v2.0.0/data/test-set/expansion_data_test_blinded.csv > /tmp/expansion_data_test_blinded.csv
git show feature/multi-isoform-v2:notebooks/v2.0.0/data/test-set/expansion_data_test_full.csv > /tmp/expansion_data_test_full.csv
git show feature/multi-isoform-v2:notebooks/v2.0.0/data/test-set/expansion_data_test_full_lb_flag.csv > /tmp/expansion_data_test_full_lb_flag.csv
git show feature/multi-isoform-v2:notebooks/v2.0.0/data/all_openadmet_smiles.smi > /tmp/all_openadmet_smiles.smi
```

Also copy the reference notebook from the other repo:
```bash
cp /Users/pawelgniewek/projects/expansionrx-openadmet/v33/CALIBRATE-MTME-0.ipynb /Users/pawelgniewek/projects/golem/notebooks/CALIBRATE-MTME-0-reference.ipynb
```

**Training data format** (`expansion_log_data_train.csv`):
```
SMILES,Molecule Name,LogD,LogS,Log_HLM_CLint,Log_MLM_CLint,Log_Caco_Papp_AB,Log_Caco_ER,Log_Mouse_PPB,Log_Mouse_BPB,Log_Mouse_MPB
```
- Already log-transformed. NaN = missing endpoint.
- ~5326 molecules, 9 endpoints.

**Test data format** (`expansion_data_test_full_lb_flag.csv`):
```
Molecule Name,SMILES,LogD,KSOL,HLM CLint,MLM CLint,Caco-2 Permeability Papp A>B,Caco-2 Permeability Efflux,MPPB,MBPB,MGMB,is_leaderboard
```
- ORIGINAL scale (not log-transformed). Must log-transform for evaluation.
- `is_leaderboard=1` marks the leaderboard subset (~2282 molecules).

**Endpoint mapping** (original name → log name → multiplier):
| Original | Log Name | Multiplier | Transform |
|----------|----------|------------|-----------|
| LogD | LogD | 1 | None (already log-scale) |
| KSOL | LogS | 1e-6 | log10((val+1) * mult) |
| HLM CLint | Log_HLM_CLint | 1 | log10((val+1) * mult) |
| MLM CLint | Log_MLM_CLint | 1 | log10((val+1) * mult) |
| Caco-2 Perm Papp A>B | Log_Caco_Papp_AB | 1e-6 | log10((val+1) * mult) |
| Caco-2 Perm Efflux | Log_Caco_ER | 1 | log10((val+1) * mult) |
| MPPB | Log_Mouse_PPB | 1 | log10((val+1) * mult) |
| MBPB | Log_Mouse_BPB | 1 | log10((val+1) * mult) |
| MGMB | Log_Mouse_MPB | 1 | log10((val+1) * mult) |

### E.5 — Model Architecture (shared by all 4 experiments)

All models use IDENTICAL architecture from the CALIBRATE notebook, adapted for gt-pyg main:

```python
from gt_pyg import GraphTransformerNet
from gt_pyg.data import get_atom_feature_dim, get_bond_feature_dim

model = GraphTransformerNet(
    node_dim_in=get_atom_feature_dim(),   # dynamically computed
    edge_dim_in=get_bond_feature_dim(),   # dynamically computed
    num_gt_layers=4,
    hidden_dim=128,
    num_heads=8,
    norm="bn",                            # BatchNorm (not LayerNorm)
    gt_aggregators=["sum", "mean"],       # dual aggregation in GT layers
    aggregators=["sum", "mean", "max", "std"],  # 4 global aggregators → pool dim = 4*128 = 512
    dropout=0.3,
    act="gelu",
    gate=True,                            # gated attention
    qkv_bias=False,
    num_tasks=num_tasks,                  # 1 for ST-LogD, 9 for MT
)
```

### E.6 — The Four-Model Experiment

**Goal**: Train 4 models to test whether Golem pretraining helps. All settings IDENTICAL except:
1. **num_tasks**: 1 (ST-LogD) or 9 (MT-all-endpoints)
2. **Initialization**: random or from Golem pretrained checkpoint

| Model | num_tasks | Init | Config suffix |
|-------|-----------|------|---------------|
| ST-LogD-Random | 1 | random | `st_random` |
| ST-LogD-Pretrained | 1 | golem checkpoint | `st_pretrained` |
| MT-Random | 9 | random | `mt_random` |
| MT-Pretrained | 9 | golem checkpoint | `mt_pretrained` |

**Configurable data split:**
The split is configurable as a list of fractions:
- **Experiment (4-model comparison)**: `[0.7, 0.2, 0.1]` — 70/20/10 train/val/test, random split
- **Production**: `[0.8, 0.2]` — 80/20 train/val with early stopping, no held-out test
- The CLI accepts `--split 0.7 0.2 0.1` or `--split 0.8 0.2`
- When 3 fractions given: train/val/test. When 2 fractions: train/val only.
- **Scaler is ALWAYS fit on train split only**, regardless of split configuration.

```python
def split_data(n: int, fractions: List[float], seed: int = 42) -> Tuple[np.ndarray, ...]:
    """Random split into len(fractions) subsets. Returns index arrays.
    fractions must sum to 1.0. Supports 2 or 3 splits."""
```

**Shared training settings (all 4 models):**
```python
BATCH_TRAIN = 256
BATCH_EVAL = 1024
EPOCHS = 2000            # production (10 for test run)
BASE_LR = 1e-3
MIN_LR = 1e-5            # base_lr / 100
WEIGHT_DECAY = 1e-5
WARMUP_EPOCHS = 25
GRAD_CLIP_NORM = 5.0
SEED = 1928374650

# Loss weights
W_RAE = 1.0
W_HUBER = 0.25
W_CORR = 0.25
W_TAU = 0.1
W_R2 = 0.1
HUBER_DELTA = 0.5
TAU_TEMP = 2.0
RANK_PAIRS = 512

# Data split (configurable)
SPLIT_FRACTIONS = [0.7, 0.2, 0.1]  # experiment: 70/20/10
# SPLIT_FRACTIONS = [0.8, 0.2]     # production: 80/20

# Scheduler
T_MAX = min(500, EPOCHS)  # cosine cycle length
```

**Loading pretrained weights (CRITICAL — shape-filter pattern):**

> **BUG FIX**: PyTorch `strict=False` does NOT handle shape mismatches — it only
> handles missing/extra keys. Passing shape-mismatched tensors raises `RuntimeError`.
> The implementing agent MUST use the shape-filter pattern below.

```python
# For pretrained variants:
model = GraphTransformerNet(...)  # create with correct num_tasks
checkpoint = torch.load("golem_pretrained.pt", map_location=device)
pretrain_sd = checkpoint["model_state_dict"]
finetune_sd = model.state_dict()

# Filter to shape-compatible keys only
compatible_sd = {k: v for k, v in pretrain_sd.items()
                 if k in finetune_sd and v.shape == finetune_sd[k].shape}

# Load compatible weights, skip the rest
missing, unexpected = model.load_state_dict(compatible_sd, strict=False)
print(f"Loaded {len(compatible_sd)}/{len(pretrain_sd)} pretrained keys")
print(f"Re-initialized: {missing}")  # should be only output-layer keys
```

**What transfers and what doesn't** (with CALIBRATE architecture):
- `node_emb`, `edge_emb`, `input_norm`, `gt_layers.*`, `readout_norm`, `global_pool` → ALL transfer ✓
- `mu_mlp.mlp.0.*` (Linear 512→128) → transfers ✓ (same shape regardless of num_tasks)
- `mu_mlp.mlp.3.*` (Linear 128→num_tasks) → SKIPPED (different output dim) ✓
- Same pattern for `log_var_mlp`
- ~204 of ~208 keys transfer; only 4 output-layer keys are re-initialized

**For random variants:** just create model with default initialization (no weight loading).

### E.7 — CLI Commands

```bash
# 1. Pretrain on OpenADMET training SMILES with isoform enumeration
golem pretrain \
  --smiles data/all_openadmet_smiles.smi \
  --config configs/pretrain_openadmet.yaml \
  --output experiments/pretrain

# 2. Fine-tune (runs all 4 models)
golem finetune \
  --train-csv data/train-set/expansion_log_data_train.csv \
  --test-csv data/test-set/expansion_data_test_full_lb_flag.csv \
  --pretrained-checkpoint experiments/pretrain/best_checkpoint.pt \
  --output experiments/finetune \
  --split 0.7 0.2 0.1 \      # 70/20/10 for experiment (configurable)
  --epochs 10 \               # 10 for test run
  --subsample 0.1             # 10% for test run (omit for production)

# 3. Compare predictions
golem compare \
  --predictions-dir experiments/finetune \
  --ground-truth data/test-set/expansion_data_test_full_lb_flag.csv \
  --output experiments/comparison
```

### E.8 — Module Specifications

#### E.8.1 — `golem/transforms.py` (Self-contained, from OpenADMET)

Bring the log transform functions from `notebooks/v2.0.0/src/utils.py` on the feature branch. Must be self-contained (no external imports beyond pandas/numpy).

Functions needed:
- `get_conversion_config() → dict` — maps original assay name → (log_name, multiplier, zero_handling)
- `log_transform_assay_data(df) → (df_log, conversion_dict)` — forward transform
- `inverse_log_transform_assay_data(df) → (df_orig, reverse_dict)` — inverse transform
- `log_transform_test_data(df) → df_log` — transform test data (same rules, applied to test)

The key transform logic:
```python
# For endpoints with log_scale=True:
#   forward: log10((val + 1) * multiplier)  — note the +1 BEFORE multiplier
#   inverse: (10^val / multiplier) - 1, clipped to >= 0
# For LogD: no transform (passthrough)
```

**IMPORTANT**: The training CSV is already log-transformed. The test CSV is in original scale. The comparison needs both in the same space.

#### E.8.2 — `golem/metrics.py` (Self-contained, from OpenADMET)

Official OpenADMET evaluation metrics. Must be self-contained.

```python
def official_metrics(y_true, y_pred) -> dict:
    """Compute MAE, RAE, R², Spearman R, Kendall Tau for one endpoint."""

def per_task_metrics(y_true_2d, y_pred_2d, mask_2d, endpoint_names) -> dict:
    """Compute metrics per task with NaN masking."""

def macro_average(metrics_dict) -> dict:
    """Average metrics across endpoints (NaN-safe)."""

def bootstrap_metrics(y_true, y_pred, n_bootstrap=1000, seed=42) -> dict:
    """Bootstrap confidence intervals for each metric."""
```

#### E.8.3 — `golem/loss.py` (5-component custom loss)

Adapted from the CALIBRATE notebook. All 5 components:

1. **RAE loss** — absolute error normalized by per-task MAD
2. **Huber loss** — robust L2, delta=0.5, optionally normalized by MAD
3. **Correlation loss** — 1 - Pearson correlation per task
4. **Kendall rank loss** — soft pairwise ranking with 512 pairs/task
5. **R²-style loss** — SSE / total variance per task

All losses must handle:
- `mask` tensor (1.0=present, 0.0=missing) for multi-task sparse labels
- NaN/Inf guards
- Minimum 3 valid samples per task (skip task otherwise)
- Macro-average across valid tasks

```python
def compute_task_scales(train_loader, num_tasks) -> Tensor:
    """MAD per task from training data (for RAE normalization)."""

def custom_loss(pred, y, mask, *, w_rae, w_huber, w_corr, w_tau, w_r2,
                task_scale, huber_delta, tau_temp, rank_pairs, rng) -> Tensor:
    """Combined 5-component loss."""
```

For pretraining: use simple masked MSE instead (not the 5-component loss).

#### E.8.4 — `golem/isoforms.py` (Isoform enumeration)

```python
def enumerate_isoforms(smiles: str, max_isoforms: int = 25,
                       enable_tautomers: bool = True,
                       enable_protonation: bool = True,
                       enable_neutralization: bool = True) -> List[str]:
    """Enumerate isoforms for a single SMILES. Returns deduplicated canonical SMILES.
    Always includes the original as index 0."""

def enumerate_isoforms_batch(smiles_list: List[str], max_isoforms: int = 25,
                             **kwargs) -> Dict[str, List[str]]:
    """Batch enumeration. Returns {parent_smiles: [isoform_smiles]}."""
```

Key implementation details:
- Tautomers: `RDKit.Chem.MolStandardize.rdMolStandardize.TautomerEnumerator` — create LOCAL instance per call (NOT global singleton)
- Protonation: `dimorphite_dl.DimorphiteDL(min_ph=6.4, max_ph=8.4)` as primary, `Chem.MolStandardize.Uncharger` as fallback
- All isoforms canonicalized with `Chem.MolToSmiles(mol, canonical=True)`
- Deduplicated by string equality
- Robust error handling: log warnings for failures, never crash

#### E.8.5 — `golem/descriptors.py` (Mordred + scaler)

```python
def compute_mordred_descriptors(smiles_list: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Compute Mordred 2D descriptors. Returns (values, validity_mask, descriptor_names).
    validity_mask is True where descriptor computed successfully, False where NaN."""

class NaNAwareStandardScaler:
    """Scaler that uses nanmean/nanstd, fits on train only, serializable."""
    def fit(self, X, validity_mask) -> 'NaNAwareStandardScaler': ...
    def transform(self, X) -> np.ndarray: ...
    def state_dict(self) -> dict: ...
    @classmethod
    def from_state_dict(cls, d) -> 'NaNAwareStandardScaler': ...
```

#### E.8.6 — `golem/pretrain.py` (Pretraining loop — uses GraphTransformerNet directly)

```python
def pretrain(smiles_path, config, output_dir):
    """Full MDAE pretraining pipeline.
    Model: GraphTransformerNet(num_tasks=num_descriptors) — no wrapper needed.
    """
    # 1. Load SMILES
    # 2. Enumerate isoforms (max 25 per molecule)
    # 3. Compute Mordred descriptors + validity masks
    # 4. Split (configurable: [0.7, 0.2, 0.1] or [0.8, 0.2]) — BEFORE fitting scaler
    # 5. Fit scaler on TRAIN ONLY
    # 6. Transform + winsorize [-6, 6]
    # 7. Build PyG datasets:
    #    - Graph features: get_tensor_data(smiles, dummy_y, gnm=True)
    #    - Descriptor targets: stored as data.descriptor_targets tensor [D]
    #    - Validity mask: stored as data.descriptor_valid_mask tensor [D]
    # 8. Create model: GraphTransformerNet(num_tasks=num_descriptors, ...)
    # 9. Train with masked MSE loss (zero_var=True for deterministic predictions)
    # 10. Early stopping on val loss (patience=50)
    # 11. Report test RMSE (if test split exists)
    # 12. Save checkpoint via model.save_checkpoint() — full model, scaler in extras
```

**Pretraining forward call:**
```python
# Use zero_var=True for deterministic predictions during pretraining
pred, log_var = model(batch.x, batch.edge_index, batch.edge_attr,
                      batch=batch.batch, zero_var=True)
# pred shape: [B, num_descriptors] — these ARE the descriptor predictions
# log_var is ignored for pretraining loss
```

**Pretraining loss (masked MSE):**
```python
rand_mask = (torch.rand_like(targets) < masking_ratio).bool()  # random 15% mask
valid_mask = descriptor_valid_mask.bool()                       # NaN positions excluded
final_mask = rand_mask & valid_mask                             # only masked+valid
if final_mask.sum() == 0:
    # Fallback: use all valid positions (extremely rare)
    final_mask = valid_mask
loss = F.mse_loss(pred[final_mask], targets[final_mask])
```

**Note on data.y vs descriptor targets:** `get_tensor_data` creates `data.y` and `data.y_mask` for the normal training pipeline. For pretraining, we need separate descriptor tensors. Options:
- Store descriptors in `data.y` (set `y=descriptor_values` when calling get_tensor_data) — simplest, but conflates task labels with descriptor targets.
- Store descriptors as a separate attribute: after building Data objects with get_tensor_data, add `data.descriptor_targets` and `data.descriptor_valid_mask` as extra tensors. This is cleaner.

**Recommended approach:** Use `get_tensor_data(smiles, dummy_y, gnm=True)` where `dummy_y = [[0.0]*num_descriptors] * N` (or the actual descriptor values). Then for pretraining, the model's forward() predicts `num_descriptors` outputs and the loss uses the descriptor values. The `y_mask` from get_tensor_data would be all 1.0 (since we pass finite values), but we use `descriptor_valid_mask` for the actual masking. **Or simpler**: pass the actual descriptors as `y` to get_tensor_data, and the validity mask becomes the y_mask. This works because `y_mask = isfinite(y)` and we've already replaced NaN with 0.0 in descriptors — but then y_mask would be all 1.0. So we need the separate valid_mask.

**Simplest approach:** Build Data objects manually in the pretrain dataset builder instead of using get_tensor_data for the descriptor part. Use get_tensor_data only for graph features.

#### E.8.7 — `golem/finetune.py` (Fine-tuning loop)

```python
def finetune(train_csv, test_csv, pretrained_checkpoint, output_dir,
             split=[0.7, 0.2, 0.1], epochs=2000, subsample=None,
             calibrate=False, seed=1928374650):
    """Run all 4 model variants with identical settings.

    The split, seed, architecture, optimizer, loss, and all hyperparameters
    are IDENTICAL across variants. Only num_tasks and initialization differ.
    """
    # 1. Load log-transformed training CSV
    # 2. Optionally subsample (for test runs — subsamples training data only)
    # 3. Split train/val(/test) using configurable fractions (same seed for ALL 4)
    # 4. Build PyG datasets via get_tensor_data(smiles, y_matrix, gnm=True)
    #    - For ST-LogD: y = [logd_value] (single column, num_tasks=1)
    #    - For MT: y = [logd, logs, ...all 9] (num_tasks=9)
    #    Note: get_tensor_data handles NaN→y_mask automatically
    # 5. Compute task_scales (MAD per task from TRAIN split only)
    # 6. For each of the 4 variants:
    #    a. Create model (same architecture, different num_tasks)
    #    b. Optionally load pretrained weights (shape-filter pattern, see E.6)
    #    c. Train with 5-component custom loss (zero_var=False for fine-tuning)
    #    d. Track GLOBAL best checkpoint by macro RAE on validation set
    #    e. After training: load best checkpoint
    #    f. Predict on test set
    #    g. Clip predictions to [train_min - δ*range, train_max + δ*range] (δ=0.2)
    #    h. Optionally calibrate: fit affine (a, b) on validation set, apply to test
    #    i. Inverse log-transform predictions back to original scale
    #    j. Save predictions CSV
```

**Simplification: Global best checkpoint only.** No per-endpoint best tracking. One model per variant, selected by lowest macro RAE on validation set. This eliminates the complexity of loading 9 different checkpoints per MT model.

**Calibration is optional** (default off). Enable with `--calibrate`. For the 4-model comparison, keeping calibration off ensures we're comparing raw model quality. Add calibration for production submissions.

**Prediction CSV format** (saved per variant):
```
# File: experiments/finetune/predictions_st_random.csv
# File: experiments/finetune/predictions_st_pretrained.csv
# File: experiments/finetune/predictions_mt_random.csv
# File: experiments/finetune/predictions_mt_pretrained.csv
Molecule Name,SMILES,LogD,KSOL,HLM CLint,...  (original scale, inverse-transformed)
```

For ST models, only the LogD column has predictions; other columns are NaN.

**Evaluation space**: All metrics are computed in ORIGINAL scale (inverse-transformed). This matches what OpenADMET reports.

#### E.8.9 — `golem/predict.py` (Prediction + calibration)

```python
def predict_and_save(model, test_smiles, test_mol_names, endpoints,
                     train_ranges, calibration_params, output_path):
    """Generate predictions, clip, calibrate, inverse-transform, save CSV."""

def fit_calibration(y_true, y_pred, min_points=10) -> Tuple[float, float]:
    """Fit y_true ≈ a * y_pred + b via least squares. Returns (a, b)."""
```

#### E.8.10 — `golem compare` CLI command + optional notebook

The `golem compare` command replaces the notebook as the primary comparison tool. It is testable, scriptable, and self-contained. An optional notebook in `notebooks/` can import the same functions for interactive analysis.

```python
def compare(predictions_dir, ground_truth_csv, output_dir, leaderboard_only=False):
    """Compare 4 model variants against ground truth.
    All comparisons in ORIGINAL scale (predictions already inverse-transformed).
    """
    # 1. Load 4 prediction CSVs from predictions_dir
    # 2. Load ground truth from ground_truth_csv
    # 3. Optionally filter by is_leaderboard=1
    # 4. Merge predictions with ground truth on Molecule Name
    # 5. Compute metrics per endpoint per model: MAE, RAE, R², Spearman R, Kendall Tau
    # 6. Bootstrap CIs (1000 samples) per metric per endpoint per model
    # 7. Print formatted comparison table to stdout
    # 8. Save summary.csv, bar_chart.png, scatter_plots.png to output_dir
```

**Output table format:**
```
Endpoint       | Metric | ST-Random      | ST-Pretrained  | MT-Random      | MT-Pretrained
LogD           | MAE    | 0.432 ± 0.023  | 0.389 ± 0.021  | 0.401 ± 0.020  | 0.367 ± 0.019
LogD           | RAE    | 0.612 ± 0.031  | 0.551 ± 0.029  | 0.568 ± 0.028  | 0.520 ± 0.027
...
Macro Average  | RAE    | 0.612          | 0.551          | 0.623          | 0.534
```

### E.9 — Dependencies

```toml
[project]
name = "golem"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=1.13.0",
    "torch_geometric",
    "mordredcommunity",
    "rdkit",
    "dimorphite_dl",
    "numpy",
    "pandas",
    "pyyaml",
    "tqdm",
    "click",
    "scipy",                 # for spearmanr, kendalltau
    "scikit-learn",          # for r2_score
    "matplotlib",            # for comparison plots
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-cov", "jupyter"]

[project.scripts]
golem = "golem.cli:main"
```

### E.10 — Test Run Protocol

For initial testing (10 epochs, subsampled):

```bash
# Step 1: Pretrain (can be quick — small data, few epochs)
golem pretrain \
  --smiles data/all_openadmet_smiles.smi \
  --config configs/pretrain_openadmet.yaml \
  --output experiments/test_pretrain \
  --max-epochs 10 \
  --subsample 0.1

# Step 2: Fine-tune all 4 models (10 epochs, 10% data)
golem finetune \
  --train-csv data/train-set/expansion_log_data_train.csv \
  --test-csv data/test-set/expansion_data_test_full_lb_flag.csv \
  --pretrained-checkpoint experiments/test_pretrain/best_checkpoint.pt \
  --output experiments/test_finetune \
  --epochs 10 \
  --subsample 0.1

# Step 3: Compare
golem compare \
  --predictions-dir experiments/test_finetune \
  --ground-truth data/test-set/expansion_data_test_full_lb_flag.csv \
  --output experiments/test_comparison
```

### E.11 — Production Run Protocol

```bash
# Step 1: Full pretraining with isoform enumeration (max 25 isoforms per molecule)
golem pretrain \
  --smiles data/train-set/expansion_log_data_train.csv \  # uses SMILES column
  --config configs/pretrain_openadmet.yaml \
  --output experiments/pretrain \
  --max-isoforms 25

# Step 2: Full fine-tuning (2000 epochs, all data, 80/20 for production)
golem finetune \
  --train-csv data/train-set/expansion_log_data_train.csv \
  --test-csv data/test-set/expansion_data_test_full_lb_flag.csv \
  --pretrained-checkpoint experiments/pretrain/best_checkpoint.pt \
  --output experiments/finetune \
  --split 0.8 0.2 \           # production: 80/20 train/val (early stopping on val)
  --epochs 2000

# Step 3: Compare
golem compare \
  --predictions-dir experiments/finetune \
  --ground-truth data/test-set/expansion_data_test_full_lb_flag.csv \
  --output experiments/comparison
```

### E.14 — Resolved Design Decisions (MUST follow — do not propagate errors)

These decisions have been explicitly resolved. The implementing agent should NOT make different choices without asking the user.

**D1. Pretraining uses zero_var=True, fine-tuning uses zero_var=False**
- Pretraining: `model(... zero_var=True)` → deterministic mu predictions for clean descriptor MSE
- Fine-tuning: `model(... zero_var=False)` → variational noise during training (acts as regularization, matches CALIBRATE notebook behavior)
- Evaluation (both): `model.eval()` mode automatically disables noise regardless of zero_var

**D2. Weight loading uses shape-filter, NOT strict=False alone**
- `strict=False` in PyTorch does NOT handle shape mismatches — it raises RuntimeError
- Must filter state_dict to shape-compatible keys before loading (see E.6 for exact code)
- This is a real bug in the original plan and CALIBRATE notebook code

**D3. Log transforms: use v1 functions (matching the existing training CSV)**
- The file `expansion_log_data_train.csv` was created with `log_transform_assay_data()` (v1)
- The v1 transform: `log10((val + 1) * multiplier)` with no special zero handling
- For consistency, use v1 transforms everywhere. Bring v2 functions too but don't use by default.
- Inverse transform: `(10^val / multiplier) - 1`, clipped to >= 0

**D4. Comparison is in ORIGINAL scale**
- Model predictions are in log scale (matching training labels)
- Inverse-transform predictions before computing metrics
- Test ground truth is already in original scale (`expansion_data_test_full_lb_flag.csv`)
- Metrics (MAE, RAE, R², Spearman, Kendall) computed on original-scale values

**D5. Single-task model predicts ONLY LogD**
- ST models: `num_tasks=1`, trained on the LogD column only
- In prediction CSV: LogD column has values, all other 8 columns are NaN
- Comparison table shows ST results only for LogD endpoint

**D6. Subsampling affects training data only**
- `--subsample 0.1` subsamples the training CSV before splitting
- Test set predictions always use the full test set (inference is fast)
- The split fractions apply to the subsampled training data

**D7. Global best checkpoint only (no per-endpoint best)**
- Track one best model per variant by lowest macro RAE on validation set
- This single model is used for ALL endpoint predictions
- Per-endpoint best tracking can be added later if needed

**D8. Config approach: dataclasses internally, YAML optional**
- `config.py` defines `@dataclass` classes: `ModelConfig`, `PretrainConfig`, `FinetuneConfig`
- Factory: `load_config(yaml_path=None, **cli_overrides)` → merges YAML + CLI args
- Defaults are sufficient for quick test runs (no YAML needed)
- YAML provides reproducibility for production runs

**D9. No custom collate_fn**
- Standard PyG `DataLoader` (which uses its default `Batch.from_data_list` collate)
- No isoform aggregation, no custom batching

**D10. Pretrain data pipeline builds Data objects in two steps**
- Step 1: `get_tensor_data(smiles, dummy_y, gnm=True)` → creates graph features (x, edge_index, edge_attr)
- Step 2: Overwrite `data.y = descriptor_targets` and `data.y_mask = descriptor_valid_mask`
- Or build Data objects manually with both graph features and descriptor targets
- The implementing agent should choose the cleanest approach — the key constraint is that descriptor_valid_mask must NOT be the standard `isfinite(y)` mask (since NaN descriptors are stored as 0.0)

### E.12 — Verification Checklist

The implementing agent should verify:

1. **`uv pip install -e .`** succeeds
2. **`golem --help`** shows pretrain, finetune, compare commands
3. **Feature dimensions** match: `get_atom_feature_dim()` and `get_bond_feature_dim()` return consistent values
4. **get_tensor_data** creates Data objects with correct y and y_mask shapes
5. **Pretrained weight loading**: backbone keys load, head keys are skipped, no errors
6. **Loss masking**: NaN positions in y_mask=0 are excluded from loss (verify gradient is zero for those)
7. **Scaler data leakage**: verify scaler.mean_ differs when fit on train vs full dataset
8. **Prediction CSV**: inverse-transformed values are in original assay scale
9. **Comparison**: metrics are computed correctly between prediction CSVs and ground truth
10. **4 models differ only in init and num_tasks**: all other hyperparameters identical (spot-check by logging)

### E.13 — Config Files

**`configs/pretrain_openadmet.yaml`:**
```yaml
model:
  hidden_dim: 128
  num_gt_layers: 4
  num_heads: 8
  norm: "bn"
  gt_aggregators: ["sum", "mean"]
  aggregators: ["sum", "mean", "max", "std"]
  dropout: 0.3
  act: "gelu"
  gate: true

pretrain:
  masking_ratio: 0.15
  descriptor_head_hidden: 512
  batch_size: 128
  max_epochs: 500
  patience: 50
  lr: 1.0e-4
  weight_decay: 1.0e-5
  warmup_epochs: 25
  num_workers: 4
  winsorize_range: [-6.0, 6.0]
  split_ratios: [0.7, 0.2, 0.1]
  seed: 42

isoforms:
  enabled: true
  max_isoforms: 25
  tautomers: { enabled: true, max_tautomers: 25 }
  protonation: { enabled: true, tool: "dimorphite_dl", ph_range: [6.4, 8.4], fallback: "uncharger" }
  neutralization: { enabled: true }
  deduplication: true
```

**`configs/finetune_openadmet.yaml`:**
```yaml
model:
  hidden_dim: 128
  num_gt_layers: 4
  num_heads: 8
  norm: "bn"
  gt_aggregators: ["sum", "mean"]
  aggregators: ["sum", "mean", "max", "std"]
  dropout: 0.3
  act: "gelu"
  gate: true

finetune:
  batch_train: 256
  batch_eval: 1024
  epochs: 2000
  lr: 1.0e-3
  min_lr: 1.0e-5
  weight_decay: 1.0e-5
  warmup_epochs: 25
  grad_clip_norm: 5.0
  split: [0.7, 0.2, 0.1]     # experiment: 70/20/10; production: [0.8, 0.2]
  seed: 1928374650
  loss:
    w_rae: 1.0
    w_huber: 0.25
    w_corr: 0.25
    w_tau: 0.1
    w_r2: 0.1
    huber_delta: 0.5
    tau_temp: 2.0
    rank_pairs: 512
  calibration:
    enabled: true
    clip_delta: 0.2
```

---

---

## Summary

### What to Reuse from Reference Code
- **Golem branch**: validity mask pattern, NaN-aware scaler, YAML config, warmup+cosine schedule
- **CALIBRATE notebook**: model architecture, 5-component loss, training loop structure, calibration, prediction pipeline
- **OpenADMET utils**: log transform functions, endpoint mapping
- **Comparison notebooks**: metrics computation, bootstrap analysis, visualization patterns

### What to Fix vs Reference Code
1. **Scaler data leakage** (golem) — fit on train only
2. **API mismatches** (CALIBRATE) — no `pe` param, no `pe_in_dim`, use `gnm` not `gnn`
3. **Missing `encode()`** (gt-pyg main) — must add before pretrain model can work
4. **Code duplication** (notebooks) — factor into importable modules
5. **Non-functional curriculum learning** (golem) — don't include
6. **Global mutable enumerator** (golem) — use local instances

### Implementation Priority Order
1. Set up golem project structure + pyproject.toml + CLI skeleton + config dataclasses
2. Copy OpenADMET data files from gt-pyg feature branch
3. Implement `transforms.py` and `metrics.py` (self-contained, from OpenADMET utils)
4. Implement `loss.py` (5-component custom loss, self-contained)
5. Implement `utils.py` (seeding, split_data, DataLoader creation)
6. Implement `finetune.py` + `predict.py` (the 4-model experiment — this is the core deliverable)
7. Implement `compare` CLI command
8. **TEST**: Run 10 epochs on 10% subsampled data → verify 4 prediction CSVs → verify comparison table
9. Implement `isoforms.py` + `descriptors.py` + `pretrain.py` (pretraining pipeline)
10. **TEST**: Run pretrain on small data → verify checkpoint → load into fine-tune → verify it works
11. Production runs

**Note**: Steps 6-8 (fine-tuning + comparison) can run without pretraining — just use random init for all 4 models to verify the pipeline works. Then add pretraining (steps 9-10) and re-run with pretrained checkpoints.
