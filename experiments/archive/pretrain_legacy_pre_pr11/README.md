# Legacy Pretrain Experiment

This directory is an archived artifact from a pre-fix training run.

- **Status**: legacy, kept only for provenance
- **Golem version**: 0.1.0 (`gt-pyg` 1.6.1b0)
- **Data**: OpenADMET ([`data/openadmet/expansion_rx/train_test_smiles.smi`](../../../data/openadmet/expansion_rx/train_test_smiles.smi))
- **Best epoch**: 1058 (val loss 0.0501)
- **Training stopped**: epoch 1080 (manually terminated to prevent severe overfitting)

Do not use this run as a current baseline or reproduction target.

Reasons:
- It was trained before the parent-level split leakage fix.
- `resolved_config.yaml` reflects the older config schema from that run.
- The checkpoint is retained for historical comparison only.

## Files

| File | Description |
|------|-------------|
| `best_checkpoint.pt` | Historical checkpoint from the legacy run. |
| `best_checkpoint.pt.gz` | Gzip-compressed copy of the same checkpoint. |
| `resolved_config.yaml` | Legacy config snapshot from that run. |
| `metrics.csv` | Legacy training metrics. |
| `pretrain.log` | Legacy training log. |

## Historical Command

```bash
golem pretrain \
  --smiles data/openadmet/expansion_rx/train_test_smiles.smi \
  --config configs/golem-2d.yaml \
  --output experiments/pretrain
```
