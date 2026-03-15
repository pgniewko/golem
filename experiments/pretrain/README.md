# Pretrain Experiment

- **Golem version**: 0.1.0 (`gt-pyg` 1.6.1b0)
- **Data**: OpenADMET ([`data/openadmet/train_test_smiles.smi`](../../data/openadmet/train_test_smiles.smi))
- **Best epoch**: 1058 (val loss 0.0501)
- **Training stopped**: epoch 1080 (manually terminated to prevent severe overfitting)

**Note:** This run used an older code version that split data *after* isoform expansion
(parent-level split leakage, fixed in PR #11).

## Files

| File | Description |
|------|-------------|
| `best_checkpoint.pt.gz` | Gzip-compressed model checkpoint (epoch 1058). Decompress with `gunzip best_checkpoint.pt.gz`. |
| `resolved_config.yaml` | Full resolved training configuration |

## Command

```bash
golem pretrain \
  --smiles data/openadmet/train_test_smiles.smi \
  --config configs/pretrain_openadmet.yaml \
  --output experiments/pretrain
```
