#! /bin/bash -x

.venv/bin/golem pretrain --smiles data/openadmet/expansion_rx/train_test_smiles.smi --config configs/golem-2d.yaml --output experiments/test/golem-2d-only-test --max-epochs 10 --batch-size 128 --seed 123

.venv/bin/golem pretrain --smiles data/openadmet/expansion_rx/train_test_smiles.smi --config configs/golem-2d-plus-latent.yaml --output experiments/test/golem-2d-plus-latent --max-epochs 10 --batch-size 128 --seed 123

.venv/bin/golem pretrain --smiles data/openadmet/expansion_rx/train_test_smiles.smi --config configs/golem-2d-3d-plus-latent.yaml --output experiments/test/golem-2d-3d-plus-latent-xs --max-epochs 10 --batch-size 128 --seed 123 --subsample 0.5
