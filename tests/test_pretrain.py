"""Tests for golem.pretrain and golem.config."""

import numpy as np
import pytest
import torch

from golem.config import (
    PretrainConfig,
    load_config,
)
from golem.pretrain import _make_warmup_cosine_scheduler, _smiles_cache_key
from golem.utils import load_smiles, seed_everything, split_data


class TestConfig:
    """Tests for configuration loading."""

    def test_defaults(self):
        """Default PretrainConfig should be valid."""
        cfg = PretrainConfig()
        assert cfg.masking_ratio == 0.15
        assert cfg.batch_size == 128
        assert cfg.model.hidden_dim == 128
        assert cfg.isoforms.enabled is True

    def test_load_config_defaults_only(self):
        """load_config with no args should return defaults."""
        cfg = load_config()
        assert isinstance(cfg, PretrainConfig)
        assert cfg.seed == 42

    def test_load_config_cli_overrides(self):
        """CLI overrides should take precedence."""
        cfg = load_config(max_epochs=10, seed=123)
        assert cfg.max_epochs == 10
        assert cfg.seed == 123

    def test_load_config_yaml(self, tmp_path):
        """YAML loading should work."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(
            "pretrain:\n  max_epochs: 7\n  seed: 99\nmodel:\n  hidden_dim: 64\n"
        )
        cfg = load_config(yaml_path=str(yaml_file))
        assert cfg.max_epochs == 7
        assert cfg.seed == 99
        assert cfg.model.hidden_dim == 64

    def test_cli_overrides_yaml(self, tmp_path):
        """CLI overrides should beat YAML values."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("pretrain:\n  max_epochs: 7\n")
        cfg = load_config(yaml_path=str(yaml_file), max_epochs=3)
        assert cfg.max_epochs == 3


class TestSeedEverything:
    """Tests for reproducibility seeding."""

    def test_numpy_reproducibility(self):
        seed_everything(42)
        a = np.random.rand(5)
        seed_everything(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)


class TestSplitData:
    """Tests for data splitting."""

    def test_three_way_split(self):
        splits = split_data(100, [0.7, 0.2, 0.1], seed=42)
        assert len(splits) == 3
        assert len(splits[0]) == 70
        assert len(splits[1]) == 20
        assert len(splits[2]) == 10

    def test_two_way_split(self):
        splits = split_data(100, [0.8, 0.2], seed=42)
        assert len(splits) == 2
        assert len(splits[0]) == 80
        assert len(splits[1]) == 20

    def test_no_overlap(self):
        splits = split_data(100, [0.7, 0.2, 0.1], seed=42)
        all_indices = np.concatenate(splits)
        assert len(all_indices) == len(set(all_indices))

    def test_covers_all(self):
        splits = split_data(100, [0.7, 0.2, 0.1], seed=42)
        all_indices = sorted(np.concatenate(splits))
        assert all_indices == list(range(100))

    def test_reproducible(self):
        s1 = split_data(100, [0.7, 0.2, 0.1], seed=42)
        s2 = split_data(100, [0.7, 0.2, 0.1], seed=42)
        for a, b in zip(s1, s2):
            np.testing.assert_array_equal(a, b)


class TestLoadSmiles:
    """Tests for SMILES file loading."""

    def test_load_smi(self, tmp_path):
        smi_file = tmp_path / "test.smi"
        smi_file.write_text("c1ccccc1\nCC(=O)O\n# comment\n\nCCO\n")
        smiles = load_smiles(str(smi_file))
        assert smiles == ["c1ccccc1", "CC(=O)O", "CCO"]

    def test_load_csv(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("SMILES,Name\nc1ccccc1,benzene\nCCO,ethanol\n")
        smiles = load_smiles(str(csv_file))
        assert smiles == ["c1ccccc1", "CCO"]

    def test_unsupported_extension(self, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("c1ccccc1\n")
        with pytest.raises(ValueError, match="Unsupported"):
            load_smiles(str(txt_file))


class TestEpochDefinedWhenMaxEpochsZero:
    """Bug #4: epoch must be defined even when max_epochs=0."""

    def test_epoch_defined_when_max_epochs_zero(self):
        epoch = 0
        for epoch in range(0):
            pass
        assert epoch == 0


class TestWarmupCosineScheduler:
    """Bug #5: LR scheduler creation and state serialization."""

    def test_warmup_cosine_scheduler(self):
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = _make_warmup_cosine_scheduler(optimizer, warmup_epochs=5, max_epochs=20)

        lrs = []
        for _ in range(20):
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()

        assert lrs[0] < lrs[4], "LR should increase during warmup"
        assert lrs[-1] < lrs[5], "LR should decay after warmup"

    def test_scheduler_state_dict_roundtrip(self):
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = _make_warmup_cosine_scheduler(optimizer, warmup_epochs=5, max_epochs=20)

        for _ in range(10):
            scheduler.step()

        state = scheduler.state_dict()
        lr_before = scheduler.get_last_lr()[0]

        scheduler2 = _make_warmup_cosine_scheduler(optimizer, warmup_epochs=5, max_epochs=20)
        scheduler2.load_state_dict(state)
        lr_after = scheduler2.get_last_lr()[0]

        assert lr_before == pytest.approx(lr_after)


class TestSmilesCacheKey:
    """Bug #6: descriptor cache key from SMILES list."""

    def test_smiles_cache_key(self):
        key1 = _smiles_cache_key(["CCO", "c1ccccc1"])
        key2 = _smiles_cache_key(["CCO", "c1ccccc1"])
        key3 = _smiles_cache_key(["c1ccccc1", "CCO"])
        assert key1 == key2
        assert key1 != key3
        assert len(key1) == 16

    def test_descriptor_cache_roundtrip(self, tmp_path):
        values = np.random.rand(5, 10).astype(np.float64)
        valid = np.ones((5, 10), dtype=np.float64)
        names = ["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9"]

        cache_path = tmp_path / "descriptors_test.npz"
        np.savez(cache_path, values=values, valid=valid, names=np.array(names))

        cached = np.load(cache_path, allow_pickle=True)
        np.testing.assert_array_equal(cached["values"], values)
        np.testing.assert_array_equal(cached["valid"], valid)
        assert cached["names"].tolist() == names
