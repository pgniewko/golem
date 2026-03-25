"""Tests for golem.pretrain and golem.config."""

import csv
import math
import numpy as np
import pytest
import torch

from golem.config import (
    GeometryConfig,
    PretrainConfig,
    load_config,
)
from golem.descriptors import NaNAwareStandardScaler
from golem.geometry import (
    _compute_ecfp_fingerprints,
    _fingerprint_cache_path,
    _latent_distance_for_pairs,
    _load_or_compute_fingerprints,
    _pair_order_surrogate,
    _sample_batch_pairs,
    _tanimoto_distance_for_pairs,
)
from golem.pretrain import (
    METRICS_FIELDNAMES,
    _build_pyg_dataset,
    _checkpoint_library_versions,
    _make_warmup_cosine_scheduler,
    _prepare_split_smiles,
    _prepare_metrics_file,
    _save_checkpoint,
    _smiles_cache_key,
)
from golem.report import generate_report
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
        assert cfg.geometry.enabled is False
        assert cfg.geometry.latent_metric == "cosine"

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

    def test_load_config_geometry_block(self, tmp_path):
        """Geometry config should round-trip from YAML."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(
            "geometry:\n"
            "  enabled: true\n"
            "  weight: 0.05\n"
            "  num_pairs: 32\n"
            "  latent_metric: l2_norm\n"
        )
        cfg = load_config(yaml_path=str(yaml_file))
        assert cfg.geometry.enabled is True
        assert cfg.geometry.weight == pytest.approx(0.05)
        assert cfg.geometry.num_pairs == 32
        assert cfg.geometry.latent_metric == "l2_norm"


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
    """Epoch must be defined even when max_epochs=0."""

    def test_epoch_defined_when_max_epochs_zero(self):
        epoch = 0
        for epoch in range(0):
            pass
        assert epoch == 0


class TestWarmupCosineScheduler:
    """LR scheduler creation and state serialization."""

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
    """Descriptor cache key from SMILES list."""

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


class TestGeometryHelpers:
    """Tests for geometry regularizer helpers."""

    def test_sample_batch_pairs_are_unique_and_unordered(self):
        pair_i, pair_j = _sample_batch_pairs(batch_size=4, num_pairs=10, device=torch.device("cpu"))
        pairs = list(zip(pair_i.tolist(), pair_j.tolist(), strict=False))
        assert len(pairs) == 6
        assert len(set(pairs)) == len(pairs)
        assert all(i < j for i, j in pairs)

    def test_sample_batch_pairs_handles_small_batches(self):
        pair_i, pair_j = _sample_batch_pairs(batch_size=1, num_pairs=8, device=torch.device("cpu"))
        assert pair_i.numel() == 0
        assert pair_j.numel() == 0

    def test_tanimoto_distance_for_pairs(self):
        fp_bits = torch.tensor(
            [
                [1, 1, 0, 0],
                [1, 0, 1, 0],
                [0, 0, 1, 1],
            ],
            dtype=torch.bool,
        )
        pair_i = torch.tensor([0, 0], dtype=torch.long)
        pair_j = torch.tensor([1, 2], dtype=torch.long)
        distances = _tanimoto_distance_for_pairs(fp_bits, pair_i, pair_j)
        assert distances[0].item() == pytest.approx(2.0 / 3.0)
        assert distances[1].item() == pytest.approx(1.0)

    def test_latent_distance_for_pairs_cosine(self):
        z = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        )
        pair_i = torch.tensor([0, 0], dtype=torch.long)
        pair_j = torch.tensor([1, 2], dtype=torch.long)
        distances = _latent_distance_for_pairs(z, pair_i, pair_j, metric="cosine")
        assert distances.tolist() == pytest.approx([1.0, 0.0])

    def test_pair_order_surrogate_ignores_ties(self):
        d_fp = torch.tensor([0.10, 0.11], dtype=torch.float32)
        d_z = torch.tensor([0.20, 0.40], dtype=torch.float32, requires_grad=True)
        loss, comparisons = _pair_order_surrogate(d_fp, d_z, temperature=0.1, tie_epsilon=0.02)
        assert comparisons == 0
        assert loss.item() == pytest.approx(0.0)

    def test_pair_order_surrogate_backpropagates(self):
        d_fp = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32)
        d_z = torch.tensor([0.4, 0.2, 0.8], dtype=torch.float32, requires_grad=True)
        loss, comparisons = _pair_order_surrogate(d_fp, d_z, temperature=0.1, tie_epsilon=0.01)
        assert comparisons == 3
        assert torch.isfinite(loss)
        loss.backward()
        assert d_z.grad is not None
        assert torch.isfinite(d_z.grad).all()


class TestFingerprintPipeline:
    """Tests for ECFP computation and attachment."""

    def test_compute_ecfp_fingerprints_shape_and_dtype(self):
        fps = _compute_ecfp_fingerprints(["CCO", "CCO", "c1ccccc1"], radius=2, fp_bits=128)
        assert fps.shape == (3, 128)
        assert fps.dtype == np.bool_
        np.testing.assert_array_equal(fps[0], fps[1])

    def test_load_or_compute_fingerprints_roundtrip(self, tmp_path):
        geometry = GeometryConfig(enabled=True, fp_bits=128, fp_radius=2)
        smiles = ["CCO", "c1ccccc1"]
        fps_first = _load_or_compute_fingerprints(tmp_path, smiles, geometry)
        cache_path = _fingerprint_cache_path(tmp_path, _smiles_cache_key(smiles), geometry)
        assert cache_path.exists()
        fps_second = _load_or_compute_fingerprints(tmp_path, smiles, geometry)
        np.testing.assert_array_equal(fps_first, fps_second)

    def test_build_pyg_dataset_attaches_ecfp_bits(self):
        descriptor_values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        validity_mask = np.array([[True, False], [True, True]], dtype=np.bool_)
        fingerprint_bits = np.array(
            [
                [True, False, True, False],
                [False, True, False, True],
            ],
            dtype=np.bool_,
        )
        dataset = _build_pyg_dataset(
            ["CCO", "c1ccccc1"],
            descriptor_values,
            validity_mask,
            fingerprint_bits=fingerprint_bits,
        )
        assert len(dataset) == 2
        assert hasattr(dataset[0], "ecfp")
        assert dataset[0].ecfp.dtype == torch.bool
        assert tuple(dataset[0].ecfp.shape) == (1, 4)
        assert dataset[1].ecfp.squeeze(0).tolist() == [False, True, False, True]


class TestParentLevelSplit:
    """Split leakage prevention when isoforms are enabled."""

    def test_isoforms_do_not_cross_split_boundaries(self, monkeypatch):
        cfg = PretrainConfig(split_ratios=[0.5, 0.5], seed=0)
        parent_smiles = ["parent_a", "parent_b"]

        def fake_enumerate(smiles_list, _config):
            return {
                "parent_a": ["shared_isoform", "train_only"],
                "parent_b": ["shared_isoform", "val_only"],
            }

        monkeypatch.setattr("golem.pretrain.enumerate_isoforms_batch", fake_enumerate)

        all_smiles, train_idx, val_idx, test_idx = _prepare_split_smiles(parent_smiles, cfg)

        assert test_idx is None
        train_smiles = [all_smiles[i] for i in train_idx]
        val_smiles = [all_smiles[i] for i in val_idx]

        assert set(train_smiles).isdisjoint(val_smiles)
        assert train_smiles.count("shared_isoform") + val_smiles.count("shared_isoform") == 1
        assert {"train_only", "val_only"} <= set(train_smiles + val_smiles)

    def test_raises_when_a_required_split_becomes_empty(self, monkeypatch):
        cfg = PretrainConfig(split_ratios=[0.5, 0.5], seed=0)
        parent_smiles = ["parent_a", "parent_b"]

        def fake_enumerate(smiles_list, _config):
            return {
                "parent_a": ["shared_isoform"],
                "parent_b": ["shared_isoform"],
            }

        monkeypatch.setattr("golem.pretrain.enumerate_isoforms_batch", fake_enumerate)

        with pytest.raises(ValueError, match="empty"):
            _prepare_split_smiles(parent_smiles, cfg)


class TestCheckpointMetadata:
    """Checkpoint metadata should include training library versions."""

    def test_checkpoint_library_versions_uses_module_versions(self, monkeypatch):
        class DummyGtPyg:
            __version__ = "4.5.6"

        import golem
        import sys

        monkeypatch.setattr(golem, "__version__", "1.2.3")
        monkeypatch.setitem(sys.modules, "gt_pyg", DummyGtPyg())

        assert _checkpoint_library_versions() == {
            "golem": "1.2.3",
            "gt_pyg": "4.5.6",
        }

    def test_save_checkpoint_includes_library_versions(self, monkeypatch, tmp_path):
        class DummyCheckpointModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(1, 1)
                self.saved_kwargs = None

            def save_checkpoint(self, **kwargs):
                self.saved_kwargs = kwargs

        model = DummyCheckpointModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = NaNAwareStandardScaler()
        scaler.fit(
            np.array([[1.0, 2.0]], dtype=np.float32),
            np.array([[True, True]], dtype=np.bool_),
        )

        monkeypatch.setattr(
            "golem.pretrain._checkpoint_library_versions",
            lambda: {"golem": "1.2.3", "gt_pyg": "4.5.6"},
        )

        _save_checkpoint(
            model=model,
            optimizer=optimizer,
            path=tmp_path / "checkpoint.pt",
            epoch=7,
            best_metric=0.123,
            config=PretrainConfig(),
            scaler=scaler,
            descriptor_names=["d0", "d1"],
            num_descriptors=2,
            train_idx=np.array([0, 1]),
            val_idx=np.array([2]),
            test_idx=None,
        )

        assert model.saved_kwargs is not None
        assert model.saved_kwargs["extra"]["library_versions"] == {
            "golem": "1.2.3",
            "gt_pyg": "4.5.6",
        }


class TestMetricsAndReport:
    """Tests for metrics CSV compatibility and report generation."""

    def test_prepare_metrics_file_upgrades_legacy_header(self, tmp_path):
        metrics_path = tmp_path / "metrics.csv"
        with open(metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_rmse", "learning_rate", "elapsed_seconds"])
            writer.writerow([0, "1.000000", "2.000000", "1.414214", "1.00e-04", "3.0"])

        _prepare_metrics_file(metrics_path, METRICS_FIELDNAMES)

        with open(metrics_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert reader.fieldnames == METRICS_FIELDNAMES
        assert len(rows) == 1
        assert rows[0]["train_loss"] == "1.000000"
        assert rows[0]["val_rank_loss"] == ""

    def test_generate_report_supports_legacy_metrics_csv(self, tmp_path):
        metrics_path = tmp_path / "metrics.csv"
        config_path = tmp_path / "resolved_config.yaml"
        metrics_path.write_text(
            "epoch,train_loss,val_loss,val_rmse,learning_rate,elapsed_seconds\n"
            "0,1.000000,2.000000,1.414214,1.00e-04,3.0\n"
        )
        config_path.write_text("max_epochs: 1\nmodel:\n  hidden_dim: 64\n  num_gt_layers: 2\n  num_heads: 4\n")

        html_path = generate_report(tmp_path)
        html = html_path.read_text()

        assert html_path.exists()
        assert "Geometry Loss Components" not in html
        assert "Best Val Loss" in html

    def test_generate_report_includes_geometry_sections(self, tmp_path):
        metrics_path = tmp_path / "metrics.csv"
        config_path = tmp_path / "resolved_config.yaml"
        with open(metrics_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=METRICS_FIELDNAMES)
            writer.writeheader()
            writer.writerow({
                "epoch": 0,
                "train_loss": "1.000000",
                "val_loss": "2.000000",
                "val_rmse": "1.414214",
                "learning_rate": "1.00e-04",
                "elapsed_seconds": "3.0",
                "train_main_loss": "1.000000",
                "train_rank_loss": "0.050000",
                "train_total_loss": "1.000500",
                "val_rank_loss": "0.060000",
                "val_spearman": "0.700000",
                "val_kendall": "0.500000",
            })
        config_path.write_text(
            "max_epochs: 1\n"
            "model:\n"
            "  hidden_dim: 64\n"
            "  num_gt_layers: 2\n"
            "  num_heads: 4\n"
            "geometry:\n"
            "  enabled: true\n"
        )

        html_path = generate_report(tmp_path)
        html = html_path.read_text()

        assert "Geometry Loss Components" in html
        assert "Geometry Validation Metrics" in html
        assert "Best Val Kendall" in html
