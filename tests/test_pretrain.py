"""Tests for golem.pretrain and golem.config."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from golem.config import (
    PretrainConfig,
    load_config,
)
from golem.descriptors import NaNAwareStandardScaler
from golem.pretrain import (
    _checkpoint_library_versions,
    _make_warmup_cosine_scheduler,
    _prepare_split_smiles,
    _save_checkpoint,
    _train_one_epoch,
    _validate,
    pretrain,
)
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
        assert cfg.num_workers == 0
        assert cfg.subsample is None

    def test_load_config_defaults_only(self):
        """load_config with no args should return defaults."""
        cfg = load_config()
        assert isinstance(cfg, PretrainConfig)
        assert cfg.seed == 42

    def test_load_config_cli_overrides(self):
        """CLI overrides should take precedence."""
        cfg = load_config(max_epochs=10, seed=123, num_workers=2, subsample=0.25)
        assert cfg.max_epochs == 10
        assert cfg.seed == 123
        assert cfg.num_workers == 2
        assert cfg.subsample == 0.25

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

    def test_load_config_rejects_removed_conformer_knobs(self, tmp_path):
        """Removed conformer keys should fail fast."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(
            "conformers:\n"
            "  energy_window_kcal: 2.73\n"
            "  n_keep: 4\n"
            "  prune_rms: 0.75\n"
            "  timeout_seconds: 15\n"
        )
        with pytest.raises(ValueError, match="Removed config keys"):
            load_config(yaml_path=str(yaml_file))

    def test_load_config_rejects_unknown_conformer_key(self, tmp_path):
        """Unknown conformer keys should fail fast."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("conformers:\n  energy_window_kcl: 2.73\n")
        with pytest.raises(ValueError, match="Unknown config keys"):
            load_config(yaml_path=str(yaml_file))

    def test_load_config_rejects_unknown_three_d_setting_key(self, tmp_path):
        """Unknown 3D descriptor settings should fail fast."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(
            "descriptors:\n"
            "  three_d_settings:\n"
            "    rdkit_include_getway: true\n"
        )
        with pytest.raises(ValueError, match="Unknown config keys"):
            load_config(yaml_path=str(yaml_file))

    def test_load_config_rejects_unknown_descriptor_key(self, tmp_path):
        """Unknown descriptor keys should fail fast."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("descriptors:\n  include_3d_targerts: true\n")
        with pytest.raises(ValueError, match="Unknown config keys"):
            load_config(yaml_path=str(yaml_file))

    def test_load_config_rejects_non_positive_n_generate(self, tmp_path):
        """n_generate must be a positive integer."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("conformers:\n  n_generate: 0\n")
        with pytest.raises(ValueError, match=r"conformers\.n_generate must be >= 1"):
            load_config(yaml_path=str(yaml_file))

    def test_load_config_rejects_non_positive_subsample(self):
        """subsample must be strictly positive."""
        with pytest.raises(ValueError, match=r"subsample must be in the interval \(0.0, 1.0\]"):
            load_config(subsample=0.0)

    def test_load_config_rejects_subsample_above_one(self):
        """subsample must not exceed 1.0."""
        with pytest.raises(ValueError, match=r"subsample must be in the interval \(0.0, 1.0\]"):
            load_config(subsample=1.1)

    def test_load_config_rejects_unknown_pretrain_key(self, tmp_path):
        """Unknown pretrain keys should fail fast."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("pretrain:\n  max_epochz: 10\n")
        with pytest.raises(ValueError, match="Unknown config keys"):
            load_config(yaml_path=str(yaml_file))

    def test_load_config_treats_null_section_as_empty_override(self, tmp_path):
        """Explicit null nested sections should behave like empty overrides."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("descriptors: null\n")
        cfg = load_config(yaml_path=str(yaml_file))
        assert cfg.descriptors.include_2d_targets is True
        assert cfg.descriptors.include_3d_targets is False


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


class TestResolvedConfig:
    def test_pretrain_writes_effective_subsample_to_resolved_config(
        self,
        monkeypatch,
        tmp_path,
    ):
        smiles_file = tmp_path / "mols.smi"
        smiles_file.write_text(
            "\n".join(
                [
                    "C",
                    "CC",
                    "CCC",
                    "CCCC",
                    "CCO",
                    "CCN",
                    "CCCl",
                    "c1ccccc1",
                    "O",
                    "N",
                ]
            )
            + "\n"
        )

        cfg = PretrainConfig()
        cfg.isoforms.enabled = False
        cfg.subsample = 0.5

        def stop_after_config(*args, **kwargs):
            raise RuntimeError("stop after config write")

        monkeypatch.setattr("golem.pretrain.compute_descriptor_targets", stop_after_config)

        with pytest.raises(RuntimeError, match="stop after config write"):
            pretrain(
                smiles_path=str(smiles_file),
                config=cfg,
                output_dir=str(tmp_path / "out"),
            )

        resolved = yaml.safe_load((tmp_path / "out" / "resolved_config.yaml").read_text())
        assert resolved["subsample"] == pytest.approx(0.5)

    def test_pretrain_writes_last_checkpoint_each_run(self, monkeypatch, tmp_path):
        smiles = ["CCO", "CCN", "CCC", "CCCl"]
        desc_values = np.array(
            [
                [1.0, 2.0],
                [2.0, 3.0],
                [3.0, 4.0],
                [4.0, 5.0],
            ],
            dtype=np.float32,
        )
        desc_valid = np.ones_like(desc_values, dtype=np.bool_)
        saved_paths: list[str] = []

        class DummyCheckpointModel(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.prediction = torch.nn.Parameter(
                    torch.zeros((1, kwargs["num_tasks"]), dtype=torch.float32)
                )

            def forward(self, x, edge_index, edge_attr, batch, zero_var=False, return_latent=False):
                pred = self.prediction.expand(x.shape[0], -1)
                log_var = torch.zeros_like(pred)
                if return_latent:
                    return pred, log_var, pred
                return pred, log_var

            def num_parameters(self):
                return sum(p.numel() for p in self.parameters())

            def save_checkpoint(self, path, **kwargs):
                saved_paths.append(str(path))
                path.write_bytes(b"checkpoint")

            def load_weights(self, path, map_location=None):
                return None

        cfg = PretrainConfig(max_epochs=1, patience=5, batch_size=2)
        cfg.isoforms.enabled = False

        monkeypatch.setattr("golem.pretrain.load_smiles", lambda _path: smiles)
        monkeypatch.setattr(
            "golem.pretrain._prepare_split_smiles",
            lambda _smiles, _cfg: (
                smiles,
                np.array([0, 1]),
                np.array([2]),
                np.array([3]),
            ),
        )
        monkeypatch.setattr(
            "golem.pretrain.compute_descriptor_targets",
            lambda *_args, **_kwargs: (desc_values, desc_valid, ["d0", "d1"]),
        )
        monkeypatch.setattr(
            "golem.pretrain._build_pyg_dataset",
            lambda _smiles, values, mask, fingerprint_bits=None: [
                _DummyBatch(
                    torch.tensor(values[i : i + 1], dtype=torch.float32),
                    torch.tensor(mask[i : i + 1], dtype=torch.float32),
                )
                for i in range(len(values))
            ],
        )
        monkeypatch.setattr(
            "golem.pretrain.make_loader",
            lambda dataset, batch_size, shuffle=False, num_workers=0: _DummyLoader(dataset),
        )
        monkeypatch.setattr("golem.pretrain.generate_report", lambda *_args, **_kwargs: None)

        dummy_gt_pyg = SimpleNamespace(
            GraphTransformerNet=DummyCheckpointModel,
            __version__="0.0.test",
        )
        dummy_gt_pyg_data = SimpleNamespace(
            get_atom_feature_dim=lambda: 1,
            get_bond_feature_dim=lambda: 1,
        )
        monkeypatch.setitem(sys.modules, "gt_pyg", dummy_gt_pyg)
        monkeypatch.setitem(sys.modules, "gt_pyg.data", dummy_gt_pyg_data)
        monkeypatch.setattr(
            "golem.pretrain._checkpoint_library_versions",
            lambda: {"golem": "1.2.3", "gt_pyg": "0.0.test"},
        )

        pretrain(
            smiles_path=str(tmp_path / "smiles.smi"),
            config=cfg,
            output_dir=str(tmp_path / "out"),
        )

        assert (tmp_path / "out" / "best_checkpoint.pt").exists()
        assert (tmp_path / "out" / "last_checkpoint.pt").exists()
        assert str(tmp_path / "out" / "last_checkpoint.pt") in saved_paths

    def test_pretrain_argument_subsample_overrides_config_subsample(
        self,
        monkeypatch,
        tmp_path,
    ):
        smiles_file = tmp_path / "mols.smi"
        smiles_file.write_text("C\nCC\nCCC\nCCCC\nCCO\n")
        smiles = ["C", "CC", "CCC", "CCCC", "CCO"]

        cfg = PretrainConfig()
        cfg.isoforms.enabled = False
        cfg.subsample = 0.5

        def stop_after_config(*args, **kwargs):
            raise RuntimeError("stop after config write")

        monkeypatch.setattr(
            "golem.pretrain._prepare_split_smiles",
            lambda _smiles, _cfg: (
                smiles,
                np.array([0, 1, 2]),
                np.array([3]),
                np.array([4]),
            ),
        )
        monkeypatch.setattr("golem.pretrain.compute_descriptor_targets", stop_after_config)

        with pytest.raises(RuntimeError, match="stop after config write"):
            pretrain(
                smiles_path=str(smiles_file),
                config=cfg,
                output_dir=str(tmp_path / "out"),
                subsample=0.25,
            )

        resolved = yaml.safe_load((tmp_path / "out" / "resolved_config.yaml").read_text())
        assert resolved["subsample"] == pytest.approx(0.25)

    def test_pretrain_rejects_invalid_runtime_subsample(self, tmp_path):
        smiles_file = tmp_path / "mols.smi"
        smiles_file.write_text("C\nCC\n")

        cfg = PretrainConfig()
        cfg.isoforms.enabled = False

        with pytest.raises(ValueError, match=r"subsample must be in the interval \(0.0, 1.0\]"):
            pretrain(
                smiles_path=str(smiles_file),
                config=cfg,
                output_dir=str(tmp_path / "out"),
                subsample=1.5,
            )


class TestWarmupCosineScheduler:
    """LR scheduler creation."""

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


class _DummyBatch:
    def __init__(self, targets: torch.Tensor, mask: torch.Tensor):
        self.y = targets
        self.y_mask = mask
        batch_size = targets.shape[0]
        self.x = torch.zeros((batch_size, 1), dtype=torch.float32)
        self.edge_index = torch.zeros((2, 0), dtype=torch.long)
        self.edge_attr = torch.zeros((0, 1), dtype=torch.float32)
        self.batch = torch.arange(batch_size, dtype=torch.long)

    def to(self, device: torch.device):
        self.y = self.y.to(device)
        self.y_mask = self.y_mask.to(device)
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_attr = self.edge_attr.to(device)
        self.batch = self.batch.to(device)
        return self


class _DummyLoader(list):
    ecfp_latent_alignment = None


class _DummyModel(torch.nn.Module):
    def __init__(self, prediction: torch.Tensor, latent_value: float = 2.0):
        super().__init__()
        self.prediction = torch.nn.Parameter(prediction.clone())
        self.latent = torch.nn.Parameter(torch.tensor([[latent_value]], dtype=torch.float32))

    def forward(self, x, edge_index, edge_attr, batch, zero_var=False, return_latent=False):
        pred = self.prediction.expand(x.shape[0], -1)
        log_var = torch.zeros_like(pred)
        if return_latent:
            z = self.latent.expand(x.shape[0], -1)
            return pred, log_var, z
        return pred, log_var


class TestWeightedObjective:
    def test_validate_uses_weighted_objective_and_preserves_descriptor_metrics(
        self,
        monkeypatch,
    ):
        cfg = PretrainConfig()
        cfg.descriptors.loss_weight = 0.0
        cfg.ecfp_latent_alignment.enabled = True
        cfg.ecfp_latent_alignment.weight = 0.5
        cfg.ecfp_latent_alignment.log_rank_metrics = False

        loader = _DummyLoader(
            [
                _DummyBatch(
                    torch.tensor([[1.0]], dtype=torch.float32),
                    torch.tensor([[1.0]], dtype=torch.float32),
                )
            ]
        )
        loader.ecfp_latent_alignment = cfg.ecfp_latent_alignment
        model = _DummyModel(torch.tensor([[0.0]], dtype=torch.float32))

        monkeypatch.setattr(
            "golem.pretrain.compute_alignment_batch",
            lambda batch, z, alignment_cfg, deterministic_pairs=False: (
                z.mean(),
                torch.tensor([0.25], dtype=torch.float32),
                torch.tensor([0.5], dtype=torch.float32),
            ),
        )

        (
            objective_loss,
            descriptor_loss,
            rmse,
            alignment_loss,
            alignment_spearman,
            alignment_kendall,
        ) = _validate(
            model,
            loader,
            cfg.descriptors.loss_weight,
            torch.device("cpu"),
        )

        assert objective_loss == pytest.approx(1.0)
        assert descriptor_loss == pytest.approx(1.0)
        assert rmse == pytest.approx(1.0)
        assert alignment_loss == pytest.approx(2.0)
        assert np.isnan(alignment_spearman)
        assert np.isnan(alignment_kendall)

    def test_train_one_epoch_can_optimize_alignment_with_empty_descriptor_mask(
        self,
        monkeypatch,
    ):
        cfg = PretrainConfig()
        cfg.descriptors.loss_weight = 0.0
        cfg.ecfp_latent_alignment.enabled = True
        cfg.ecfp_latent_alignment.weight = 0.5
        cfg.ecfp_latent_alignment.log_rank_metrics = False

        loader = _DummyLoader(
            [
                _DummyBatch(
                    torch.tensor([[0.0]], dtype=torch.float32),
                    torch.tensor([[0.0]], dtype=torch.float32),
                )
            ]
        )
        loader.ecfp_latent_alignment = cfg.ecfp_latent_alignment
        model = _DummyModel(torch.tensor([[0.0]], dtype=torch.float32))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        monkeypatch.setattr(
            "golem.pretrain.compute_alignment_batch",
            lambda batch, z, alignment_cfg, deterministic_pairs=False: (
                z.mean(),
                torch.tensor([0.25], dtype=torch.float32),
                torch.tensor([0.5], dtype=torch.float32),
            ),
        )

        train_loss, train_descriptor_loss, train_alignment_loss = _train_one_epoch(
            model,
            loader,
            optimizer,
            masking_ratio=1.0,
            descriptor_loss_weight=cfg.descriptors.loss_weight,
            device=torch.device("cpu"),
        )

        assert train_loss == pytest.approx(1.0)
        assert train_descriptor_loss == pytest.approx(0.0)
        assert train_alignment_loss == pytest.approx(2.0)

    def test_validate_ignores_alignment_when_no_pairs_can_be_sampled(
        self,
        monkeypatch,
    ):
        cfg = PretrainConfig()
        cfg.descriptors.loss_weight = 0.0
        cfg.ecfp_latent_alignment.enabled = True
        cfg.ecfp_latent_alignment.weight = 0.5
        cfg.ecfp_latent_alignment.log_rank_metrics = True

        loader = _DummyLoader(
            [
                _DummyBatch(
                    torch.tensor([[0.0]], dtype=torch.float32),
                    torch.tensor([[0.0]], dtype=torch.float32),
                )
            ]
        )
        loader.ecfp_latent_alignment = cfg.ecfp_latent_alignment
        loader[0].ecfp = torch.tensor([[True, False]], dtype=torch.bool)
        model = _DummyModel(torch.tensor([[0.0]], dtype=torch.float32))

        monkeypatch.setattr(
            "golem.pretrain.compute_alignment_batch",
            lambda batch, z, alignment_cfg, deterministic_pairs=False: (
                z.sum() * 0.0,
                torch.empty(0, dtype=torch.float32),
                torch.empty(0, dtype=torch.float32),
            ),
        )

        (
            objective_loss,
            descriptor_loss,
            rmse,
            alignment_loss,
            alignment_spearman,
            alignment_kendall,
        ) = _validate(
            model,
            loader,
            cfg.descriptors.loss_weight,
            torch.device("cpu"),
        )

        assert np.isnan(objective_loss)
        assert np.isnan(descriptor_loss)
        assert np.isnan(rmse)
        assert np.isnan(alignment_loss)
        assert np.isnan(alignment_spearman)
        assert np.isnan(alignment_kendall)


class TestParentLevelSplit:
    """Split leakage prevention when isoforms are enabled."""

    def test_synonymous_parent_smiles_are_collapsed_before_split(self):
        cfg = PretrainConfig(split_ratios=[0.5, 0.5], seed=0)
        cfg.isoforms.enabled = False

        all_smiles, train_idx, val_idx, test_idx = _prepare_split_smiles(
            ["CCO", "OCC", "CCN"],
            cfg,
        )

        assert test_idx is None
        assert "OCC" not in all_smiles
        assert all_smiles.count("CCO") == 1
        train_smiles = [all_smiles[i] for i in train_idx]
        val_smiles = [all_smiles[i] for i in val_idx]
        assert set(train_smiles).isdisjoint(val_smiles)

    def test_isoforms_do_not_cross_split_boundaries(self, monkeypatch):
        cfg = PretrainConfig(split_ratios=[0.5, 0.5], seed=0)
        parent_smiles = ["CCO", "CCN"]

        def fake_enumerate(smiles_list, _config):
            return {
                "CCO": ["shared_isoform", "train_only"],
                "CCN": ["shared_isoform", "val_only"],
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
        parent_smiles = ["CCO", "CCN"]

        def fake_enumerate(smiles_list, _config):
            return {
                "CCO": ["shared_isoform"],
                "CCN": ["shared_isoform"],
            }

        monkeypatch.setattr("golem.pretrain.enumerate_isoforms_batch", fake_enumerate)

        with pytest.raises(ValueError, match="empty"):
            _prepare_split_smiles(parent_smiles, cfg)

    def test_rejects_batch_norm_training_with_singleton_final_batch(
        self,
        monkeypatch,
        tmp_path,
    ):
        smiles = ["CCO", "CCN", "CCC", "CCCl"]
        desc_values = np.array(
            [
                [1.0, 2.0],
                [2.0, 3.0],
                [3.0, 4.0],
                [4.0, 5.0],
            ],
            dtype=np.float32,
        )
        desc_valid = np.ones_like(desc_values, dtype=np.bool_)

        cfg = PretrainConfig(max_epochs=1, batch_size=2)
        cfg.isoforms.enabled = False
        cfg.model.norm = "bn"

        monkeypatch.setattr("golem.pretrain.load_smiles", lambda _path: smiles)
        monkeypatch.setattr(
            "golem.pretrain._prepare_split_smiles",
            lambda _smiles, _cfg: (
                smiles,
                np.array([0, 1, 2]),
                np.array([3]),
                None,
            ),
        )
        monkeypatch.setattr(
            "golem.pretrain.compute_descriptor_targets",
            lambda *_args, **_kwargs: (desc_values, desc_valid, ["d0", "d1"]),
        )
        monkeypatch.setattr(
            "golem.pretrain._build_pyg_dataset",
            lambda _smiles, values, mask, fingerprint_bits=None: [
                _DummyBatch(
                    torch.tensor(values[i : i + 1], dtype=torch.float32),
                    torch.tensor(mask[i : i + 1], dtype=torch.float32),
                )
                for i in range(len(values))
            ],
        )

        with pytest.raises(ValueError, match="singleton final training batch"):
            pretrain(
                smiles_path=str(tmp_path / "smiles.smi"),
                config=cfg,
                output_dir=str(tmp_path / "out"),
            )

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
