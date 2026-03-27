"""Focused tests for ECFP-latent alignment."""

import numpy as np
import torch

from golem.config import ECFPLatentAlignmentConfig, load_config
from golem.ecfp_latent_alignment import (
    compute_alignment_batch,
    load_or_compute_fingerprints,
)
from golem.pretrain import _build_pyg_dataset, _validate


class _Batch:
    def __init__(self, y: torch.Tensor, y_mask: torch.Tensor, ecfp: torch.Tensor):
        self.y = y
        self.y_mask = y_mask
        self.ecfp = ecfp
        self.x = torch.zeros((y.shape[0], 1), dtype=torch.float32)
        self.edge_index = torch.zeros((2, 0), dtype=torch.long)
        self.edge_attr = None
        self.batch = torch.arange(y.shape[0], dtype=torch.long)

    def to(self, device: torch.device):
        self.y = self.y.to(device)
        self.y_mask = self.y_mask.to(device)
        self.ecfp = self.ecfp.to(device)
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.batch = self.batch.to(device)
        return self


class _Loader(list):
    pass


class _Model(torch.nn.Module):
    def __init__(self, pred: torch.Tensor, z: torch.Tensor):
        super().__init__()
        self.pred = pred
        self.z = z

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
        zero_var: bool = False,
        return_latent: bool = False,
    ):
        pred = self.pred.to(x.device)
        log_var = torch.zeros_like(pred)
        if return_latent:
            return pred, log_var, self.z.to(x.device)
        return pred, log_var


def test_load_config_reads_ecfp_latent_alignment_block(tmp_path):
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text(
        "ecfp_latent_alignment:\n"
        "  enabled: true\n"
        "  weight: 0.05\n"
        "  num_pairs: 32\n"
    )
    cfg = load_config(yaml_path=str(yaml_file))
    assert cfg.ecfp_latent_alignment.enabled is True
    assert cfg.ecfp_latent_alignment.weight == 0.05
    assert cfg.ecfp_latent_alignment.num_pairs == 32


def test_load_or_compute_fingerprints_roundtrip(tmp_path):
    cfg = ECFPLatentAlignmentConfig(enabled=True, fp_bits=128, fp_radius=2)
    smiles = ["CCO", "c1ccccc1"]
    first = load_or_compute_fingerprints(tmp_path, smiles, cfg)
    second = load_or_compute_fingerprints(tmp_path, smiles, cfg)
    np.testing.assert_array_equal(first, second)


def test_compute_alignment_batch_backpropagates():
    batch = _Batch(
        y=torch.zeros((3, 1), dtype=torch.float32),
        y_mask=torch.ones((3, 1), dtype=torch.bool),
        ecfp=torch.tensor(
            [
                [True, True, False, False],
                [True, False, True, False],
                [False, False, True, True],
            ],
            dtype=torch.bool,
        ),
    )
    z = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    cfg = ECFPLatentAlignmentConfig(enabled=True, num_pairs=3)

    loss, d_fp, d_z = compute_alignment_batch(batch, z, cfg, deterministic_pairs=True)

    assert d_fp.shape == (3,)
    assert d_z.shape == (3,)
    assert torch.isfinite(loss)
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_build_pyg_dataset_attaches_ecfp_bits():
    dataset = _build_pyg_dataset(
        ["CCO", "c1ccccc1"],
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.array([[True, False], [True, True]], dtype=np.bool_),
        fingerprint_bits=np.array(
            [
                [True, False, True, False],
                [False, True, False, True],
            ],
            dtype=np.bool_,
        ),
    )

    assert hasattr(dataset[0], "ecfp")
    assert dataset[0].ecfp.dtype == torch.bool
    assert dataset[1].ecfp.squeeze(0).tolist() == [False, True, False, True]


def test_validate_reports_deterministic_alignment_metrics():
    loader = _Loader(
        [
            _Batch(
                y=torch.zeros((4, 1), dtype=torch.float32),
                y_mask=torch.ones((4, 1), dtype=torch.bool),
                ecfp=torch.tensor(
                    [
                        [True, True, True, True],
                        [True, True, False, False],
                        [False, True, True, False],
                        [True, False, False, False],
                    ],
                    dtype=torch.bool,
                ),
            )
        ]
    )
    loader.ecfp_latent_alignment = ECFPLatentAlignmentConfig(enabled=True, num_pairs=3)
    model = _Model(
        pred=torch.zeros((4, 1), dtype=torch.float32),
        z=torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.6, 0.4],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
    )

    first = _validate(model, loader, torch.device("cpu"))
    second = _validate(model, loader, torch.device("cpu"))

    assert np.isfinite(first[2])
    assert np.isfinite(first[3])
    assert np.isfinite(first[4])
    assert first == second
