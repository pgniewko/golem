"""Focused tests for ECFP-latent alignment."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from golem.config import ECFPLatentAlignmentConfig, load_config
from golem.ecfp_latent_alignment import (
    compute_alignment_batch,
    load_or_compute_fingerprints,
)
from golem.pretrain import _build_pyg_dataset


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


def test_load_or_compute_fingerprints_uses_disk_cache(tmp_path):
    cfg = ECFPLatentAlignmentConfig(enabled=True, fp_bits=128, fp_radius=2)
    smiles = ["CCO", "c1ccccc1"]
    first = load_or_compute_fingerprints(tmp_path, smiles, cfg)

    with patch(
        "golem.ecfp_latent_alignment.AllChem.GetMorganFingerprintAsBitVect",
        side_effect=AssertionError("cache miss"),
    ):
        second = load_or_compute_fingerprints(tmp_path, smiles, cfg)

    assert len(list(tmp_path.glob("ecfp_r2_b128_*.npz"))) == 1
    np.testing.assert_array_equal(first, second)


def test_compute_alignment_batch_matches_expected_pair_order_loss():
    batch = SimpleNamespace(
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

    torch.testing.assert_close(
        d_fp,
        torch.tensor([0.66666663, 1.0, 0.66666663], dtype=torch.float32),
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        d_z,
        torch.tensor([1.0, 0.29289323, 0.29289323], dtype=torch.float32),
        atol=1e-6,
        rtol=1e-6,
    )
    assert abs(loss.item() - 3.8825321197509766) < 1e-6
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
