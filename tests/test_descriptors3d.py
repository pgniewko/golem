"""Tests for optional 3D descriptor target generation."""

import math

import numpy as np
import pytest

from golem.config import ConformerConfig, DescriptorConfig
from golem.conformers import ConformerEnsemble, _molecule_seed
from golem.descriptors3d import (
    compute_aggregated_3d_descriptors,
    compute_descriptor_targets,
)


class _FakeCalculator:
    """Minimal fake calculator with stable columns and per-conformer outputs."""

    def __init__(self, columns, outputs):
        self.columns = columns
        self.outputs = outputs

    def __call__(self, mol, conformer_id=-1):
        value = self.outputs[(mol, conformer_id)]
        if isinstance(value, Exception):
            raise value
        return np.asarray(value, dtype=np.float64)


def test_molecule_seed_fits_rdkit_signed_32bit_range():
    for smiles, seed in [("CCO", 42), ("c1ccccc1", 123), ("CC(=O)O", 2_147_483_647)]:
        value = _molecule_seed(smiles, seed)
        assert 0 <= value <= 0x7FFFFFFF


def test_compute_descriptor_targets_returns_2d_only_when_3d_disabled(monkeypatch):
    values_2d = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mask_2d = np.array([[True, False], [True, True]], dtype=np.bool_)
    names_2d = ["a", "b"]

    monkeypatch.setattr(
        "golem.descriptors3d.compute_mordred_descriptors",
        lambda smiles_list: (values_2d, mask_2d, names_2d),
    )
    monkeypatch.setattr(
        "golem.descriptors3d.compute_aggregated_3d_descriptors",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("3D path should not run")),
    )

    values, mask, names = compute_descriptor_targets(
        ["CCO", "c1ccccc1"],
        DescriptorConfig(include_2d_targets=True, use_3d_targets=False),
        ConformerConfig(),
        seed=42,
    )

    np.testing.assert_array_equal(values, values_2d)
    np.testing.assert_array_equal(mask, mask_2d)
    assert names == names_2d


def test_compute_descriptor_targets_returns_3d_only_when_2d_disabled(monkeypatch):
    values_3d = np.array([[10.0, 11.0], [12.0, 13.0]], dtype=np.float32)
    mask_3d = np.array([[True, True], [False, True]], dtype=np.bool_)
    names_3d = ["rdkit3d:x", "usrcat:y"]

    monkeypatch.setattr(
        "golem.descriptors3d.compute_mordred_descriptors",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("2D path should not run")
        ),
    )
    monkeypatch.setattr(
        "golem.descriptors3d.compute_aggregated_3d_descriptors",
        lambda smiles_list, three_d, conformers, seed: (values_3d, mask_3d, names_3d),
    )

    values, mask, names = compute_descriptor_targets(
        ["CCO", "c1ccccc1"],
        DescriptorConfig(include_2d_targets=False, use_3d_targets=True),
        ConformerConfig(),
        seed=42,
    )

    np.testing.assert_array_equal(values, values_3d)
    np.testing.assert_array_equal(mask, mask_3d)
    assert names == names_3d


def test_compute_descriptor_targets_appends_3d_block_when_enabled(monkeypatch):
    values_2d = np.array([[1.0], [2.0]], dtype=np.float32)
    mask_2d = np.array([[True], [False]], dtype=np.bool_)
    names_2d = ["mordred:a"]
    values_3d = np.array([[10.0, 11.0], [12.0, 13.0]], dtype=np.float32)
    mask_3d = np.array([[True, True], [False, True]], dtype=np.bool_)
    names_3d = ["rdkit3d:x", "usrcat:y"]

    monkeypatch.setattr(
        "golem.descriptors3d.compute_mordred_descriptors",
        lambda smiles_list: (values_2d, mask_2d, names_2d),
    )
    monkeypatch.setattr(
        "golem.descriptors3d.compute_aggregated_3d_descriptors",
        lambda smiles_list, three_d, conformers, seed: (values_3d, mask_3d, names_3d),
    )

    values, mask, names = compute_descriptor_targets(
        ["CCO", "c1ccccc1"],
        DescriptorConfig(include_2d_targets=True, use_3d_targets=True),
        ConformerConfig(),
        seed=42,
    )

    np.testing.assert_array_equal(
        values,
        np.array([[1.0, 10.0, 11.0], [2.0, 12.0, 13.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        mask,
        np.array([[True, True, True], [False, False, True]], dtype=np.bool_),
    )
    assert names == ["mordred:a", "rdkit3d:x", "usrcat:y"]


def test_compute_descriptor_targets_requires_at_least_one_target_family():
    with pytest.raises(ValueError, match="At least one descriptor target family"):
        compute_descriptor_targets(
            ["CCO"],
            DescriptorConfig(include_2d_targets=False, use_3d_targets=False),
            ConformerConfig(),
            seed=42,
        )


def test_compute_aggregated_3d_descriptors_masks_family_and_conformer_failures(monkeypatch):
    decay = math.exp(-1.0)

    def fake_generate(smiles, config, *, seed):
        if smiles == "no_conf":
            return None
        return ConformerEnsemble(
            mol=smiles,
            conformer_ids=[0, 1],
            relative_energies_kcal=np.array([0.0, 0.593], dtype=np.float64),
        )

    calculators = {
        "rdkit3d": _FakeCalculator(
            ["a", "b"],
            {
                ("ok", 0): [1.0, np.nan],
                ("ok", 1): [3.0, 5.0],
                ("partial", 0): [2.0, 4.0],
                ("partial", 1): [4.0, 6.0],
            },
        ),
        "usrcat": _FakeCalculator(
            ["u"],
            {
                ("ok", 0): [10.0],
                ("ok", 1): [20.0],
                ("partial", 0): RuntimeError("boom"),
                ("partial", 1): RuntimeError("boom"),
            },
        ),
        "electroshape": _FakeCalculator(
            ["e"],
            {
                ("ok", 0): [7.0],
                ("ok", 1): [9.0],
                ("partial", 0): [1.0],
                ("partial", 1): [3.0],
            },
        ),
    }

    monkeypatch.setattr(
        "golem.descriptors3d.generate_conformer_ensemble",
        fake_generate,
    )
    monkeypatch.setattr(
        "golem.descriptors3d._build_3d_calculators",
        lambda three_d: calculators,
    )

    values, mask, names = compute_aggregated_3d_descriptors(
        ["ok", "partial", "no_conf"],
        DescriptorConfig().three_d,
        ConformerConfig(),
        seed=42,
    )

    expected_ok = np.array(
        [
            (1.0 + 3.0 * decay) / (1.0 + decay),
            5.0,
            (10.0 + 20.0 * decay) / (1.0 + decay),
            (7.0 + 9.0 * decay) / (1.0 + decay),
        ],
        dtype=np.float32,
    )
    expected_partial = np.array(
        [
            (2.0 + 4.0 * decay) / (1.0 + decay),
            (4.0 + 6.0 * decay) / (1.0 + decay),
            0.0,
            (1.0 + 3.0 * decay) / (1.0 + decay),
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(values[0], expected_ok, rtol=1e-6)
    np.testing.assert_allclose(values[1], expected_partial, rtol=1e-6)
    np.testing.assert_array_equal(
        values[2],
        np.zeros(4, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        mask,
        np.array(
            [
                [True, True, True, True],
                [True, True, False, True],
                [False, False, False, False],
            ],
            dtype=np.bool_,
        ),
    )
    assert names == ["rdkit3d:a", "rdkit3d:b", "usrcat:u", "electroshape:e"]
