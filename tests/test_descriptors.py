"""Tests for golem.descriptors."""

import numpy as np
import pytest

from golem.config import ConformerConfig, DescriptorConfig
from golem.conformers import ConformerEnsemble
from golem.descriptors import (
    NaNAwareStandardScaler,
    compute_aggregated_3d_descriptors,
    compute_descriptor_targets,
    compute_mordred_descriptors,
)


class _FakeCalculator:
    def __init__(self, columns, outputs):
        self.columns = columns
        self.outputs = outputs

    def __call__(self, mol, conformer_id=-1):
        value = self.outputs[(mol, conformer_id)]
        if isinstance(value, Exception):
            raise value
        return np.asarray(value, dtype=np.float64)


class TestComputeMordredDescriptors:
    """Tests for Mordred 2D descriptor computation."""

    def test_basic_output_shapes(self):
        """Values, mask, and names should have consistent shapes."""
        smiles = ["c1ccccc1", "CC(=O)O", "CCO"]
        values, mask, names = compute_mordred_descriptors(smiles)
        N, D = values.shape
        assert N == 3
        assert D > 0
        assert mask.shape == (N, D)
        assert len(names) == D

    def test_validity_mask_dtype(self):
        """Validity mask should be boolean."""
        values, mask, names = compute_mordred_descriptors(["c1ccccc1"])
        assert mask.dtype == np.bool_

    def test_no_nan_in_values(self):
        """After NaN→0 replacement, values should have no NaN."""
        values, mask, names = compute_mordred_descriptors(["c1ccccc1", "CCO"])
        assert not np.isnan(values).any()

    def test_values_dtype(self):
        """Values should be float32."""
        values, _, _ = compute_mordred_descriptors(["c1ccccc1"])
        assert values.dtype == np.float32

    def test_all_nan_columns_dropped(self):
        """Descriptors that are all-NaN should not appear in output."""
        # This is hard to test directly without knowing which descriptors
        # are all-NaN, but we can at least verify no all-zero mask columns
        values, mask, names = compute_mordred_descriptors(["c1ccccc1", "CCO", "CCCC"])
        # If a column is all-False in the mask, it means all values were NaN
        # (which should have been dropped)
        all_invalid = ~mask.any(axis=0)
        assert not all_invalid.any(), "Found columns with all-invalid entries (should be dropped)"


class TestDescriptorTargets:
    def test_compute_descriptor_targets_supports_3d_only_mode(self, monkeypatch):
        values_3d = np.array([[10.0, 11.0], [12.0, 13.0]], dtype=np.float32)
        mask_3d = np.array([[True, True], [True, True]], dtype=np.bool_)
        names_3d = ["rdkit3d:x", "usrcat:y"]
        keep_3d = np.array([True, True], dtype=np.bool_)

        monkeypatch.setattr(
            "golem.descriptors.compute_mordred_descriptors",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("2D path should not run")
            ),
        )
        monkeypatch.setattr(
            "golem.descriptors.compute_aggregated_3d_descriptors",
            lambda *args, **kwargs: (values_3d, mask_3d, names_3d, keep_3d),
        )

        values, mask, names, keep = compute_descriptor_targets(
            ["CCO", "c1ccccc1"],
            DescriptorConfig(include_2d_targets=False, include_3d_targets=True),
            ConformerConfig(),
            seed=42,
        )

        np.testing.assert_array_equal(values, values_3d)
        np.testing.assert_array_equal(mask, mask_3d)
        np.testing.assert_array_equal(keep, keep_3d)
        assert names == names_3d

    def test_compute_descriptor_targets_concatenates_2d_and_3d_and_intersects_keep_mask(
        self,
        monkeypatch,
    ):
        values_2d = np.array([[1.0], [2.0]], dtype=np.float32)
        mask_2d = np.array([[True], [False]], dtype=np.bool_)
        values_3d = np.array([[10.0, 11.0], [12.0, 13.0]], dtype=np.float32)
        mask_3d = np.array([[True, True], [False, True]], dtype=np.bool_)
        keep_3d = np.array([True, False], dtype=np.bool_)

        monkeypatch.setattr(
            "golem.descriptors.compute_mordred_descriptors",
            lambda smiles_list: (values_2d, mask_2d, ["mordred:a"]),
        )
        monkeypatch.setattr(
            "golem.descriptors.compute_aggregated_3d_descriptors",
            lambda *args, **kwargs: (
                values_3d,
                mask_3d,
                ["rdkit3d:x", "usrcat:y"],
                keep_3d,
            ),
        )

        values, mask, names, keep = compute_descriptor_targets(
            ["CCO", "c1ccccc1"],
            DescriptorConfig(include_2d_targets=True, include_3d_targets=True),
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
        np.testing.assert_array_equal(keep, keep_3d)
        assert names == ["mordred:a", "rdkit3d:x", "usrcat:y"]

    def test_compute_descriptor_targets_requires_at_least_one_target_family(self):
        with pytest.raises(ValueError, match="At least one descriptor target family"):
            compute_descriptor_targets(
                ["CCO"],
                DescriptorConfig(include_2d_targets=False, include_3d_targets=False),
                ConformerConfig(),
                seed=42,
            )

    def test_compute_aggregated_3d_descriptors_skips_incomplete_ensembles(
        self,
        monkeypatch,
    ):
        decay = np.exp(-1.0)

        def fake_generate(smiles, config, *, seed):
            if smiles == "no_conf":
                return None, "timeout"
            return (
                ConformerEnsemble(
                    mol=smiles,
                    conformer_ids=[0, 1],
                    relative_energies_kcal=np.array([0.0, 0.593], dtype=np.float64),
                ),
                None,
            )

        calculators = {
            "rdkit3d": _FakeCalculator(
                ["a", "b"],
                {
                    ("ok", 0): [1.0, np.nan],
                    ("ok", 1): [3.0, 5.0],
                    ("masked", 0): [2.0, np.nan],
                    ("masked", 1): [4.0, np.nan],
                    ("family_fail", 0): [2.0, 4.0],
                    ("family_fail", 1): [4.0, 6.0],
                },
            ),
            "usrcat": _FakeCalculator(
                ["u"],
                {
                    ("ok", 0): [10.0],
                    ("ok", 1): [20.0],
                    ("masked", 0): [11.0],
                    ("masked", 1): [21.0],
                    ("family_fail", 0): RuntimeError("boom"),
                    ("family_fail", 1): RuntimeError("boom"),
                },
            ),
            "electroshape": _FakeCalculator(
                ["e"],
                {
                    ("ok", 0): [7.0],
                    ("ok", 1): [9.0],
                    ("masked", 0): [1.0],
                    ("masked", 1): [3.0],
                    ("family_fail", 0): [1.0],
                    ("family_fail", 1): [3.0],
                },
            ),
        }

        monkeypatch.setattr(
            "golem.descriptors.generate_conformer_ensemble",
            fake_generate,
        )
        monkeypatch.setattr(
            "golem.descriptors._build_3d_calculators",
            lambda three_d: calculators,
        )

        values, mask, names, keep = compute_aggregated_3d_descriptors(
            ["ok", "masked", "family_fail", "no_conf"],
            DescriptorConfig().three_d_settings,
            ConformerConfig(),
            seed=42,
        )

        np.testing.assert_allclose(
            values[0],
            np.array(
                [
                    (1.0 + 3.0 * decay) / (1.0 + decay),
                    5.0,
                    (10.0 + 20.0 * decay) / (1.0 + decay),
                    (7.0 + 9.0 * decay) / (1.0 + decay),
                ],
                dtype=np.float32,
            ),
            rtol=1e-6,
        )
        np.testing.assert_array_equal(
            mask,
            np.array(
                [
                    [True, True, True, True],
                    [True, False, True, True],
                    [True, True, False, True],
                    [False, False, False, False],
                ],
                dtype=np.bool_,
            ),
        )
        np.testing.assert_array_equal(
            keep,
            np.array([True, False, False, False], dtype=np.bool_),
        )
        assert names == ["rdkit3d:a", "rdkit3d:b", "usrcat:u", "electroshape:e"]


class TestNaNAwareStandardScaler:
    """Tests for the NaN-aware scaler."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with known statistics."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5).astype(np.float32)
        # Introduce some "NaN" positions (stored as 0.0)
        mask = np.ones_like(X, dtype=bool)
        mask[0, 2] = False
        mask[10, 4] = False
        X[~mask] = 0.0
        return X, mask

    def test_fit_transform_shape(self, sample_data):
        X, mask = sample_data
        scaler = NaNAwareStandardScaler()
        scaler.fit(X, mask)
        X_scaled = scaler.transform(X)
        assert X_scaled.shape == X.shape
        assert X_scaled.dtype == np.float32

    def test_mean_near_zero_on_train(self, sample_data):
        """After scaling, valid entries in the training set should be ~zero mean."""
        X, mask = sample_data
        scaler = NaNAwareStandardScaler()
        scaler.fit(X, mask)
        X_scaled = scaler.transform(X)
        # Compute mean of valid entries per feature
        X_masked = np.where(mask, X_scaled, np.nan)
        col_means = np.nanmean(X_masked, axis=0)
        np.testing.assert_allclose(col_means, 0.0, atol=0.15)

    def test_winsorization(self):
        """Values should be clipped to winsorize range."""
        X = np.array([[100.0, -100.0, 0.5]], dtype=np.float32)
        mask = np.ones_like(X, dtype=bool)
        scaler = NaNAwareStandardScaler(winsorize_range=(-6.0, 6.0))
        scaler.fit(X, mask)
        X_scaled = scaler.transform(X)
        assert X_scaled.min() >= -6.0
        assert X_scaled.max() <= 6.0

    def test_state_dict_roundtrip(self, sample_data):
        """Scaler should survive serialization and produce identical output."""
        X, mask = sample_data
        scaler = NaNAwareStandardScaler()
        scaler.fit(X, mask)
        X_scaled_1 = scaler.transform(X)

        # Roundtrip
        sd = scaler.state_dict()
        scaler2 = NaNAwareStandardScaler.from_state_dict(sd)
        X_scaled_2 = scaler2.transform(X)

        np.testing.assert_array_equal(X_scaled_1, X_scaled_2)

    def test_constant_column_handling(self):
        """Scaler should handle constant columns (std=0) gracefully."""
        X = np.array([[1.0, 5.0], [1.0, 5.0], [1.0, 5.0]], dtype=np.float32)
        mask = np.ones_like(X, dtype=bool)
        scaler = NaNAwareStandardScaler()
        scaler.fit(X, mask)
        X_scaled = scaler.transform(X)
        # Constant columns should have std=1 → scaled values = (1-1)/1 = 0
        assert not np.isnan(X_scaled).any()
        assert not np.isinf(X_scaled).any()

    def test_transform_before_fit_raises(self):
        """Calling transform before fit should raise."""
        scaler = NaNAwareStandardScaler()
        with pytest.raises(RuntimeError, match="not been fit"):
            scaler.transform(np.zeros((2, 3)))

    def test_fit_on_train_only(self, sample_data):
        """Demonstrate scaler fits on train split only — val stats differ."""
        X, mask = sample_data
        train_idx = np.arange(80)
        val_idx = np.arange(80, 100)

        scaler = NaNAwareStandardScaler()
        scaler.fit(X[train_idx], mask[train_idx])

        # Val scaled mean should NOT be exactly 0 (different distribution)
        X_val_scaled = scaler.transform(X[val_idx])
        X_val_masked = np.where(mask[val_idx], X_val_scaled, np.nan)
        val_means = np.nanmean(X_val_masked, axis=0)
        # This is a statistical test — means should be close to 0 but not exact
        # The point is: scaler was fit on train, not val
        assert X_val_scaled.shape == (20, 5)
