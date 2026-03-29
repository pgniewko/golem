"""Tests for golem.descriptors."""

import numpy as np
import pytest

from golem.descriptors import (
    NaNAwareStandardScaler,
    compute_mordred_descriptors,
)


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
        values, mask, names = compute_mordred_descriptors(["c1ccccc1", "CCO", "CCCC"])
        all_invalid = ~mask.any(axis=0)
        assert not all_invalid.any(), "Found columns with all-invalid entries (should be dropped)"

class TestNaNAwareStandardScaler:
    """Tests for the NaN-aware scaler."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with known statistics."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5).astype(np.float32)
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
        assert not np.isnan(X_scaled).any()
        assert not np.isinf(X_scaled).any()

    def test_transform_before_fit_raises(self):
        """Calling transform before fit should raise."""
        scaler = NaNAwareStandardScaler()
        with pytest.raises(RuntimeError, match="not been fit"):
            scaler.transform(np.zeros((2, 3)))
