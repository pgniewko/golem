"""Test DescriptorTransformer functionalities"""

import numpy as np

import pytest

from golem.descriptors import DescriptorTransformer


_KINDS = ["none", "log", "quantile", "none", "none", "none"]


def test_state_dict_roundtrip():
    rng = np.random.RandomState(0)
    X = rng.randn(20, 4).astype(np.float32)
    validity = np.ones((20, 4), dtype=bool)

    validity[0, 1] = False

    names = ["a", "b", "c", "d"]

    scaler = DescriptorTransformer(winsorize_range=(-6.0, 6.0))
    scaler.fit(X, validity, names=names)

    restored = DescriptorTransformer.from_state_dict(scaler.state_dict())

    assert restored.names == scaler.names
    assert restored.versions == scaler.versions

    np.testing.assert_array_equal(restored.keep_mask, scaler.keep_mask)
    np.testing.assert_allclose(restored.mean_, scaler.mean_)
    np.testing.assert_allclose(restored.std_, scaler.std_)
    np.testing.assert_allclose(restored.transform(X), scaler.transform(X))


@pytest.fixture
def heavy_tailed_data():
    rng = np.random.default_rng(0)
    n = 2000
    X = np.column_stack([
        rng.normal(5, 2, n),                                # gaussian -> none
        rng.lognormal(0, 2, n),                             # heavy tail -> log
        np.r_[rng.normal(0, 1, n - 20), np.full(20, 1e6)],  # extreme outliers -> quantile
        rng.poisson(0.3, n).astype(float),                  # sparse counts -> none
        np.full(n, 3.0),                                    # constant -> none
        rng.normal(0, 1, n),                                # will be few valid -> none
    ])

    validity = np.ones_like(X, dtype=bool)
    validity[20:, 5] = False
    X[~validity] = 0.0
    names = [f"c{i}" for i in range(X.shape[1])]
    scaler = DescriptorTransformer(winsorize_range=(-6, 6)).fit(X, validity, names=names)

    return X, validity, scaler


def test_transform_kinds(heavy_tailed_data):
    _, _, scaler = heavy_tailed_data
    assert scaler.kinds == _KINDS


def test_transform_output_standardized(heavy_tailed_data):
    X, validity, scaler = heavy_tailed_data
    Z = scaler.transform(X)
    assert Z.dtype == np.float64
    assert np.isfinite(Z).all()

    for j in (0, 1, 2, 3, 5):
        z = Z[validity[:, j], j]
        assert abs(z.mean()) < 1e-6
        assert abs(z.std() - 1.0) < 1e-3


def test_inverse_transform(heavy_tailed_data):
    X, validity, scaler = heavy_tailed_data
    Z = scaler.transform(X)
    X_back = scaler.inverse_transform(Z)
    in_range = validity & (np.abs(Z) < 5.99)
    for j, kind in enumerate(scaler.kinds):
        rtol = 1e-2 if kind == "quantile" else 1e-5
        np.testing.assert_allclose(X_back[in_range[:, j], j], X[in_range[:, j], j], rtol=rtol, atol=1e-5)


def test_state_dict_roundtrip_with_kinds(heavy_tailed_data):
    X, _, scaler = heavy_tailed_data
    restored_scaler = DescriptorTransformer.from_state_dict(scaler.state_dict())

    assert restored_scaler.kinds == scaler.kinds
    assert restored_scaler.params == scaler.params

    np.testing.assert_array_equal(restored_scaler.transform(X), scaler.transform(X))
