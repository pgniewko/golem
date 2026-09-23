"""Test NaNAwareStandardScaler functionalities"""

import numpy as np

from golem.descriptors import NaNAwareStandardScaler

def test_state_dict_roundtrip():
    rng = np.random.RandomState(0)
    X =rng.randn(20, 4).astype(np.float32)
    validity = np.ones((20,4), dtype=bool)

    validity[0, 1] = False

    names = ["a", "b", "c", "d"]

    scaler = NaNAwareStandardScaler(winsorize_range=(-6.0, 6.0))
    scaler.fit(X, validity, names=names)

    restored = NaNAwareStandardScaler.from_state_dict(scaler.state_dict())

    assert restored.names == scaler.names
    assert restored.versions == scaler.versions

    np.testing.assert_array_equal(restored.keep_mask, scaler.keep_mask)
    np.testing.assert_allclose(restored.mean_, scaler.mean_)
    np.testing.assert_allclose(restored.std_, scaler.std_)
    np.testing.assert_allclose(restored.transform(X), scaler.transform(X))


