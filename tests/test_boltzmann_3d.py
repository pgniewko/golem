from __future__ import annotations

import numpy as np
import pytest
import torch

from golem.config import ConformerConfig, PretrainConfig, load_config, validate_pretrain_config
from golem.conformers import (
    OptimizedConformer,
    OptimizedConformerPool,
    retain_boltzmann_conformers,
)
from golem.descriptors import (
    Conformer3DPool,
    compute_boltzmann_weighted_3d_statistics,
    materialize_boltzmann_mean_targets,
)
from golem.pretrain import _BoltzmannTrainingDataset


class _DummySample:
    def __init__(self, y: torch.Tensor, y_mask: torch.Tensor) -> None:
        self.y = y
        self.y_mask = y_mask

    def clone(self) -> "_DummySample":
        return _DummySample(self.y.clone(), self.y_mask.clone())


def _make_pool(
    values: list[list[float]],
    validity_mask: list[list[bool]],
    weights: list[float],
) -> Conformer3DPool:
    return Conformer3DPool(
        values=np.asarray(values, dtype=np.float32),
        validity_mask=np.asarray(validity_mask, dtype=np.bool_),
        boltzmann_weights=np.asarray(weights, dtype=np.float32),
        energies=np.zeros(len(weights), dtype=np.float32),
        delta_energies=np.zeros(len(weights), dtype=np.float32),
    )


def test_pretrain_config_defaults_include_boltzmann_controls() -> None:
    config = PretrainConfig()

    assert config.descriptors.three_d_settings.target_mode == "lowest_energy"
    assert config.conformers.n_generate == 8
    assert config.conformers.n_keep_best == 3
    assert config.conformers.max_delta_energy_kcal == pytest.approx(3.0)


def test_validate_pretrain_config_allows_boltzmann_workers() -> None:
    config = PretrainConfig()
    config.descriptors.include_3d_targets = True
    config.descriptors.three_d_settings.target_mode = "boltzmann"
    config.num_workers = 1

    validated = validate_pretrain_config(config)

    assert validated.num_workers == 1


def test_load_config_rejects_legacy_conformer_n_keep(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "descriptors:\n"
        "  include_3d_targets: true\n"
        "conformers:\n"
        "  n_keep: 2\n"
    )

    with pytest.raises(ValueError, match="conformers.n_keep"):
        load_config(str(config_path))


def test_retain_boltzmann_conformers_filters_by_delta_energy_and_best_n() -> None:
    pool = OptimizedConformerPool(
        mol=None,  # type: ignore[arg-type]
        conformers=[
            OptimizedConformer(conformer_id=0, energy=1.0, delta_energy=0.0),
            OptimizedConformer(conformer_id=1, energy=1.5, delta_energy=0.5),
            OptimizedConformer(conformer_id=2, energy=3.9, delta_energy=2.9),
            OptimizedConformer(conformer_id=3, energy=4.2, delta_energy=3.2),
            OptimizedConformer(conformer_id=4, energy=5.0, delta_energy=4.0),
        ],
    )
    config = ConformerConfig(n_generate=8, n_keep_best=3, max_delta_energy_kcal=3.0)

    retained = retain_boltzmann_conformers(pool, config)

    assert [conformer.conformer_id for conformer in retained.conformers] == [0, 1, 2]


def test_materialize_boltzmann_mean_targets_renormalizes_over_valid_conformers() -> None:
    pool = _make_pool(
        values=[[1.0, 10.0], [3.0, 999.0]],
        validity_mask=[[True, True], [True, False]],
        weights=[0.75, 0.25],
    )

    values, validity_mask = materialize_boltzmann_mean_targets([pool])

    assert values.shape == (1, 2)
    assert validity_mask.tolist() == [[True, True]]
    assert values[0, 0] == pytest.approx(1.5)
    assert values[0, 1] == pytest.approx(10.0)


def test_compute_boltzmann_weighted_3d_statistics_uses_pool_weights_and_masks() -> None:
    pools = [
        _make_pool(
            values=[[1.0, 10.0], [3.0, 0.0]],
            validity_mask=[[True, True], [True, False]],
            weights=[0.75, 0.25],
        ),
        _make_pool(
            values=[[5.0, 20.0]],
            validity_mask=[[True, True]],
            weights=[1.0],
        ),
    ]

    mean, std = compute_boltzmann_weighted_3d_statistics(pools)

    expected_mean_0 = (0.75 * 1.0 + 0.25 * 3.0 + 1.0 * 5.0) / 2.0
    expected_mean_1 = (0.75 * 10.0 + 1.0 * 20.0) / 1.75
    expected_var_0 = (
        0.75 * (1.0 - expected_mean_0) ** 2
        + 0.25 * (3.0 - expected_mean_0) ** 2
        + 1.0 * (5.0 - expected_mean_0) ** 2
    ) / 2.0
    expected_var_1 = (
        0.75 * (10.0 - expected_mean_1) ** 2
        + 1.0 * (20.0 - expected_mean_1) ** 2
    ) / 1.75

    assert mean[0] == pytest.approx(expected_mean_0)
    assert mean[1] == pytest.approx(expected_mean_1)
    assert std[0] == pytest.approx(np.sqrt(expected_var_0))
    assert std[1] == pytest.approx(np.sqrt(expected_var_1))


def test_boltzmann_training_dataset_samples_one_cached_conformer_per_epoch() -> None:
    base_sample = _DummySample(
        y=torch.tensor([[11.0, 0.0, 0.0]], dtype=torch.float32),
        y_mask=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
    )
    pool = _make_pool(
        values=[[100.0, 200.0], [300.0, 400.0]],
        validity_mask=[[True, True], [True, False]],
        weights=[0.25, 0.75],
    )
    dataset = _BoltzmannTrainingDataset([base_sample], [pool], slice(1, 3), seed=7)

    observed_targets = []
    expected_targets = []
    for epoch in range(4):
        dataset.set_epoch(epoch)
        observed_targets.append(dataset[0].y[0, 1:].tolist())
        observed_targets.append(dataset[0].y[0, 1:].tolist())
        rng = np.random.default_rng(np.random.SeedSequence([7, epoch, 0]))
        expected_index = int(rng.choice(2, p=np.asarray([0.25, 0.75], dtype=np.float64)))
        expected_targets.append(pool.values[expected_index].tolist())
        expected_targets.append(pool.values[expected_index].tolist())

    assert observed_targets == expected_targets
    assert base_sample.y.tolist() == [[11.0, 0.0, 0.0]]
    assert base_sample.y_mask.tolist() == [[1.0, 0.0, 0.0]]


def test_boltzmann_training_dataset_renormalizes_float32_sampling_weights() -> None:
    base_sample = _DummySample(
        y=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        y_mask=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
    )
    pool = _make_pool(
        values=[[10.0], [20.0], [30.0]],
        validity_mask=[[True], [True], [True]],
        weights=[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    )
    dataset = _BoltzmannTrainingDataset([base_sample], [pool], slice(1, 2), seed=5)

    dataset.set_epoch(0)
    sample = dataset[0]

    probabilities = np.asarray(pool.boltzmann_weights, dtype=np.float64)
    probabilities /= probabilities.sum()
    rng = np.random.default_rng(np.random.SeedSequence([5, 0, 0]))
    expected_index = int(rng.choice(3, p=probabilities))

    assert sample.y.tolist() == [[0.0, pool.values[expected_index, 0]]]
    assert sample.y_mask.tolist() == [[0.0, 1.0]]
