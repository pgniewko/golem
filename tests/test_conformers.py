"""Tests for golem.conformers."""

from types import SimpleNamespace

from golem.config import ConformerConfig
from golem.conformers import generate_conformer_ensemble


class _FakeConformer:
    def __init__(self, conf_id: int):
        self._conf_id = conf_id

    def GetId(self) -> int:
        return self._conf_id


class _FakeMol:
    def __init__(self, conformer_ids: list[int]):
        self._conformers = [_FakeConformer(conf_id) for conf_id in conformer_ids]

    def GetConformers(self):
        return self._conformers


def test_generate_conformer_ensemble_keeps_all_low_energy_conformers(monkeypatch):
    fake_mol = _FakeMol([0, 1, 2, 3])
    params = SimpleNamespace(randomSeed=None, pruneRmsThresh=None, timeout=None)

    monkeypatch.setattr("golem.conformers.Chem.MolFromSmiles", lambda smiles: object())
    monkeypatch.setattr("golem.conformers.Chem.AddHs", lambda mol: fake_mol)
    monkeypatch.setattr("golem.conformers.AllChem.ETKDGv3", lambda: params)
    monkeypatch.setattr(
        "golem.conformers.AllChem.EmbedMultipleConfs",
        lambda mol, numConfs, params: [0, 1, 2, 3],
    )
    monkeypatch.setattr(
        "golem.conformers._optimize_conformers",
        lambda mol, method: {2: 1.5, 0: 0.0, 1: 0.1, 3: 12.0} if method == "MMFF" else None,
    )
    monkeypatch.setattr(
        "golem.conformers.AllChem.GetConformerRMS",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("RMS should not be called")),
        raising=False,
    )

    ensemble, failure_reason = generate_conformer_ensemble(
        "CCO",
        ConformerConfig(
            energy_window_kcal=1.0,
            n_keep=1,
            prune_rms=0.0,
        ),
        seed=123,
    )

    assert failure_reason is None
    assert ensemble is not None
    assert ensemble.conformer_ids == [0, 1]
    assert ensemble.relative_energies_kcal.tolist() == [0.0, 0.1]


def test_generate_conformer_ensemble_returns_energy_filtered_when_window_excludes_all_but_none(monkeypatch):
    fake_mol = _FakeMol([0, 1])
    params = SimpleNamespace(randomSeed=None, pruneRmsThresh=None, timeout=None)

    monkeypatch.setattr("golem.conformers.Chem.MolFromSmiles", lambda smiles: object())
    monkeypatch.setattr("golem.conformers.Chem.AddHs", lambda mol: fake_mol)
    monkeypatch.setattr("golem.conformers.AllChem.ETKDGv3", lambda: params)
    monkeypatch.setattr(
        "golem.conformers.AllChem.EmbedMultipleConfs",
        lambda mol, numConfs, params: [0, 1],
    )
    monkeypatch.setattr(
        "golem.conformers._optimize_conformers",
        lambda mol, method: {0: 0.0, 1: 1.0} if method == "MMFF" else None,
    )

    ensemble, failure_reason = generate_conformer_ensemble(
        "CCO",
        ConformerConfig(energy_window_kcal=-1.0),
        seed=123,
    )

    assert ensemble is None
    assert failure_reason == "energy_filtered"
