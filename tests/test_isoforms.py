"""Tests for golem.isoforms."""

import pytest
from rdkit import Chem

from golem.config import IsoformConfig
from golem.isoforms import enumerate_isoforms, enumerate_isoforms_batch


@pytest.fixture
def default_config():
    return IsoformConfig()


@pytest.fixture
def no_isoform_config():
    return IsoformConfig(
        tautomers=False,
        protonation=False,
        neutralization=False,
    )


class TestEnumerateIsoforms:
    """Tests for single-SMILES isoform enumeration."""

    def test_original_always_first(self, default_config):
        """Original SMILES should always be at index 0."""
        smi = "c1ccccc1"
        isoforms = enumerate_isoforms(smi, default_config)
        assert len(isoforms) >= 1
        # index 0 should be the canonical form of the input
        assert isoforms[0] == Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True)

    def test_invalid_smiles_returns_original(self, default_config):
        """Invalid SMILES should return the original string."""
        smi = "NOT_A_SMILES"
        isoforms = enumerate_isoforms(smi, default_config)
        assert isoforms == [smi]

    def test_deduplication(self, default_config):
        """All returned isoforms should be unique."""
        smi = "c1ccccc1"
        isoforms = enumerate_isoforms(smi, default_config)
        assert len(isoforms) == len(set(isoforms))

    def test_max_isoforms_respected(self):
        """Should not exceed max_isoforms."""
        config = IsoformConfig(max_isoforms=3)
        smi = "c1ccc(O)cc1"  # phenol — can have tautomers
        isoforms = enumerate_isoforms(smi, config)
        assert len(isoforms) <= 3

    def test_no_enumeration(self, no_isoform_config):
        """With all enumeration disabled, should return only the original."""
        smi = "c1ccccc1"
        isoforms = enumerate_isoforms(smi, no_isoform_config)
        assert len(isoforms) == 1

    def test_canonical_smiles_returned(self, default_config):
        """All returned SMILES should be canonical."""
        smi = "C(=O)O"  # non-canonical acetic acid fragment
        isoforms = enumerate_isoforms(smi, default_config)
        for iso in isoforms:
            mol = Chem.MolFromSmiles(iso)
            if mol is not None:
                assert iso == Chem.MolToSmiles(mol, canonical=True)

    def test_tautomers_expand(self):
        """Warfarin-like molecules should produce tautomers."""
        config = IsoformConfig(
            tautomers=True,
            protonation=False,
            neutralization=False,
            max_isoforms=25,
        )
        # Guanine has well-known tautomers
        smi = "O=c1[nH]c2[nH]cnc2c(=O)[nH]1"
        isoforms = enumerate_isoforms(smi, config)
        assert len(isoforms) >= 2, f"Expected tautomers, got {len(isoforms)}"


class TestEnumerateIsoformsBatch:
    """Tests for batch isoform enumeration."""

    def test_batch_returns_all_parents(self, default_config):
        """All parent SMILES should appear as keys in the result."""
        smiles_list = ["c1ccccc1", "CC(=O)O", "CCO"]
        results = enumerate_isoforms_batch(smiles_list, default_config)
        assert set(results.keys()) == set(smiles_list)

    def test_batch_original_at_index_0(self, default_config):
        """For each parent, the first isoform should be the canonical form."""
        smiles_list = ["c1ccccc1", "CCO"]
        results = enumerate_isoforms_batch(smiles_list, default_config)
        for smi in smiles_list:
            can = Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True)
            assert results[smi][0] == can

    def test_batch_empty_list(self, default_config):
        """Empty input should return empty dict."""
        results = enumerate_isoforms_batch([], default_config)
        assert results == {}
