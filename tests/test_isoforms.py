"""Tests for golem.isoforms."""

import pytest
from rdkit import Chem

from golem.config import IsoformConfig
from golem.isoforms import (
    _desalt,
    _enumerate_protonation,
    _enumerate_tautomers,
    _is_valid_protomer,
    _neutralize,
    enumerate_isoforms,
    enumerate_isoforms_batch,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# TestTautomerEnumeration
# ---------------------------------------------------------------------------

class TestTautomerEnumeration:
    """Tests for tautomer enumeration."""

    def test_guanine_produces_tautomers(self):
        """Guanine should produce >=2 tautomers."""
        smi = "O=c1[nH]c2[nH]cnc2c(=O)[nH]1"
        mol = Chem.MolFromSmiles(smi)
        tautomers = _enumerate_tautomers(mol, max_tautomers=10)
        assert len(tautomers) >= 2, f"Expected >=2 tautomers for guanine, got {len(tautomers)}"

    def test_benzene_no_extra_tautomers(self):
        """Benzene should not produce extra tautomers beyond the original."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        tautomers = _enumerate_tautomers(mol, max_tautomers=10)
        # Should produce at most 1 (the canonical tautomer itself)
        smiles_set = {Chem.MolToSmiles(t, canonical=True) for t in tautomers}
        assert len(smiles_set) == 1, f"Benzene should have 1 unique tautomer, got {len(smiles_set)}"

    def test_max_tautomers_respected(self):
        """max_tautomers limit should be respected."""
        # Guanine has many tautomers
        mol = Chem.MolFromSmiles("O=c1[nH]c2[nH]cnc2c(=O)[nH]1")
        tautomers = _enumerate_tautomers(mol, max_tautomers=3)
        assert len(tautomers) <= 3


# ---------------------------------------------------------------------------
# TestProtonationEnumeration
# ---------------------------------------------------------------------------

class TestProtonationEnumeration:
    """Tests for protonation state enumeration."""

    def test_acetic_acid_deprotonated(self):
        """Acetic acid at pH ~7 should produce deprotonated form [O-]."""
        protomers = _enumerate_protonation("CC(=O)O", ph_range=(6.4, 8.4), max_protomers=10)
        protomer_smiles = {Chem.MolToSmiles(m, canonical=True) for m in protomers}
        # Should contain deprotonated carboxylate
        has_deprotonated = any("[O-]" in s for s in protomer_smiles)
        assert has_deprotonated, f"Expected deprotonated form in {protomer_smiles}"

    def test_amine_protonated(self):
        """Amine at pH ~7 should produce protonated form [NH3+]."""
        protomers = _enumerate_protonation("CCN", ph_range=(6.4, 8.4), max_protomers=10)
        protomer_smiles = {Chem.MolToSmiles(m, canonical=True) for m in protomers}
        # Should contain protonated amine
        has_protonated = any("[NH3+]" in s or "NH3+" in s for s in protomer_smiles)
        assert has_protonated, f"Expected protonated amine in {protomer_smiles}"

    def test_tertiary_amide_not_protonated(self):
        """Tertiary amide N should NOT be protonated."""
        smi = "CC(=O)N1CCCCC1"  # N-acetylpiperidine (tertiary amide)
        protomers = _enumerate_protonation(smi, ph_range=(6.4, 8.4), max_protomers=10)
        for pmol in protomers:
            psmi = Chem.MolToSmiles(pmol, canonical=True)
            # The amide nitrogen should not gain a hydrogen
            # Check: no atom should be N with formal charge +1 bonded to C=O
            for atom in pmol.GetAtoms():
                if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() > 0:
                    # Check if this N is bonded to C=O (amide)
                    for neighbor in atom.GetNeighbors():
                        if neighbor.GetAtomicNum() == 6:
                            for bond in neighbor.GetBonds():
                                other = bond.GetOtherAtom(neighbor)
                                if (
                                    other.GetAtomicNum() == 8
                                    and bond.GetBondTypeAsDouble() == 2.0
                                ):
                                    pytest.fail(
                                        f"Tertiary amide N protonated in {psmi}"
                                    )

    def test_nitrogen_4h_rejected(self):
        """No protomer should have nitrogen with >= 4 hydrogens."""
        # Test with simple amine that protonation tool might over-protonate
        protomers = _enumerate_protonation("CCN", ph_range=(6.4, 8.4), max_protomers=10)
        for pmol in protomers:
            for atom in pmol.GetAtoms():
                if atom.GetAtomicNum() == 7:
                    assert atom.GetTotalNumHs() < 4, (
                        f"N with {atom.GetTotalNumHs()} H in "
                        f"{Chem.MolToSmiles(pmol, canonical=True)}"
                    )

    def test_indole_no_extra_protonation(self):
        """Indole NH at physiological pH should remain as-is."""
        smi = "c1ccc2[nH]ccc2c1"  # indole
        protomers = _enumerate_protonation(smi, ph_range=(6.4, 8.4), max_protomers=10)
        protomer_smiles = {Chem.MolToSmiles(m, canonical=True) for m in protomers}
        # Should mostly contain the original — no deprotonation of pyrrole NH
        canonical_original = Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True)
        assert canonical_original in protomer_smiles, (
            f"Expected original indole in protomers: {protomer_smiles}"
        )

    def test_max_protomers_respected(self):
        """max_protomers limit should be respected."""
        protomers = _enumerate_protonation("CCN", ph_range=(6.4, 8.4), max_protomers=2)
        assert len(protomers) <= 2

    def test_all_protomers_are_valid_mols(self):
        """All returned protomers should be valid RDKit molecules."""
        protomers = _enumerate_protonation("CC(=O)O", ph_range=(6.4, 8.4), max_protomers=10)
        for pmol in protomers:
            assert pmol is not None
            smi = Chem.MolToSmiles(pmol, canonical=True)
            assert smi is not None
            # Should pass sanitization
            roundtrip = Chem.MolFromSmiles(smi)
            assert roundtrip is not None


# ---------------------------------------------------------------------------
# TestValidProtomer
# ---------------------------------------------------------------------------

class TestValidProtomer:
    """Tests for _is_valid_protomer validator."""

    def test_valid_protomer_accepted(self):
        """A normal molecule should pass validation."""
        mol = Chem.MolFromSmiles("CC(=O)[O-]")
        assert _is_valid_protomer(mol) is True

    def test_nh4_rejected(self):
        """Nitrogen with 4 hydrogens should be rejected."""
        # Manually create NH4+ scenario: ammonium
        mol = Chem.MolFromSmiles("[NH4+]")
        if mol is not None:
            assert _is_valid_protomer(mol) is False

    def test_protonated_tertiary_amide_is_rejected_even_after_atom_reordering(self):
        """False tertiary-amide protonation should not depend on atom order."""
        protomer = Chem.MolFromSmiles("[NH+]1(CCCCC1)C(C)=O")

        assert protomer is not None
        assert _is_valid_protomer(protomer) is False


# ---------------------------------------------------------------------------
# TestNeutralization
# ---------------------------------------------------------------------------

class TestNeutralization:
    """Tests for neutralization."""

    def test_charged_molecule_neutralized(self):
        """A charged molecule should be neutralized."""
        mol = Chem.MolFromSmiles("CC(=O)[O-]")
        neutralized = _neutralize(mol)
        assert len(neutralized) >= 1
        neutral_smi = Chem.MolToSmiles(neutralized[0], canonical=True)
        # Should be neutral form
        assert "[O-]" not in neutral_smi, f"Expected neutral form, got {neutral_smi}"

    def test_neutral_molecule_unchanged(self):
        """Already neutral molecule should remain unchanged."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        neutralized = _neutralize(mol)
        assert len(neutralized) >= 1
        smi = Chem.MolToSmiles(neutralized[0], canonical=True)
        assert smi == "c1ccccc1"


# ---------------------------------------------------------------------------
# TestDesalting
# ---------------------------------------------------------------------------

class TestDesalting:
    """Tests for desalting (salt fragment removal)."""

    def test_sodium_salt_desalted(self):
        """Sodium acetate salt should be desalted to acetic acid fragment."""
        mol = Chem.MolFromSmiles("CC(=O)[O-].[Na+]")
        result = _desalt(mol)
        assert result is not None
        smi = Chem.MolToSmiles(result, canonical=True)
        assert "Na" not in smi

    def test_hcl_salt_desalted(self):
        """HCl salt of amine should be desalted to the amine."""
        mol = Chem.MolFromSmiles("CC[NH3+].[Cl-]")
        result = _desalt(mol)
        assert result is not None
        smi = Chem.MolToSmiles(result, canonical=True)
        assert smi == "CC[NH3+]", f"Expected 'CC[NH3+]', got '{smi}'"

    def test_non_salt_unchanged(self):
        """Non-salt molecule should be returned unchanged."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        result = _desalt(mol)
        assert result is not None
        smi = Chem.MolToSmiles(result, canonical=True)
        assert smi == "c1ccccc1"

    def test_desalting_in_pipeline(self):
        """Desalting should add the desalted form as an isoform."""
        config = IsoformConfig(
            desalting=True,
            tautomers=False,
            protonation=False,
            neutralization=False,
        )
        isoforms = enumerate_isoforms("CC(=O)[O-].[Na+]", config)
        assert len(isoforms) >= 2
        assert isoforms[0] == "CC(=O)[O-].[Na+]"
        assert "CC(=O)[O-]" in isoforms, f"Expected desalted form 'CC(=O)[O-]' in {isoforms}"


# ---------------------------------------------------------------------------
# TestSMILESValidation
# ---------------------------------------------------------------------------

class TestSMILESValidation:
    """Tests for SMILES validation in the pipeline."""

    def test_all_isoforms_are_valid_smiles(self, default_config):
        """All output SMILES from enumerate_isoforms should pass RDKit parsing + sanitization."""
        test_smiles = ["CC(=O)O", "CCN", "c1ccccc1", "CC(=O)N1CCCCC1", "O=c1[nH]c2[nH]cnc2c(=O)[nH]1"]
        for parent in test_smiles:
            isoforms = enumerate_isoforms(parent, default_config)
            for iso in isoforms:
                mol = Chem.MolFromSmiles(iso)
                assert mol is not None, f"Invalid SMILES from enumeration: {iso} (parent: {parent})"
                try:
                    Chem.SanitizeMol(mol)
                except Exception as e:
                    pytest.fail(f"Sanitization failed for {iso} (parent: {parent}): {e}")

    def test_invalid_smiles_filtered(self):
        """Invalid SMILES should not appear in enumerate_isoforms output (beyond the original)."""
        config = IsoformConfig()
        # Invalid SMILES returns original string
        isoforms = enumerate_isoforms("NOT_A_SMILES", config)
        assert isoforms == ["NOT_A_SMILES"]


# ---------------------------------------------------------------------------
# TestEnumerateIsoforms (original + expanded)
# ---------------------------------------------------------------------------

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
        """Guanine should produce tautomers."""
        config = IsoformConfig(
            tautomers=True,
            protonation=False,
            neutralization=False,
        )
        smi = "O=c1[nH]c2[nH]cnc2c(=O)[nH]1"
        isoforms = enumerate_isoforms(smi, config)
        assert len(isoforms) >= 2, f"Expected tautomers, got {len(isoforms)}"

    def test_protonation_expands(self):
        """Acetic acid should produce at least the original + deprotonated form."""
        config = IsoformConfig(
            tautomers=False,
            protonation=True,
            neutralization=False,
        )
        smi = "CC(=O)O"
        isoforms = enumerate_isoforms(smi, config)
        assert len(isoforms) >= 2, f"Expected protonation states, got {isoforms}"


# ---------------------------------------------------------------------------
# TestBatchEnumeration
# ---------------------------------------------------------------------------

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

    def test_batch_mixed_valid_invalid(self, default_config):
        """Batch with mix of valid and invalid SMILES should handle gracefully."""
        smiles_list = ["c1ccccc1", "INVALID", "CCO"]
        results = enumerate_isoforms_batch(smiles_list, default_config)
        assert len(results) == 3
        assert results["INVALID"] == ["INVALID"]

    def test_batch_deduplication_per_parent(self, default_config):
        """Each parent's isoforms should be deduplicated."""
        smiles_list = ["CC(=O)O", "CCN"]
        results = enumerate_isoforms_batch(smiles_list, default_config)
        for smi, isoforms in results.items():
            assert len(isoforms) == len(set(isoforms)), (
                f"Duplicate isoforms for {smi}: {isoforms}"
            )
