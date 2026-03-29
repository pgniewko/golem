"""CLI-specific regression tests."""

from click.testing import CliRunner

from golem.cli import main


def test_pretrain_cli_applies_num_workers_and_subsample_overrides(monkeypatch, tmp_path):
    smiles_path = tmp_path / "mols.smi"
    smiles_path.write_text("CCO\nCCN\nc1ccccc1\n")
    captured = {}

    def fake_pretrain(*, smiles_path, config, output_dir, verbose=False):
        captured["smiles_path"] = smiles_path
        captured["config"] = config
        captured["output_dir"] = output_dir
        captured["verbose"] = verbose

    monkeypatch.setattr("golem.cli.pretrain", fake_pretrain)

    result = CliRunner().invoke(
        main,
        [
            "pretrain",
            "--smiles", str(smiles_path),
            "--output", str(tmp_path / "out"),
            "--num-workers", "3",
            "--subsample", "0.25",
            "--no-isoforms",
            "--verbose",
        ],
    )

    assert result.exit_code == 0
    assert captured["smiles_path"] == str(smiles_path)
    assert captured["output_dir"] == str(tmp_path / "out")
    assert captured["verbose"] is True
    assert captured["config"].num_workers == 3
    assert captured["config"].subsample == 0.25
    assert captured["config"].isoforms.enabled is False
