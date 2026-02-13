"""Click CLI for Golem.  Commands: ``golem pretrain``, ``golem report``."""

from __future__ import annotations

import click

from golem.config import load_config
from golem.pretrain import pretrain
from golem.report import generate_report


@click.group()
@click.version_option(package_name="golem")
def main() -> None:
    """Golem — MDAE pretraining for Graph Transformers on molecular descriptors."""


@main.command()
@click.option(
    "--smiles", required=True, type=click.Path(exists=True),
    help="Path to SMILES file (.smi or .csv with SMILES column).",
)
@click.option(
    "--config", "config_path", default=None, type=click.Path(exists=True),
    help="Path to YAML config file (optional — defaults are sufficient).",
)
@click.option(
    "--output", required=True, type=click.Path(),
    help="Output directory for checkpoints, logs, and metrics.",
)
@click.option("--max-epochs", default=None, type=int, help="Override max training epochs.")
@click.option("--batch-size", default=None, type=int, help="Override batch size.")
@click.option("--lr", default=None, type=float, help="Override learning rate.")
@click.option(
    "--subsample", default=None, type=float,
    help="Subsample fraction of SMILES (e.g. 0.1 for 10%%).",
)
@click.option("--seed", default=None, type=int, help="Override random seed.")
@click.option(
    "--no-isoforms", is_flag=True, default=False,
    help="Disable isoform enumeration.",
)
@click.option(
    "--verbose", is_flag=True, default=False,
    help="Show DEBUG-level logs on console.",
)
def pretrain_cmd(
    smiles: str,
    config_path: str | None,
    output: str,
    max_epochs: int | None,
    batch_size: int | None,
    lr: float | None,
    subsample: float | None,
    seed: int | None,
    no_isoforms: bool,
    verbose: bool,
) -> None:
    """Run MDAE pretraining on molecular descriptors."""
    # Build CLI overrides dict (None values are ignored by load_config)
    overrides: dict = {}
    if max_epochs is not None:
        overrides["max_epochs"] = max_epochs
    if batch_size is not None:
        overrides["batch_size"] = batch_size
    if lr is not None:
        overrides["lr"] = lr
    if seed is not None:
        overrides["seed"] = seed
    if no_isoforms:
        overrides["isoforms"] = {"enabled": False}

    cfg = load_config(yaml_path=config_path, **overrides)

    pretrain(
        smiles_path=smiles,
        config=cfg,
        output_dir=output,
        subsample=subsample,
        verbose=verbose,
    )


@main.command()
@click.argument("output_dir", type=click.Path(exists=True))
@click.option(
    "--output", "html_path", default=None, type=click.Path(),
    help="Path for the HTML report (default: <output_dir>/pretrain_report.html).",
)
def report(output_dir: str, html_path: str | None) -> None:
    """Generate an HTML report from an existing experiment directory."""
    path = generate_report(output_dir, html_path=html_path)
    click.echo(f"Report written to {path}")
