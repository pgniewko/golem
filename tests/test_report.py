"""Focused report tests."""

import csv

from golem.pretrain import METRICS_FIELDNAMES
from golem.report import generate_report


def test_generate_report_handles_legacy_metrics_csv(tmp_path):
    (tmp_path / "metrics.csv").write_text(
        "epoch,train_loss,val_loss,val_rmse,learning_rate,elapsed_seconds\n"
        "0,1.000000,2.000000,1.414214,1.00e-04,3.0\n"
    )
    (tmp_path / "resolved_config.yaml").write_text(
        "max_epochs: 1\nmodel:\n  hidden_dim: 64\n  num_gt_layers: 2\n  num_heads: 4\n"
    )

    html = generate_report(tmp_path).read_text()

    assert "ECFP-Latent Alignment" not in html


def test_generate_report_adds_alignment_chart_when_metrics_exist(tmp_path):
    with open(tmp_path / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_FIELDNAMES)
        writer.writeheader()
        writer.writerow(
            {
                "epoch": 0,
                "train_loss": "1.000000",
                "val_loss": "2.000000",
                "val_rmse": "1.414214",
                "learning_rate": "1.00e-04",
                "elapsed_seconds": "3.0",
                "train_alignment_loss": "0.050000",
                "val_alignment_loss": "0.060000",
                "val_alignment_spearman": "0.700000",
                "val_alignment_kendall": "0.500000",
            }
        )
    (tmp_path / "resolved_config.yaml").write_text(
        "max_epochs: 1\nmodel:\n  hidden_dim: 64\n  num_gt_layers: 2\n  num_heads: 4\n"
    )

    html = generate_report(tmp_path).read_text()

    assert "ECFP-Latent Alignment" in html
    assert "Val Spearman" in html
