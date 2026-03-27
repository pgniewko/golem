"""HTML report generator for Golem pretraining experiments.

Reads ``metrics.csv`` and ``resolved_config.yaml`` from an experiment output
directory and produces a self-contained Chart.js dashboard with:

- Training & validation objective curves
- Validation RMSE curve
- Learning-rate schedule
- Summary cards (best epoch, best val objective, elapsed time, sample counts)
- Epoch-by-epoch table with the best row highlighted

Public API::

    generate_report(output_dir, html_path=None) -> Path
"""

from __future__ import annotations

import csv
import logging
import math
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

OPTIONAL_ALIGNMENT_FIELDS = [
    "train_alignment_loss",
    "val_alignment_loss",
    "val_alignment_spearman",
    "val_alignment_kendall",
]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_metrics(metrics_path: Path) -> List[Dict[str, Any]]:
    """Load metrics.csv into a list of dicts with typed values."""
    rows: List[Dict[str, Any]] = []
    with open(metrics_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {
                "epoch": int(row["epoch"]),
                "train_loss": float(row["train_loss"]),
                "val_loss": float(row["val_loss"]),
                "train_descriptor_loss": float(row["train_descriptor_loss"]),
                "val_descriptor_loss": float(row["val_descriptor_loss"]),
                "val_rmse": float(row["val_rmse"]),
                "learning_rate": float(row["learning_rate"]),
                "elapsed_seconds": float(row["elapsed_seconds"]),
            }
            for field in OPTIONAL_ALIGNMENT_FIELDS:
                parsed[field] = _parse_optional_float(row, field)
            rows.append(parsed)
    return rows


def _parse_optional_float(row: Dict[str, str], key: str) -> float:
    raw = row.get(key, "")
    if raw in ("", None):
        return math.nan
    try:
        return float(raw)
    except (TypeError, ValueError):
        return math.nan


def _has_alignment_metrics(metrics: List[Dict[str, Any]]) -> bool:
    return any(
        math.isfinite(row.get(field, math.nan))
        for row in metrics
        for field in OPTIONAL_ALIGNMENT_FIELDS
    )


def _load_config(config_path: Path) -> Dict[str, Any]:
    """Load resolved_config.yaml."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def _fmt_elapsed(seconds: float) -> str:
    """Format elapsed seconds as a human-readable string."""
    td = timedelta(seconds=seconds)
    total_secs = int(td.total_seconds())
    hours, remainder = divmod(total_secs, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{seconds:.1f}s"


# ---------------------------------------------------------------------------
# Summary extraction
# ---------------------------------------------------------------------------

def _compute_summary(
    metrics: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Derive summary stats from metrics + config."""
    best_row = min(metrics, key=lambda r: r["val_loss"])
    last_row = metrics[-1]

    # Sample counts from split_ratios (approximation from config)
    split_ratios = config.get("split_ratios", [0.7, 0.2, 0.1])

    return {
        "best_epoch": best_row["epoch"],
        "best_val_objective": best_row["val_loss"],
        "best_val_rmse": best_row["val_rmse"],
        "total_epochs": len(metrics),
        "max_epochs": config.get("max_epochs", "?"),
        "elapsed": _fmt_elapsed(last_row["elapsed_seconds"]),
        "elapsed_seconds": last_row["elapsed_seconds"],
        "final_train_objective": last_row["train_loss"],
        "final_val_objective": last_row["val_loss"],
        "final_val_rmse": last_row["val_rmse"],
        "split_ratios": split_ratios,
        "batch_size": config.get("batch_size", "?"),
        "lr": config.get("lr", "?"),
        "patience": config.get("patience", "?"),
        "masking_ratio": config.get("masking_ratio", "?"),
        "seed": config.get("seed", "?"),
        "hidden_dim": config.get("model", {}).get("hidden_dim", "?"),
        "num_gt_layers": config.get("model", {}).get("num_gt_layers", "?"),
        "num_heads": config.get("model", {}).get("num_heads", "?"),
    }


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Golem Pretrain Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  :root {
    --bg: #0f172a; --surface: #1e293b; --border: #334155;
    --text: #e2e8f0; --muted: #94a3b8; --accent: #38bdf8;
    --green: #4ade80; --red: #f87171; --yellow: #facc15; --purple: #a78bfa;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    background: var(--bg); color: var(--text); padding: 2rem;
    line-height: 1.6;
  }
  h1 { font-size: 1.8rem; margin-bottom: 0.25rem; }
  .subtitle { color: var(--muted); margin-bottom: 2rem; font-size: 0.95rem; }
  /* Cards */
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; }
  .card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.25rem;
  }
  .card .label { font-size: 0.8rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
  .card .value { font-size: 1.6rem; font-weight: 700; margin-top: 0.25rem; }
  .card .detail { font-size: 0.8rem; color: var(--muted); margin-top: 0.25rem; }
  .best  .value { color: var(--green); }
  .time  .value { color: var(--yellow); }
  .info  .value { color: var(--accent); }
  /* Charts */
  .charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }
  .chart-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.5rem;
  }
  .chart-card h3 { font-size: 1rem; margin-bottom: 1rem; color: var(--muted); }
  /* Config */
  .config-section {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.5rem; margin-bottom: 2rem;
  }
  .config-section h3 { font-size: 1rem; margin-bottom: 1rem; color: var(--muted); }
  .config-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 0.5rem; }
  .config-item { display: flex; justify-content: space-between; padding: 0.4rem 0; border-bottom: 1px solid var(--border); }
  .config-key { color: var(--muted); font-size: 0.85rem; }
  .config-val { font-family: 'Fira Code', monospace; font-size: 0.85rem; }
  /* Table */
  .table-wrap {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.5rem; overflow-x: auto;
  }
  .table-wrap h3 { font-size: 1rem; margin-bottom: 1rem; color: var(--muted); }
  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  th { text-align: left; padding: 0.6rem 1rem; border-bottom: 2px solid var(--border); color: var(--muted); font-weight: 600; }
  td { padding: 0.5rem 1rem; border-bottom: 1px solid var(--border); }
  tr.best-row { background: rgba(74, 222, 128, 0.08); }
  tr.best-row td { color: var(--green); font-weight: 600; }
  footer { text-align: center; color: var(--muted); font-size: 0.75rem; margin-top: 3rem; }
</style>
</head>
<body>

<h1>Golem Pretrain Report</h1>
<p class="subtitle">Experiment directory: <code>{{OUTPUT_DIR}}</code></p>

<!-- Summary Cards -->
<div class="cards">
  <div class="card best">
    <div class="label">Best Val Objective</div>
    <div class="value">{{BEST_VAL_OBJECTIVE}}</div>
    <div class="detail">Epoch {{BEST_EPOCH}} / {{TOTAL_EPOCHS}}</div>
  </div>
  <div class="card best">
    <div class="label">Best Val RMSE</div>
    <div class="value">{{BEST_VAL_RMSE}}</div>
    <div class="detail">At best epoch</div>
  </div>
  <div class="card time">
    <div class="label">Elapsed Time</div>
    <div class="value">{{ELAPSED}}</div>
    <div class="detail">{{TOTAL_EPOCHS}} / {{MAX_EPOCHS}} epochs</div>
  </div>
  <div class="card info">
    <div class="label">Architecture</div>
    <div class="value">GT-{{NUM_GT_LAYERS}}L</div>
    <div class="detail">{{HIDDEN_DIM}}d / {{NUM_HEADS}}h</div>
  </div>
</div>

<!-- Charts -->
<div class="charts">
  <div class="chart-card">
    <h3>Training &amp; Validation Objective</h3>
    <canvas id="lossChart"></canvas>
  </div>
  <div class="chart-card">
    <h3>Descriptor Loss</h3>
    <canvas id="descriptorChart"></canvas>
  </div>
  <div class="chart-card">
    <h3>Validation RMSE</h3>
    <canvas id="rmseChart"></canvas>
  </div>
  <div class="chart-card">
    <h3>Learning Rate Schedule</h3>
    <canvas id="lrChart"></canvas>
  </div>
  <div class="chart-card">
    <h3>Train vs Val Objective Gap</h3>
    <canvas id="gapChart"></canvas>
  </div>
  {{ALIGNMENT_CHART}}
</div>

<!-- Config -->
<div class="config-section">
  <h3>Configuration</h3>
  <div class="config-grid">
    {{CONFIG_ITEMS}}
  </div>
</div>

<!-- Epoch Table -->
<div class="table-wrap">
  <h3>Epoch Details</h3>
  <table>
    <thead>
      <tr>
        <th>Epoch</th><th>Train Obj</th><th>Val Obj</th>
        <th>Train Desc</th><th>Val Desc</th><th>Val RMSE</th>
        <th>LR</th><th>Elapsed</th>
      </tr>
    </thead>
    <tbody>
      {{TABLE_ROWS}}
    </tbody>
  </table>
</div>

<footer>Generated by Golem</footer>

<script>
const epochs = {{EPOCHS_JSON}};
const trainLoss = {{TRAIN_LOSS_JSON}};
const valLoss = {{VAL_LOSS_JSON}};
const trainDescriptorLoss = {{TRAIN_DESCRIPTOR_LOSS_JSON}};
const valDescriptorLoss = {{VAL_DESCRIPTOR_LOSS_JSON}};
const valRmse = {{VAL_RMSE_JSON}};
const lr = {{LR_JSON}};
const gap = trainLoss.map((t, i) => t - valLoss[i]);
{{ALIGNMENT_JSON}}

const gridColor = 'rgba(148,163,184,0.1)';
const tickColor = '#94a3b8';

function makeOpts(yLabel) {
  return {
    responsive: true,
    interaction: { mode: 'index', intersect: false },
    plugins: { legend: { labels: { color: tickColor, usePointStyle: true, pointStyle: 'circle' } } },
    scales: {
      x: { title: { display: true, text: 'Epoch', color: tickColor }, ticks: { color: tickColor }, grid: { color: gridColor } },
      y: { title: { display: true, text: yLabel, color: tickColor }, ticks: { color: tickColor }, grid: { color: gridColor } },
    },
  };
}

new Chart(document.getElementById('lossChart'), {
  type: 'line',
  data: {
    labels: epochs,
    datasets: [
      { label: 'Train Objective', data: trainLoss, borderColor: '#38bdf8', backgroundColor: 'rgba(56,189,248,0.1)', tension: 0.3, pointRadius: 2 },
      { label: 'Val Objective',   data: valLoss,   borderColor: '#4ade80', backgroundColor: 'rgba(74,222,128,0.1)', tension: 0.3, pointRadius: 2 },
    ],
  },
  options: makeOpts('Weighted Objective'),
});

new Chart(document.getElementById('descriptorChart'), {
  type: 'line',
  data: {
    labels: epochs,
    datasets: [
      { label: 'Train Descriptor Loss', data: trainDescriptorLoss, borderColor: '#f97316', backgroundColor: 'rgba(249,115,22,0.1)', tension: 0.3, pointRadius: 2 },
      { label: 'Val Descriptor Loss',   data: valDescriptorLoss,   borderColor: '#fb7185', backgroundColor: 'rgba(251,113,133,0.1)', tension: 0.3, pointRadius: 2 },
    ],
  },
  options: makeOpts('Descriptor MSE'),
});

new Chart(document.getElementById('rmseChart'), {
  type: 'line',
  data: {
    labels: epochs,
    datasets: [
      { label: 'Val RMSE', data: valRmse, borderColor: '#a78bfa', backgroundColor: 'rgba(167,139,250,0.1)', tension: 0.3, pointRadius: 2 },
    ],
  },
  options: makeOpts('RMSE'),
});

new Chart(document.getElementById('lrChart'), {
  type: 'line',
  data: {
    labels: epochs,
    datasets: [
      { label: 'Learning Rate', data: lr, borderColor: '#facc15', backgroundColor: 'rgba(250,204,21,0.1)', tension: 0.3, pointRadius: 2 },
    ],
  },
  options: makeOpts('Learning Rate'),
});

new Chart(document.getElementById('gapChart'), {
  type: 'bar',
  data: {
    labels: epochs,
    datasets: [
      { label: 'Train - Val Objective', data: gap, backgroundColor: gap.map(v => v > 0 ? 'rgba(248,113,113,0.6)' : 'rgba(74,222,128,0.6)') },
    ],
  },
  options: makeOpts('Objective Gap'),
});
{{ALIGNMENT_SCRIPT}}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(
    output_dir: str | Path,
    html_path: Optional[str | Path] = None,
) -> Path:
    """Generate an HTML pretrain report from an experiment directory.

    Args:
        output_dir: Experiment directory containing ``metrics.csv`` and
            ``resolved_config.yaml``.
        html_path: Where to write the HTML file.  Defaults to
            ``<output_dir>/pretrain_report.html``.

    Returns:
        Path to the generated HTML file.
    """
    output_dir = Path(output_dir)
    metrics_path = output_dir / "metrics.csv"
    config_path = output_dir / "resolved_config.yaml"

    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.csv not found in {output_dir}")

    # Load data
    metrics = _load_metrics(metrics_path)
    if not metrics:
        raise ValueError(f"metrics.csv in {output_dir} is empty")

    config: Dict[str, Any] = {}
    if config_path.exists():
        config = _load_config(config_path)

    summary = _compute_summary(metrics, config)

    # Destination path
    if html_path is None:
        html_path = output_dir / "pretrain_report.html"
    html_path = Path(html_path)

    # Build JSON arrays for Chart.js
    import json

    epochs_json = json.dumps([r["epoch"] for r in metrics])
    train_loss_json = json.dumps([r["train_loss"] for r in metrics])
    val_loss_json = json.dumps([r["val_loss"] for r in metrics])
    train_descriptor_loss_json = json.dumps(
        [r["train_descriptor_loss"] for r in metrics]
    )
    val_descriptor_loss_json = json.dumps(
        [r["val_descriptor_loss"] for r in metrics]
    )
    val_rmse_json = json.dumps([r["val_rmse"] for r in metrics])
    lr_json = json.dumps([r["learning_rate"] for r in metrics])
    alignment_chart = ""
    alignment_json = ""
    alignment_script = ""

    if _has_alignment_metrics(metrics):
        alignment_chart = """
  <div class="chart-card">
    <h3>ECFP-Latent Alignment</h3>
    <canvas id="alignmentChart"></canvas>
  </div>
"""
        alignment_json = (
            f"const trainAlignmentLoss = {json.dumps([r['train_alignment_loss'] for r in metrics])};\n"
            f"const valAlignmentLoss = {json.dumps([r['val_alignment_loss'] for r in metrics])};\n"
            f"const valAlignmentSpearman = {json.dumps([r['val_alignment_spearman'] for r in metrics])};\n"
            f"const valAlignmentKendall = {json.dumps([r['val_alignment_kendall'] for r in metrics])};"
        )
        alignment_script = """
new Chart(document.getElementById('alignmentChart'), {
  type: 'line',
  data: {
    labels: epochs,
    datasets: [
      { label: 'Train Alignment Loss', data: trainAlignmentLoss, borderColor: '#f97316', backgroundColor: 'rgba(249,115,22,0.1)', tension: 0.3, pointRadius: 2, yAxisID: 'y' },
      { label: 'Val Alignment Loss', data: valAlignmentLoss, borderColor: '#facc15', backgroundColor: 'rgba(250,204,21,0.1)', tension: 0.3, pointRadius: 2, yAxisID: 'y' },
      { label: 'Val Spearman', data: valAlignmentSpearman, borderColor: '#a78bfa', backgroundColor: 'rgba(167,139,250,0.1)', tension: 0.3, pointRadius: 2, yAxisID: 'y1' },
      { label: 'Val Kendall', data: valAlignmentKendall, borderColor: '#fb7185', backgroundColor: 'rgba(251,113,133,0.1)', tension: 0.3, pointRadius: 2, yAxisID: 'y1' },
    ],
  },
  options: {
    responsive: true,
    interaction: { mode: 'index', intersect: false },
    plugins: { legend: { labels: { color: tickColor, usePointStyle: true, pointStyle: 'circle' } } },
    scales: {
      x: { title: { display: true, text: 'Epoch', color: tickColor }, ticks: { color: tickColor }, grid: { color: gridColor } },
      y: { title: { display: true, text: 'Alignment Loss', color: tickColor }, ticks: { color: tickColor }, grid: { color: gridColor } },
      y1: { position: 'right', title: { display: true, text: 'Rank Correlation', color: tickColor }, ticks: { color: tickColor }, grid: { drawOnChartArea: false } },
    },
  },
});
"""

    # Build config items HTML
    flat_config = _flatten_config(config)
    config_items = "\n    ".join(
        f'<div class="config-item"><span class="config-key">{k}</span>'
        f'<span class="config-val">{v}</span></div>'
        for k, v in flat_config.items()
    )

    # Build table rows
    best_epoch = summary["best_epoch"]
    table_rows = []
    for r in metrics:
        cls = ' class="best-row"' if r["epoch"] == best_epoch else ""
        table_rows.append(
            f"<tr{cls}>"
            f"<td>{r['epoch']}</td>"
            f"<td>{r['train_loss']:.6f}</td>"
            f"<td>{r['val_loss']:.6f}</td>"
            f"<td>{r['train_descriptor_loss']:.6f}</td>"
            f"<td>{r['val_descriptor_loss']:.6f}</td>"
            f"<td>{r['val_rmse']:.6f}</td>"
            f"<td>{r['learning_rate']:.2e}</td>"
            f"<td>{_fmt_elapsed(r['elapsed_seconds'])}</td>"
            f"</tr>"
        )
    table_rows_html = "\n      ".join(table_rows)

    # Fill template
    html = _HTML_TEMPLATE
    replacements = {
        "{{OUTPUT_DIR}}": str(output_dir.resolve()),
        "{{BEST_VAL_OBJECTIVE}}": f"{summary['best_val_objective']:.4f}",
        "{{BEST_VAL_RMSE}}": f"{summary['best_val_rmse']:.4f}",
        "{{BEST_EPOCH}}": str(summary["best_epoch"]),
        "{{TOTAL_EPOCHS}}": str(summary["total_epochs"]),
        "{{MAX_EPOCHS}}": str(summary["max_epochs"]),
        "{{ELAPSED}}": summary["elapsed"],
        "{{HIDDEN_DIM}}": str(summary["hidden_dim"]),
        "{{NUM_GT_LAYERS}}": str(summary["num_gt_layers"]),
        "{{NUM_HEADS}}": str(summary["num_heads"]),
        "{{CONFIG_ITEMS}}": config_items,
        "{{TABLE_ROWS}}": table_rows_html,
        "{{EPOCHS_JSON}}": epochs_json,
        "{{TRAIN_LOSS_JSON}}": train_loss_json,
        "{{VAL_LOSS_JSON}}": val_loss_json,
        "{{TRAIN_DESCRIPTOR_LOSS_JSON}}": train_descriptor_loss_json,
        "{{VAL_DESCRIPTOR_LOSS_JSON}}": val_descriptor_loss_json,
        "{{VAL_RMSE_JSON}}": val_rmse_json,
        "{{LR_JSON}}": lr_json,
        "{{ALIGNMENT_CHART}}": alignment_chart,
        "{{ALIGNMENT_JSON}}": alignment_json,
        "{{ALIGNMENT_SCRIPT}}": alignment_script,
    }
    for placeholder, value in replacements.items():
        html = html.replace(placeholder, value)

    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html, encoding="utf-8")
    logger.info("Report written to %s", html_path)
    return html_path


def _flatten_config(
    config: Dict[str, Any],
    prefix: str = "",
) -> Dict[str, str]:
    """Flatten a nested config dict into dot-separated key-value strings."""
    flat: Dict[str, str] = {}
    for k, v in config.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            flat.update(_flatten_config(v, key))
        elif isinstance(v, (list, tuple)):
            flat[key] = ", ".join(str(x) for x in v)
        else:
            flat[key] = str(v)
    return flat
