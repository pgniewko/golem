"""HTML report generator for Golem pretraining experiments.

Reads ``metrics.csv`` and ``resolved_config.yaml`` from an experiment output
directory and produces a self-contained Chart.js dashboard with:

- Training & validation loss curves
- Validation RMSE curve
- Learning-rate schedule
- Summary cards (best epoch, best val loss, elapsed time, sample counts)
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

OPTIONAL_METRIC_FIELDS = [
    "train_main_loss",
    "train_rank_loss",
    "train_total_loss",
    "val_rank_loss",
    "val_spearman",
    "val_kendall",
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
                "val_rmse": float(row["val_rmse"]),
                "learning_rate": float(row["learning_rate"]),
                "elapsed_seconds": float(row["elapsed_seconds"]),
            }
            for field in OPTIONAL_METRIC_FIELDS:
                parsed[field] = _parse_optional_float(row, field)
            rows.append(parsed)
    return rows


def _parse_optional_float(row: Dict[str, str], key: str) -> float:
    """Parse an optional float field from a CSV row."""
    raw = row.get(key, "")
    if raw in ("", None):
        return math.nan
    try:
        return float(raw)
    except (TypeError, ValueError):
        return math.nan


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
        "best_val_loss": best_row["val_loss"],
        "best_val_rmse": best_row["val_rmse"],
        "best_val_spearman": _best_optional_metric(metrics, "val_spearman"),
        "best_val_kendall": _best_optional_metric(metrics, "val_kendall"),
        "total_epochs": len(metrics),
        "max_epochs": config.get("max_epochs", "?"),
        "elapsed": _fmt_elapsed(last_row["elapsed_seconds"]),
        "elapsed_seconds": last_row["elapsed_seconds"],
        "final_train_loss": last_row["train_loss"],
        "final_val_loss": last_row["val_loss"],
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


def _best_optional_metric(metrics: List[Dict[str, Any]], key: str) -> float:
    """Return the best finite value for an optional metric or NaN."""
    values = [row.get(key, math.nan) for row in metrics]
    finite = [value for value in values if math.isfinite(value)]
    return max(finite) if finite else math.nan


def _has_geometry_metrics(metrics: List[Dict[str, Any]]) -> bool:
    """Return True if any geometry-specific metric is finite."""
    geometry_fields = [
        "val_rank_loss",
        "val_spearman",
        "val_kendall",
    ]
    return any(
        math.isfinite(row.get(field, math.nan))
        for row in metrics
        for field in geometry_fields
    )


def _format_metric(value: float, precision: int = 6) -> str:
    """Format a metric value for HTML tables."""
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{precision}f}"


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
    <div class="label">Best Val Loss</div>
    <div class="value">{{BEST_VAL_LOSS}}</div>
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
  {{GEOMETRY_CARDS}}
</div>

<!-- Charts -->
<div class="charts">
  <div class="chart-card">
    <h3>Training &amp; Validation Loss</h3>
    <canvas id="lossChart"></canvas>
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
    <h3>Train vs Val Loss Gap</h3>
    <canvas id="gapChart"></canvas>
  </div>
  {{GEOMETRY_CHARTS}}
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
        <th>Epoch</th><th>Train Loss</th><th>Val Loss</th>
        <th>Val RMSE</th><th>LR</th><th>Elapsed</th>
        {{TABLE_EXTRA_HEADERS}}
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
const valRmse = {{VAL_RMSE_JSON}};
const lr = {{LR_JSON}};
const gap = trainLoss.map((t, i) => t - valLoss[i]);
{{GEOMETRY_JSON_DECLS}}

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
      { label: 'Train Loss', data: trainLoss, borderColor: '#38bdf8', backgroundColor: 'rgba(56,189,248,0.1)', tension: 0.3, pointRadius: 2 },
      { label: 'Val Loss',   data: valLoss,   borderColor: '#4ade80', backgroundColor: 'rgba(74,222,128,0.1)', tension: 0.3, pointRadius: 2 },
    ],
  },
  options: makeOpts('MSE Loss'),
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
      { label: 'Train - Val Loss', data: gap, backgroundColor: gap.map(v => v > 0 ? 'rgba(248,113,113,0.6)' : 'rgba(74,222,128,0.6)') },
    ],
  },
  options: makeOpts('Loss Gap'),
});
{{GEOMETRY_CHART_SCRIPTS}}
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
    geometry_enabled = bool(config.get("geometry", {}).get("enabled", False))
    has_geometry_metrics = geometry_enabled or _has_geometry_metrics(metrics)

    # Destination path
    if html_path is None:
        html_path = output_dir / "pretrain_report.html"
    html_path = Path(html_path)

    # Build JSON arrays for Chart.js
    import json

    epochs_json = json.dumps([r["epoch"] for r in metrics])
    train_loss_json = json.dumps([r["train_loss"] for r in metrics])
    val_loss_json = json.dumps([r["val_loss"] for r in metrics])
    val_rmse_json = json.dumps([r["val_rmse"] for r in metrics])
    lr_json = json.dumps([r["learning_rate"] for r in metrics])
    geometry_json_decls = ""
    geometry_chart_scripts = ""
    geometry_cards = ""
    geometry_charts = ""
    table_extra_headers = ""

    if has_geometry_metrics:
        train_main_json = json.dumps([r["train_main_loss"] for r in metrics])
        train_rank_json = json.dumps([r["train_rank_loss"] for r in metrics])
        train_total_json = json.dumps([r["train_total_loss"] for r in metrics])
        val_rank_json = json.dumps([r["val_rank_loss"] for r in metrics])
        val_spearman_json = json.dumps([r["val_spearman"] for r in metrics])
        val_kendall_json = json.dumps([r["val_kendall"] for r in metrics])
        geometry_json_decls = (
            f"const trainMainLoss = {train_main_json};\n"
            f"const trainRankLoss = {train_rank_json};\n"
            f"const trainTotalLoss = {train_total_json};\n"
            f"const valRankLoss = {val_rank_json};\n"
            f"const valSpearman = {val_spearman_json};\n"
            f"const valKendall = {val_kendall_json};"
        )
        geometry_chart_scripts = """
new Chart(document.getElementById('geometryLossChart'), {
  type: 'line',
  data: {
    labels: epochs,
    datasets: [
      { label: 'Train Main Loss',  data: trainMainLoss,  borderColor: '#38bdf8', backgroundColor: 'rgba(56,189,248,0.1)', tension: 0.3, pointRadius: 2 },
      { label: 'Train Rank Loss',  data: trainRankLoss,  borderColor: '#f97316', backgroundColor: 'rgba(249,115,22,0.1)', tension: 0.3, pointRadius: 2 },
      { label: 'Train Total Loss', data: trainTotalLoss, borderColor: '#4ade80', backgroundColor: 'rgba(74,222,128,0.1)', tension: 0.3, pointRadius: 2 },
    ],
  },
  options: makeOpts('Train Loss Components'),
});

new Chart(document.getElementById('geometryMetricChart'), {
  type: 'line',
  data: {
    labels: epochs,
    datasets: [
      { label: 'Val Rank Loss', data: valRankLoss, borderColor: '#facc15', backgroundColor: 'rgba(250,204,21,0.1)', tension: 0.3, pointRadius: 2, yAxisID: 'y' },
      { label: 'Val Spearman',  data: valSpearman, borderColor: '#c084fc', backgroundColor: 'rgba(192,132,252,0.1)', tension: 0.3, pointRadius: 2, yAxisID: 'y1' },
      { label: 'Val Kendall',   data: valKendall, borderColor: '#fb7185', backgroundColor: 'rgba(251,113,133,0.1)', tension: 0.3, pointRadius: 2, yAxisID: 'y1' },
    ],
  },
  options: {
    responsive: true,
    interaction: { mode: 'index', intersect: false },
    plugins: { legend: { labels: { color: tickColor, usePointStyle: true, pointStyle: 'circle' } } },
    scales: {
      x: { title: { display: true, text: 'Epoch', color: tickColor }, ticks: { color: tickColor }, grid: { color: gridColor } },
      y: { title: { display: true, text: 'Val Rank Loss', color: tickColor }, ticks: { color: tickColor }, grid: { color: gridColor } },
      y1: { position: 'right', title: { display: true, text: 'Rank Correlation', color: tickColor }, ticks: { color: tickColor }, grid: { drawOnChartArea: false } },
    },
  },
});
"""
        geometry_cards = (
            '<div class="card info">'
            '<div class="label">Best Val Spearman</div>'
            f'<div class="value">{_format_metric(summary["best_val_spearman"], precision=4)}</div>'
            '<div class="detail">Pair-distance rank agreement</div>'
            '</div>\n'
            '  <div class="card info">'
            '<div class="label">Best Val Kendall</div>'
            f'<div class="value">{_format_metric(summary["best_val_kendall"], precision=4)}</div>'
            '<div class="detail">Pair-distance rank agreement</div>'
            '</div>'
        )
        geometry_charts = """
  <div class="chart-card">
    <h3>Geometry Loss Components</h3>
    <canvas id="geometryLossChart"></canvas>
  </div>
  <div class="chart-card">
    <h3>Geometry Validation Metrics</h3>
    <canvas id="geometryMetricChart"></canvas>
  </div>
"""
        table_extra_headers = (
            "<th>Train Main</th><th>Train Rank</th><th>Train Total</th>"
            "<th>Val Rank</th><th>Val Spearman</th><th>Val Kendall</th>"
        )

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
        extra_cells = ""
        if has_geometry_metrics:
            extra_cells = (
                f"<td>{_format_metric(r['train_main_loss'])}</td>"
                f"<td>{_format_metric(r['train_rank_loss'])}</td>"
                f"<td>{_format_metric(r['train_total_loss'])}</td>"
                f"<td>{_format_metric(r['val_rank_loss'])}</td>"
                f"<td>{_format_metric(r['val_spearman'])}</td>"
                f"<td>{_format_metric(r['val_kendall'])}</td>"
            )
        table_rows.append(
            f"<tr{cls}>"
            f"<td>{r['epoch']}</td>"
            f"<td>{r['train_loss']:.6f}</td>"
            f"<td>{r['val_loss']:.6f}</td>"
            f"<td>{r['val_rmse']:.6f}</td>"
            f"<td>{r['learning_rate']:.2e}</td>"
            f"<td>{_fmt_elapsed(r['elapsed_seconds'])}</td>"
            f"{extra_cells}"
            f"</tr>"
        )
    table_rows_html = "\n      ".join(table_rows)

    # Fill template
    html = _HTML_TEMPLATE
    replacements = {
        "{{OUTPUT_DIR}}": str(output_dir.resolve()),
        "{{BEST_VAL_LOSS}}": f"{summary['best_val_loss']:.4f}",
        "{{BEST_VAL_RMSE}}": f"{summary['best_val_rmse']:.4f}",
        "{{BEST_EPOCH}}": str(summary["best_epoch"]),
        "{{TOTAL_EPOCHS}}": str(summary["total_epochs"]),
        "{{MAX_EPOCHS}}": str(summary["max_epochs"]),
        "{{ELAPSED}}": summary["elapsed"],
        "{{HIDDEN_DIM}}": str(summary["hidden_dim"]),
        "{{NUM_GT_LAYERS}}": str(summary["num_gt_layers"]),
        "{{NUM_HEADS}}": str(summary["num_heads"]),
        "{{GEOMETRY_CARDS}}": geometry_cards,
        "{{GEOMETRY_CHARTS}}": geometry_charts,
        "{{TABLE_EXTRA_HEADERS}}": table_extra_headers,
        "{{CONFIG_ITEMS}}": config_items,
        "{{TABLE_ROWS}}": table_rows_html,
        "{{EPOCHS_JSON}}": epochs_json,
        "{{TRAIN_LOSS_JSON}}": train_loss_json,
        "{{VAL_LOSS_JSON}}": val_loss_json,
        "{{VAL_RMSE_JSON}}": val_rmse_json,
        "{{LR_JSON}}": lr_json,
        "{{GEOMETRY_JSON_DECLS}}": geometry_json_decls,
        "{{GEOMETRY_CHART_SCRIPTS}}": geometry_chart_scripts,
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
