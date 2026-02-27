"""Configuration dataclasses with optional YAML loading."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass
class ModelConfig:
    """Graph Transformer model architecture config."""

    hidden_dim: int = 128
    num_gt_layers: int = 4
    num_heads: int = 8
    norm: str = "bn"
    gt_aggregators: List[str] = field(default_factory=lambda: ["sum", "mean"])
    aggregators: List[str] = field(
        default_factory=lambda: ["sum", "mean", "max", "std"]
    )
    dropout: float = 0.3
    act: str = "gelu"
    gate: bool = True
    qkv_bias: bool = False


@dataclass
class IsoformConfig:
    """Isoform enumeration config."""

    enabled: bool = True
    desalting: bool = True
    tautomers: bool = True
    max_tautomers: int = 10
    protonation: bool = True
    max_protomers: int = 10
    ph_range: Tuple[float, float] = (6.4, 8.4)
    neutralization: bool = True
    rdkit_fallback: bool = False


@dataclass
class PretrainConfig:
    """Full pretraining pipeline config."""

    model: ModelConfig = field(default_factory=ModelConfig)
    isoforms: IsoformConfig = field(default_factory=IsoformConfig)
    masking_ratio: float = 0.15
    batch_size: int = 128
    max_epochs: int = 500
    patience: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 25
    num_workers: int = 4
    winsorize_range: Tuple[float, float] = (-6.0, 6.0)
    split_ratios: List[float] = field(default_factory=lambda: [0.7, 0.2, 0.1])
    seed: int = 42  # pretrain seed (finetune notebooks use 1928374650)


def _deep_update(base: dict, overrides: dict) -> dict:
    """Recursively update base dict with overrides."""
    result = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_update(result[k], v)
        else:
            result[k] = v
    return result


def _dict_to_config(d: dict) -> PretrainConfig:
    """Build PretrainConfig from a flat/nested dict."""
    model_d = d.pop("model", {})
    isoform_d = d.pop("isoforms", {})

    # Handle nested YAML structures for isoforms
    if "tautomers" in isoform_d and isinstance(isoform_d["tautomers"], dict):
        taut = isoform_d.pop("tautomers")
        isoform_d["tautomers"] = taut.get("enabled", True)
        if "max_tautomers" in taut:
            isoform_d["max_tautomers"] = taut["max_tautomers"]
    if "protonation" in isoform_d and isinstance(isoform_d["protonation"], dict):
        prot = isoform_d.pop("protonation")
        isoform_d["protonation"] = prot.get("enabled", True)
        if "ph_range" in prot:
            isoform_d["ph_range"] = tuple(prot["ph_range"])
        if "max_protomers" in prot:
            isoform_d["max_protomers"] = prot["max_protomers"]
    if "neutralization" in isoform_d and isinstance(isoform_d["neutralization"], dict):
        neut = isoform_d.pop("neutralization")
        isoform_d["neutralization"] = neut.get("enabled", True)
    if "desalting" in isoform_d and isinstance(isoform_d["desalting"], dict):
        desalt = isoform_d.pop("desalting")
        isoform_d["desalting"] = desalt.get("enabled", True)
    if "rdkit_fallback" in isoform_d and isinstance(isoform_d["rdkit_fallback"], dict):
        fb = isoform_d.pop("rdkit_fallback")
        isoform_d["rdkit_fallback"] = fb.get("enabled", False)

    # Remove keys not in dataclass (e.g. deduplication, tool, fallback)
    model_fields = {f.name for f in fields(ModelConfig)}
    isoform_fields = {f.name for f in fields(IsoformConfig)}
    pretrain_fields = {f.name for f in fields(PretrainConfig)}

    model_d = {k: v for k, v in model_d.items() if k in model_fields}
    isoform_d = {k: v for k, v in isoform_d.items() if k in isoform_fields}

    # Handle residual "pretrain" sub-key (already flattened in load_config,
    # but handle gracefully if _dict_to_config is called directly)
    pretrain_d = d.pop("pretrain", {})
    for k, v in pretrain_d.items():
        if k not in d:
            d[k] = v

    # Convert tuple fields
    if "winsorize_range" in d and isinstance(d["winsorize_range"], list):
        d["winsorize_range"] = tuple(d["winsorize_range"])
    if "ph_range" in isoform_d and isinstance(isoform_d["ph_range"], list):
        isoform_d["ph_range"] = tuple(isoform_d["ph_range"])

    # Filter to known fields
    d = {k: v for k, v in d.items() if k in pretrain_fields}

    return PretrainConfig(
        model=ModelConfig(**model_d),
        isoforms=IsoformConfig(**isoform_d),
        **d,
    )


def load_config(
    yaml_path: Optional[str] = None, **cli_overrides: Any
) -> PretrainConfig:
    """Load config by merging: defaults <- YAML <- CLI overrides.

    Args:
        yaml_path: Optional path to YAML config file.
        **cli_overrides: CLI arguments that override config values.
            Keys with None values are ignored.
    """
    # Start with defaults
    defaults = asdict(PretrainConfig())

    # Layer YAML on top
    if yaml_path is not None:
        with open(yaml_path) as f:
            yaml_data = yaml.safe_load(f) or {}

        # Flatten YAML 'pretrain' sub-key to top level BEFORE merging
        # so that CLI overrides (which are always top-level) can win.
        if "pretrain" in yaml_data:
            pretrain_sub = yaml_data.pop("pretrain")
            for k, v in pretrain_sub.items():
                if k not in yaml_data:  # don't clobber explicitly set top-level keys
                    yaml_data[k] = v

        defaults = _deep_update(defaults, yaml_data)

    # Layer CLI overrides on top (skip None values)
    cli_clean = {k: v for k, v in cli_overrides.items() if v is not None}
    if cli_clean:
        defaults = _deep_update(defaults, cli_clean)

    return _dict_to_config(defaults)
