"""Configuration dataclasses with optional YAML loading."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, asdict, fields
from numbers import Integral
from typing import Any, List, Optional, Tuple

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
    num_head_layers: int = 1
    head_norm: bool = False
    head_residual: bool = False
    head_dropout: float | None = None


@dataclass
class IsoformConfig:
    """Isoform enumeration config."""

    enabled: bool = True
    desalting: bool = False
    tautomers: bool = True
    max_tautomers: int = 10
    protonation: bool = True
    max_protomers: int = 10
    ph_range: Tuple[float, float] = (6.4, 8.4)
    neutralization: bool = True
    rdkit_fallback: bool = False


@dataclass
class ECFPLatentAlignmentConfig:
    """Optional ECFP-to-latent metric alignment config."""

    enabled: bool = False
    weight: float = 0.05
    fp_bits: int = 2048
    fp_radius: int = 2
    num_pairs: int = 128
    temperature: float = 0.1
    tie_epsilon: float = 0.02
    warmup_epochs: int = 5
    log_rank_metrics: bool = True


@dataclass
class Descriptor3DSettings:
    """Optional 3D descriptor-target settings."""

    rdkit_include_getaway: bool = False


@dataclass
class DescriptorConfig:
    """Descriptor target settings for pretraining."""

    include_2d_targets: bool = True
    include_3d_targets: bool = False
    loss_weight: float = 1.0
    three_d_settings: Descriptor3DSettings = field(default_factory=Descriptor3DSettings)


@dataclass
class ConformerConfig:
    """Offline conformer generation settings for 3D descriptor targets."""

    n_generate: int = 8


def _reject_unknown_keys(section_name: str, values: dict, allowed_keys: set[str]) -> None:
    unknown_keys = sorted(set(values) - allowed_keys)
    if unknown_keys:
        prefix = f"{section_name}." if section_name else ""
        unknown = ", ".join(f"{prefix}{key}" for key in unknown_keys)
        raise ValueError(f"Unknown config keys: {unknown}.")


def _pop_section_dict(d: dict, key: str) -> dict:
    value = d.pop(key, None)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping if provided.")
    return value


def _validate_integer_field(name: str, value: Any, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")


@dataclass
class PretrainConfig:
    """Full pretraining pipeline config."""

    model: ModelConfig = field(default_factory=ModelConfig)
    isoforms: IsoformConfig = field(default_factory=IsoformConfig)
    descriptors: DescriptorConfig = field(default_factory=DescriptorConfig)
    conformers: ConformerConfig = field(default_factory=ConformerConfig)
    ecfp_latent_alignment: ECFPLatentAlignmentConfig = field(
        default_factory=ECFPLatentAlignmentConfig
    )
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
    result = deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_update(result[k], v)
        else:
            result[k] = v
    return result


def _dict_to_config(d: dict) -> PretrainConfig:
    """Build PretrainConfig from a flat/nested dict."""
    model_d = _pop_section_dict(d, "model")
    isoform_d = _pop_section_dict(d, "isoforms")
    descriptors_d = _pop_section_dict(d, "descriptors")
    conformers_d = _pop_section_dict(d, "conformers")
    alignment_d = _pop_section_dict(d, "ecfp_latent_alignment")
    descriptor3d_d = _pop_section_dict(descriptors_d, "three_d_settings")

    # Handle nested YAML structures for isoforms
    if "tautomers" in isoform_d and isinstance(isoform_d["tautomers"], dict):
        taut = isoform_d.pop("tautomers")
        _reject_unknown_keys("isoforms.tautomers", taut, {"enabled", "max_tautomers"})
        isoform_d["tautomers"] = taut.get("enabled", True)
        if "max_tautomers" in taut:
            isoform_d["max_tautomers"] = taut["max_tautomers"]
    if "protonation" in isoform_d and isinstance(isoform_d["protonation"], dict):
        prot = isoform_d.pop("protonation")
        _reject_unknown_keys(
            "isoforms.protonation",
            prot,
            {"enabled", "max_protomers", "ph_range"},
        )
        isoform_d["protonation"] = prot.get("enabled", True)
        if "ph_range" in prot:
            isoform_d["ph_range"] = tuple(prot["ph_range"])
        if "max_protomers" in prot:
            isoform_d["max_protomers"] = prot["max_protomers"]
    if "neutralization" in isoform_d and isinstance(isoform_d["neutralization"], dict):
        neut = isoform_d.pop("neutralization")
        _reject_unknown_keys("isoforms.neutralization", neut, {"enabled"})
        isoform_d["neutralization"] = neut.get("enabled", True)
    if "desalting" in isoform_d and isinstance(isoform_d["desalting"], dict):
        desalt = isoform_d.pop("desalting")
        _reject_unknown_keys("isoforms.desalting", desalt, {"enabled"})
        isoform_d["desalting"] = desalt.get("enabled", False)
    if "rdkit_fallback" in isoform_d and isinstance(isoform_d["rdkit_fallback"], dict):
        fb = isoform_d.pop("rdkit_fallback")
        _reject_unknown_keys("isoforms.rdkit_fallback", fb, {"enabled"})
        isoform_d["rdkit_fallback"] = fb.get("enabled", False)

    removed_conformer_keys = {"energy_window_kcal", "n_keep", "prune_rms", "timeout_seconds"} & set(
        conformers_d
    )
    if removed_conformer_keys:
        removed = ", ".join(f"conformers.{key}" for key in sorted(removed_conformer_keys))
        raise ValueError(
            f"Removed config keys: {removed}. "
            "Generate conformers with conformers.n_generate; the lowest-energy conformer is kept."
        )

    model_fields = {f.name for f in fields(ModelConfig)}
    isoform_fields = {f.name for f in fields(IsoformConfig)}
    descriptor_fields = {f.name for f in fields(DescriptorConfig)}
    descriptor3d_fields = {f.name for f in fields(Descriptor3DSettings)}
    conformer_fields = {f.name for f in fields(ConformerConfig)}
    alignment_fields = {f.name for f in fields(ECFPLatentAlignmentConfig)}
    pretrain_fields = {f.name for f in fields(PretrainConfig)}
    scalar_pretrain_fields = pretrain_fields - {
        "model",
        "isoforms",
        "descriptors",
        "conformers",
        "ecfp_latent_alignment",
    }

    _reject_unknown_keys("model", model_d, model_fields)
    _reject_unknown_keys("isoforms", isoform_d, isoform_fields)
    _reject_unknown_keys("descriptors", descriptors_d, descriptor_fields)
    _reject_unknown_keys("conformers", conformers_d, conformer_fields)
    _reject_unknown_keys("descriptors.three_d_settings", descriptor3d_d, descriptor3d_fields)
    _reject_unknown_keys("ecfp_latent_alignment", alignment_d, alignment_fields)

    # Handle residual "pretrain" sub-key (already flattened in load_config,
    # but handle gracefully if _dict_to_config is called directly)
    pretrain_d = _pop_section_dict(d, "pretrain")
    _reject_unknown_keys("pretrain", pretrain_d, scalar_pretrain_fields)
    for k, v in pretrain_d.items():
        if k not in d:
            d[k] = v

    # Convert tuple fields
    if "winsorize_range" in d and isinstance(d["winsorize_range"], list):
        d["winsorize_range"] = tuple(d["winsorize_range"])
    if "ph_range" in isoform_d and isinstance(isoform_d["ph_range"], list):
        isoform_d["ph_range"] = tuple(isoform_d["ph_range"])

    _reject_unknown_keys("", d, scalar_pretrain_fields)

    config = PretrainConfig(
        model=ModelConfig(**model_d),
        isoforms=IsoformConfig(**isoform_d),
        descriptors=DescriptorConfig(
            three_d_settings=Descriptor3DSettings(**descriptor3d_d),
            **descriptors_d,
        ),
        conformers=ConformerConfig(**conformers_d),
        ecfp_latent_alignment=ECFPLatentAlignmentConfig(**alignment_d),
        **d,
    )
    if config.descriptors.loss_weight < 0:
        raise ValueError("descriptors.loss_weight must be >= 0.")
    _validate_integer_field("conformers.n_generate", config.conformers.n_generate, minimum=1)
    return config


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
