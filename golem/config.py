"""Configuration dataclasses with optional YAML loading."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field, asdict, fields
from numbers import Integral, Real
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


def _validate_fraction_field(
    name: str,
    value: Any,
    *,
    minimum_exclusive: float,
    maximum_inclusive: float,
) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number.")
    if not (minimum_exclusive < float(value) <= maximum_inclusive):
        raise ValueError(
            f"{name} must be in the interval ({minimum_exclusive}, {maximum_inclusive}]."
        )


def _validate_real_field(
    name: str,
    value: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number.")
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite.")
    if minimum is not None:
        if minimum_inclusive:
            if numeric_value < minimum:
                raise ValueError(f"{name} must be >= {minimum}.")
        elif numeric_value <= minimum:
            raise ValueError(f"{name} must be > {minimum}.")
    if maximum is not None:
        if maximum_inclusive:
            if numeric_value > maximum:
                raise ValueError(f"{name} must be <= {maximum}.")
        elif numeric_value >= maximum:
            raise ValueError(f"{name} must be < {maximum}.")


def _validate_range_field(
    name: str,
    value: Any,
    *,
    allow_equal: bool,
) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a length-2 sequence.")
    lo, hi = value
    _validate_real_field(f"{name}[0]", lo)
    _validate_real_field(f"{name}[1]", hi)
    if allow_equal:
        if float(lo) > float(hi):
            raise ValueError(f"{name}[0] must be <= {name}[1].")
    elif float(lo) >= float(hi):
        raise ValueError(f"{name}[0] must be < {name}[1].")


def _validate_split_ratios(name: str, fractions: Any) -> None:
    if not isinstance(fractions, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple.")
    if len(fractions) not in (2, 3):
        raise ValueError(f"{name} must contain exactly 2 or 3 ratios.")
    total = 0.0
    for idx, frac in enumerate(fractions):
        _validate_real_field(
            f"{name}[{idx}]",
            frac,
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=False,
        )
        total += float(frac)
    if abs(total - 1.0) >= 1e-6:
        raise ValueError(f"{name} must sum to 1.0, got {total}.")


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
    num_workers: int = 0
    subsample: float | None = None
    winsorize_range: Tuple[float, float] = (-6.0, 6.0)
    split_ratios: List[float] = field(default_factory=lambda: [0.7, 0.2, 0.1])
    seed: int = 42  # pretrain seed (finetune notebooks use 1928374650)


def validate_pretrain_config(config: PretrainConfig) -> PretrainConfig:
    """Validate a resolved config and return it unchanged."""
    _validate_integer_field("model.hidden_dim", config.model.hidden_dim, minimum=1)
    _validate_integer_field("model.num_gt_layers", config.model.num_gt_layers, minimum=1)
    _validate_integer_field("model.num_heads", config.model.num_heads, minimum=1)
    if config.model.hidden_dim % config.model.num_heads != 0:
        raise ValueError(
            "model.hidden_dim must be divisible by model.num_heads."
        )
    _validate_integer_field("model.num_head_layers", config.model.num_head_layers, minimum=1)
    _validate_real_field(
        "model.dropout",
        config.model.dropout,
        minimum=0.0,
        maximum=1.0,
        maximum_inclusive=False,
    )
    if config.model.head_dropout is not None:
        _validate_real_field(
            "model.head_dropout",
            config.model.head_dropout,
            minimum=0.0,
            maximum=1.0,
            maximum_inclusive=False,
        )

    _validate_integer_field("batch_size", config.batch_size, minimum=1)
    _validate_integer_field("max_epochs", config.max_epochs, minimum=1)
    _validate_integer_field("patience", config.patience, minimum=1)
    _validate_integer_field("warmup_epochs", config.warmup_epochs, minimum=0)
    _validate_integer_field("num_workers", config.num_workers, minimum=0)
    _validate_integer_field("seed", config.seed, minimum=0)
    _validate_real_field("lr", config.lr, minimum=0.0, minimum_inclusive=False)
    _validate_real_field("weight_decay", config.weight_decay, minimum=0.0)
    _validate_fraction_field(
        "masking_ratio",
        config.masking_ratio,
        minimum_exclusive=0.0,
        maximum_inclusive=1.0,
    )
    _validate_range_field("winsorize_range", config.winsorize_range, allow_equal=False)
    _validate_split_ratios("split_ratios", config.split_ratios)
    if config.subsample is not None:
        _validate_fraction_field(
            "subsample",
            config.subsample,
            minimum_exclusive=0.0,
            maximum_inclusive=1.0,
        )

    _validate_integer_field("isoforms.max_tautomers", config.isoforms.max_tautomers, minimum=1)
    _validate_integer_field("isoforms.max_protomers", config.isoforms.max_protomers, minimum=1)
    _validate_range_field("isoforms.ph_range", config.isoforms.ph_range, allow_equal=True)

    if not config.descriptors.include_2d_targets and not config.descriptors.include_3d_targets:
        raise ValueError("At least one descriptor target family must be enabled.")
    _validate_real_field("descriptors.loss_weight", config.descriptors.loss_weight, minimum=0.0)

    _validate_integer_field("conformers.n_generate", config.conformers.n_generate, minimum=1)

    _validate_real_field(
        "ecfp_latent_alignment.weight",
        config.ecfp_latent_alignment.weight,
        minimum=0.0,
    )
    _validate_integer_field(
        "ecfp_latent_alignment.fp_bits",
        config.ecfp_latent_alignment.fp_bits,
        minimum=1,
    )
    _validate_integer_field(
        "ecfp_latent_alignment.fp_radius",
        config.ecfp_latent_alignment.fp_radius,
        minimum=0,
    )
    _validate_integer_field(
        "ecfp_latent_alignment.num_pairs",
        config.ecfp_latent_alignment.num_pairs,
        minimum=1,
    )
    _validate_real_field(
        "ecfp_latent_alignment.temperature",
        config.ecfp_latent_alignment.temperature,
        minimum=0.0,
        minimum_inclusive=False,
    )
    _validate_real_field(
        "ecfp_latent_alignment.tie_epsilon",
        config.ecfp_latent_alignment.tie_epsilon,
        minimum=0.0,
    )

    if (
        config.descriptors.loss_weight == 0.0
        and (
            not config.ecfp_latent_alignment.enabled
            or config.ecfp_latent_alignment.weight == 0.0
        )
    ):
        raise ValueError(
            "At least one training objective must have a positive weight. "
            "Set descriptors.loss_weight > 0 or enable ecfp_latent_alignment "
            "with weight > 0."
        )

    return config


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
    return validate_pretrain_config(config)


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
