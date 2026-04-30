"""Configuration dataclasses with optional YAML loading."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from numbers import Integral, Real
from typing import Any, List, Literal, Optional, Tuple

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
    target_mode: Literal["lowest_energy", "boltzmann"] = "lowest_energy"


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
    n_keep_best: int = 3
    max_delta_energy_kcal: float = 3.0


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
    device: str = "auto"
    subsample: float | None = None
    winsorize_range: Tuple[float, float] = (-6.0, 6.0)
    split_ratios: List[float] = field(default_factory=lambda: [0.7, 0.2, 0.1])
    seed: int = 42  # pretrain seed (finetune notebooks use 1928374650)

    def __post_init__(self) -> None:
        if isinstance(self.device, str):
            self.device = self.device.strip().lower()


_SECTION_TYPES = {
    "model": ModelConfig,
    "isoforms": IsoformConfig,
    "descriptors": DescriptorConfig,
    "conformers": ConformerConfig,
    "ecfp_latent_alignment": ECFPLatentAlignmentConfig,
}
_SECTION_FIELDS = {
    name: {field.name for field in fields(section_type)}
    for name, section_type in _SECTION_TYPES.items()
}
_SECTION_FIELDS["descriptors.three_d_settings"] = {
    field.name for field in fields(Descriptor3DSettings)
}
_PRETRAIN_SCALAR_FIELDS = {field.name for field in fields(PretrainConfig)} - set(
    _SECTION_TYPES
)
_INTEGER_RULES = {
    "model.hidden_dim": 1,
    "model.num_gt_layers": 1,
    "model.num_heads": 1,
    "model.num_head_layers": 1,
    "batch_size": 1,
    "max_epochs": 1,
    "patience": 1,
    "warmup_epochs": 0,
    "num_workers": 0,
    "seed": 0,
    "isoforms.max_tautomers": 1,
    "isoforms.max_protomers": 1,
    "conformers.n_generate": 1,
    "conformers.n_keep_best": 1,
    "ecfp_latent_alignment.fp_bits": 1,
    "ecfp_latent_alignment.fp_radius": 0,
    "ecfp_latent_alignment.num_pairs": 1,
}
_REAL_RULES = {
    "model.dropout": (0.0, 1.0, True, False),
    "model.head_dropout": (0.0, 1.0, True, False),
    "lr": (0.0, None, False, True),
    "weight_decay": (0.0, None, True, True),
    "masking_ratio": (0.0, 1.0, False, True),
    "subsample": (0.0, 1.0, False, True),
    "descriptors.loss_weight": (0.0, None, True, True),
    "conformers.max_delta_energy_kcal": (0.0, None, False, True),
    "ecfp_latent_alignment.weight": (0.0, None, True, True),
    "ecfp_latent_alignment.temperature": (0.0, None, False, True),
    "ecfp_latent_alignment.tie_epsilon": (0.0, None, True, True),
}
_RANGE_RULES = {
    "winsorize_range": False,
    "isoforms.ph_range": True,
}
_REMOVED_CONFORMER_KEYS = {
    "energy_window_kcal",
    "n_keep",
    "prune_rms",
    "timeout_seconds",
}
_ISOFORM_BLOCK_RULES = {
    "tautomers": ("isoforms.tautomers", {"enabled", "max_tautomers"}, True),
    "protonation": (
        "isoforms.protonation",
        {"enabled", "max_protomers", "ph_range"},
        True,
    ),
    "neutralization": ("isoforms.neutralization", {"enabled"}, True),
    "desalting": ("isoforms.desalting", {"enabled"}, False),
    "rdkit_fallback": ("isoforms.rdkit_fallback", {"enabled"}, False),
}
_DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")
_THREE_D_TARGET_MODE_CHOICES = ("lowest_energy", "boltzmann")


def _reject_unknown_keys(
    section_name: str, values: dict[str, Any], allowed_keys: set[str]
) -> None:
    unknown_keys = sorted(set(values) - allowed_keys)
    if unknown_keys:
        prefix = f"{section_name}." if section_name else ""
        unknown = ", ".join(f"{prefix}{key}" for key in unknown_keys)
        raise ValueError(f"Unknown config keys: {unknown}.")


def _pop_mapping(values: dict[str, Any], key: str) -> dict[str, Any]:
    value = values.pop(key, None)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping if provided.")
    return value


def _flatten_pretrain(values: dict[str, Any]) -> None:
    pretrain_values = _pop_mapping(values, "pretrain")
    _reject_unknown_keys("pretrain", pretrain_values, _PRETRAIN_SCALAR_FIELDS)
    for key, value in pretrain_values.items():
        values.setdefault(key, value)


def _coerce_tuple_field(values: dict[str, Any], key: str) -> None:
    if isinstance(values.get(key), list):
        values[key] = tuple(values[key])


def _normalize_isoform_blocks(values: dict[str, Any]) -> None:
    for key, (section_name, allowed_keys, default_enabled) in _ISOFORM_BLOCK_RULES.items():
        block = values.get(key)
        if not isinstance(block, dict):
            continue
        _reject_unknown_keys(section_name, block, allowed_keys)
        values[key] = block.get("enabled", default_enabled)
        for nested_key in allowed_keys - {"enabled"}:
            if nested_key not in block:
                continue
            value = block[nested_key]
            if nested_key == "ph_range" and isinstance(value, list):
                value = tuple(value)
            values[nested_key] = value


def _resolve_path(config: Any, path: str) -> Any:
    value = config
    for part in path.split("."):
        value = getattr(value, part)
    return value


def _validate_integer(name: str, value: Any, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")


def _validate_real(
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
        if minimum_inclusive and numeric_value < minimum:
            raise ValueError(f"{name} must be >= {minimum}.")
        if not minimum_inclusive and numeric_value <= minimum:
            raise ValueError(f"{name} must be > {minimum}.")
    if maximum is not None:
        if maximum_inclusive and numeric_value > maximum:
            raise ValueError(f"{name} must be <= {maximum}.")
        if not maximum_inclusive and numeric_value >= maximum:
            raise ValueError(f"{name} must be < {maximum}.")


def _validate_range(name: str, value: Any, *, allow_equal: bool) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a length-2 sequence.")
    lo, hi = value
    _validate_real(f"{name}[0]", lo)
    _validate_real(f"{name}[1]", hi)
    if allow_equal:
        if float(lo) > float(hi):
            raise ValueError(f"{name}[0] must be <= {name}[1].")
    elif float(lo) >= float(hi):
        raise ValueError(f"{name}[0] must be < {name}[1].")


def _validate_split_ratios(name: str, values: Any) -> None:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple.")
    if len(values) not in (2, 3):
        raise ValueError(f"{name} must contain exactly 2 or 3 ratios.")
    total = 0.0
    for index, value in enumerate(values):
        _validate_real(
            f"{name}[{index}]",
            value,
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=False,
        )
        total += float(value)
    if abs(total - 1.0) >= 1e-6:
        raise ValueError(f"{name} must sum to 1.0, got {total}.")


def validate_pretrain_config(config: PretrainConfig) -> PretrainConfig:
    """Validate a resolved config and return it unchanged."""
    for path, minimum in _INTEGER_RULES.items():
        _validate_integer(path, _resolve_path(config, path), minimum)
    for path, rule in _REAL_RULES.items():
        value = _resolve_path(config, path)
        if value is None:
            continue
        minimum, maximum, minimum_inclusive, maximum_inclusive = rule
        _validate_real(
            path,
            value,
            minimum=minimum,
            maximum=maximum,
            minimum_inclusive=minimum_inclusive,
            maximum_inclusive=maximum_inclusive,
        )
    for path, allow_equal in _RANGE_RULES.items():
        _validate_range(path, _resolve_path(config, path), allow_equal=allow_equal)
    _validate_split_ratios("split_ratios", config.split_ratios)

    if config.device not in _DEVICE_CHOICES:
        allowed = ", ".join(_DEVICE_CHOICES)
        raise ValueError(f"device must be one of {allowed}.")
    if config.descriptors.three_d_settings.target_mode not in _THREE_D_TARGET_MODE_CHOICES:
        allowed = ", ".join(_THREE_D_TARGET_MODE_CHOICES)
        raise ValueError(
            "descriptors.three_d_settings.target_mode must be one of "
            f"{allowed}."
        )

    if config.model.hidden_dim % config.model.num_heads != 0:
        raise ValueError("model.hidden_dim must be divisible by model.num_heads.")
    uses_boltzmann_3d_targets = (
        config.descriptors.include_3d_targets
        and config.descriptors.three_d_settings.target_mode == "boltzmann"
    )
    if (
        uses_boltzmann_3d_targets
        and config.conformers.n_keep_best > config.conformers.n_generate
    ):
        raise ValueError("conformers.n_keep_best must be <= conformers.n_generate.")
    if not (
        config.descriptors.include_2d_targets or config.descriptors.include_3d_targets
    ):
        raise ValueError("At least one descriptor target family must be enabled.")
    if config.descriptors.loss_weight == 0.0 and (
        not config.ecfp_latent_alignment.enabled
        or config.ecfp_latent_alignment.weight == 0.0
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
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _dict_to_config(values: dict[str, Any]) -> PretrainConfig:
    """Build PretrainConfig from a flat/nested dict."""
    values = deepcopy(values)
    _flatten_pretrain(values)
    sections = {name: _pop_mapping(values, name) for name in _SECTION_TYPES}
    descriptor_3d = _pop_mapping(sections["descriptors"], "three_d_settings")

    _normalize_isoform_blocks(sections["isoforms"])
    _coerce_tuple_field(values, "winsorize_range")
    _coerce_tuple_field(sections["isoforms"], "ph_range")

    removed_conformer_keys = sorted(
        _REMOVED_CONFORMER_KEYS.intersection(sections["conformers"])
    )
    if removed_conformer_keys:
        removed = ", ".join(f"conformers.{key}" for key in removed_conformer_keys)
        raise ValueError(
            f"Removed config keys: {removed}. "
            "Use conformers.n_generate, conformers.n_keep_best, and "
            "conformers.max_delta_energy_kcal instead."
        )

    _reject_unknown_keys("", values, _PRETRAIN_SCALAR_FIELDS)
    for name, section_values in sections.items():
        _reject_unknown_keys(name, section_values, _SECTION_FIELDS[name])
    _reject_unknown_keys(
        "descriptors.three_d_settings",
        descriptor_3d,
        _SECTION_FIELDS["descriptors.three_d_settings"],
    )

    config = PretrainConfig(
        **values,
        model=ModelConfig(**sections["model"]),
        isoforms=IsoformConfig(**sections["isoforms"]),
        descriptors=DescriptorConfig(
            **sections["descriptors"],
            three_d_settings=Descriptor3DSettings(**descriptor_3d),
        ),
        conformers=ConformerConfig(**sections["conformers"]),
        ecfp_latent_alignment=ECFPLatentAlignmentConfig(
            **sections["ecfp_latent_alignment"]
        ),
    )
    return validate_pretrain_config(config)


def load_config(yaml_path: Optional[str] = None, **cli_overrides: Any) -> PretrainConfig:
    """Load config by merging defaults, YAML, and CLI overrides."""
    resolved = asdict(PretrainConfig())
    if yaml_path is not None:
        with open(yaml_path) as handle:
            yaml_data = yaml.safe_load(handle) or {}
        _flatten_pretrain(yaml_data)
        resolved = _deep_update(resolved, yaml_data)

    cli_clean = {key: value for key, value in cli_overrides.items() if value is not None}
    if cli_clean:
        resolved = _deep_update(resolved, cli_clean)

    return _dict_to_config(resolved)
