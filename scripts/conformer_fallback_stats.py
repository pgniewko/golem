#!/usr/bin/env python3
"""Report MMFF-to-UFF fallback statistics for Golem conformer generation."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from tqdm import tqdm

import golem.conformers as conformer_module
from golem.config import load_config
from golem.conformers import retain_boltzmann_conformers
from golem.pretrain import _prepare_split_smiles
from golem.utils import load_smiles

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptimizationCall:
    method: str
    embedded_conformers: int
    usable_conformers: int


@dataclass
class MoleculeStats:
    index: int
    smiles: str
    status: str
    embedded_conformers: int
    mmff_usable_conformers: int
    uff_usable_conformers: int
    failed_optimization_conformers: int
    optimized_conformers: int
    retained_conformers: int
    min_energy: float | None
    max_delta_energy: float | None
    optimization_calls: list[OptimizationCall]


@dataclass
class _OptimizationTrace:
    calls: list[OptimizationCall]

    def reset(self) -> None:
        self.calls.clear()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Golem's production conformer generator on a SMILES set and "
            "summarize MMFF-to-UFF fallback usage."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/golem-2d-3d-plus-latent-pxr-v25-2000.yaml",
        help="Path to the Golem pretraining config.",
    )
    parser.add_argument(
        "--smiles",
        default="data/openadmet/pxr/train_test_smiles.smi",
        help="Path to the input SMILES file.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path for a JSON report.",
    )
    parser.add_argument(
        "--output-failures-csv",
        default=None,
        help="Optional path for a CSV containing molecules with zero retained conformers.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only check the first N post-split/post-isoform SMILES.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO logging from this script and Golem helpers.",
    )
    return parser.parse_args(argv)


def _count_input_smiles_lines(path: str) -> int:
    count = 0
    with open(path) as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                count += 1
    return count


def _apply_pretrain_subsample(smiles_list: list[str], config) -> list[str]:
    if config.subsample is None or not 0 < config.subsample < 1:
        return smiles_list

    rng = np.random.RandomState(config.seed)
    n_subsampled = max(1, int(len(smiles_list) * config.subsample))
    indices = rng.choice(len(smiles_list), size=n_subsampled, replace=False)
    return [smiles_list[index] for index in sorted(indices)]


@contextmanager
def _trace_optimizer_calls(trace: _OptimizationTrace) -> Iterator[None]:
    original_optimize = conformer_module._optimize_conformers

    def traced_optimize(mol, method: str, *, max_iters: int = 1000):
        embedded_conformers = int(mol.GetNumConformers())
        energies = original_optimize(mol, method, max_iters=max_iters)
        trace.calls.append(
            OptimizationCall(
                method=method,
                embedded_conformers=embedded_conformers,
                usable_conformers=0 if energies is None else len(energies),
            )
        )
        return energies

    conformer_module._optimize_conformers = traced_optimize
    try:
        yield
    finally:
        conformer_module._optimize_conformers = original_optimize


def _classify(calls: list[OptimizationCall], pool) -> str:
    if pool is None:
        if any(call.method == "UFF" for call in calls):
            return "uff_fallback_failure"
        return "conformer_failure"
    if any(call.method == "UFF" for call in calls):
        return "uff_fallback_success"
    if any(call.method == "MMFF" and call.usable_conformers > 0 for call in calls):
        return "mmff_success"
    return "unexpected"


def _summarize_molecule(index: int, smiles: str, config, trace: _OptimizationTrace) -> MoleculeStats:
    trace.reset()
    pool = conformer_module.generate_optimized_conformer_pool(
        smiles,
        config.conformers,
        seed=config.seed,
    )
    retained_conformers = []
    if pool is not None:
        retained_conformers = retain_boltzmann_conformers(pool, config.conformers).conformers

    calls = list(trace.calls)
    mmff_usable = next(
        (call.usable_conformers for call in calls if call.method == "MMFF"),
        0,
    )
    uff_usable = next(
        (call.usable_conformers for call in calls if call.method == "UFF"),
        0,
    )
    embedded_conformers = max(
        (call.embedded_conformers for call in calls),
        default=0 if pool is None else len(pool.conformers),
    )
    best_usable = max(
        (call.usable_conformers for call in calls),
        default=0 if pool is None else len(pool.conformers),
    )

    return MoleculeStats(
        index=index,
        smiles=smiles,
        status=_classify(calls, pool),
        embedded_conformers=embedded_conformers,
        mmff_usable_conformers=mmff_usable,
        uff_usable_conformers=uff_usable,
        failed_optimization_conformers=max(embedded_conformers - best_usable, 0),
        optimized_conformers=0 if pool is None else len(pool.conformers),
        retained_conformers=len(retained_conformers),
        min_energy=None if pool is None or not pool.conformers else pool.conformers[0].energy,
        max_delta_energy=(
            None
            if not retained_conformers
            else max(conformer.delta_energy for conformer in retained_conformers)
        ),
        optimization_calls=calls,
    )


def _distribution(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "mean": None, "median": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": int(np.min(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "max": int(np.max(array)),
    }


def _as_jsonable_record(record: MoleculeStats) -> dict[str, object]:
    payload = asdict(record)
    payload["optimization_calls"] = [asdict(call) for call in record.optimization_calls]
    return payload


def _print_summary(summary: dict[str, object], records: list[MoleculeStats]) -> None:
    status_counts = summary["status_counts"]
    assert isinstance(status_counts, dict)
    retained = summary["retained_conformer_count_distribution"]
    optimized = summary["optimized_conformer_count_distribution"]

    print("Conformer fallback statistics")
    print(f"  Config: {summary['config_path']}")
    print(f"  SMILES: {summary['smiles_path']}")
    print(f"  Raw non-comment input lines: {summary['raw_input_lines']}")
    print(f"  Loaded unique SMILES: {summary['loaded_unique_smiles']}")
    print(f"  Checked post-split/post-isoform SMILES: {summary['checked_smiles']}")
    print("")
    print("Status counts")
    for key in (
        "mmff_success",
        "uff_fallback_success",
        "uff_fallback_failure",
        "conformer_failure",
        "unexpected",
    ):
        print(f"  {key}: {status_counts.get(key, 0)}")
    print("")
    print(
        "Optimized conformers per successful molecule: "
        f"min={optimized['min']} mean={optimized['mean']} "
        f"median={optimized['median']} max={optimized['max']}"
    )
    print(
        "Retained Boltzmann conformers per successful molecule: "
        f"min={retained['min']} mean={retained['mean']} "
        f"median={retained['median']} max={retained['max']}"
    )
    print("")
    print(
        "Compounds with zero optimized conformers: "
        f"{summary['zero_optimized_conformer_compounds']}"
    )
    print(
        "Compounds with zero retained Boltzmann conformers: "
        f"{summary['zero_retained_conformer_compounds']}"
    )
    print(
        "Embedded conformers with failed optimization: "
        f"{summary['failed_optimization_conformers']}"
    )

    interesting = [
        record
        for record in records
        if record.status != "mmff_success"
    ][:20]
    if interesting:
        print("")
        print("First fallback/failure examples")
        for record in interesting:
            print(
                f"  #{record.index} {record.status}: "
                f"embedded={record.embedded_conformers} "
                f"MMFF={record.mmff_usable_conformers} "
                f"UFF={record.uff_usable_conformers} {record.smiles}"
            )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1 when provided.")

    config_path = Path(args.config)
    smiles_path = Path(args.smiles)
    config = load_config(str(config_path))
    raw_input_lines = _count_input_smiles_lines(str(smiles_path))
    loaded_smiles = load_smiles(str(smiles_path))
    sampled_smiles = _apply_pretrain_subsample(loaded_smiles, config)
    prepared_smiles, _ = _prepare_split_smiles(sampled_smiles, config)
    if args.limit is not None:
        prepared_smiles = prepared_smiles[: args.limit]

    trace = _OptimizationTrace(calls=[])
    records: list[MoleculeStats] = []
    with _trace_optimizer_calls(trace):
        for index, smiles in enumerate(
            tqdm(prepared_smiles, desc="Checking conformers", unit="mol")
        ):
            records.append(_summarize_molecule(index, smiles, config, trace))

    status_counts = Counter(record.status for record in records)
    successful_records = [record for record in records if record.optimized_conformers > 0]
    optimized_counts = Counter(record.optimized_conformers for record in records)
    retained_counts = Counter(record.retained_conformers for record in records)
    zero_optimized_records = [
        record for record in records if record.optimized_conformers == 0
    ]
    zero_retained_records = [
        record for record in records if record.retained_conformers == 0
    ]
    summary: dict[str, object] = {
        "config_path": str(config_path),
        "smiles_path": str(smiles_path),
        "seed": config.seed,
        "conformer_settings": asdict(config.conformers),
        "raw_input_lines": raw_input_lines,
        "loaded_unique_smiles": len(loaded_smiles),
        "post_subsample_smiles": len(sampled_smiles),
        "checked_smiles": len(prepared_smiles),
        "status_counts": dict(status_counts),
        "optimized_conformer_count_counts": dict(sorted(optimized_counts.items())),
        "retained_conformer_count_counts": dict(sorted(retained_counts.items())),
        "zero_optimized_conformer_compounds": len(zero_optimized_records),
        "zero_retained_conformer_compounds": len(zero_retained_records),
        "failed_optimization_conformers": sum(
            record.failed_optimization_conformers for record in records
        ),
        "optimized_conformer_count_distribution": _distribution(
            [record.optimized_conformers for record in successful_records]
        ),
        "retained_conformer_count_distribution": _distribution(
            [record.retained_conformers for record in successful_records]
        ),
    }

    fallback_or_failure_records = [
        _as_jsonable_record(record)
        for record in records
        if record.status != "mmff_success"
    ]
    report = {
        "summary": summary,
        "zero_optimized_conformer_records": [
            _as_jsonable_record(record) for record in zero_optimized_records
        ],
        "zero_retained_conformer_records": [
            _as_jsonable_record(record) for record in zero_retained_records
        ],
        "fallback_or_failure_records": fallback_or_failure_records,
    }

    _print_summary(summary, records)

    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        print("")
        print(f"Wrote JSON report to {output_path}")

    if args.output_failures_csv is not None:
        output_path = Path(args.output_failures_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "index",
                    "smiles",
                    "status",
                    "embedded_conformers",
                    "mmff_usable_conformers",
                    "uff_usable_conformers",
                    "optimized_conformers",
                    "retained_conformers",
                ],
            )
            writer.writeheader()
            for record in zero_retained_records:
                writer.writerow(
                    {
                        "index": record.index,
                        "smiles": record.smiles,
                        "status": record.status,
                        "embedded_conformers": record.embedded_conformers,
                        "mmff_usable_conformers": record.mmff_usable_conformers,
                        "uff_usable_conformers": record.uff_usable_conformers,
                        "optimized_conformers": record.optimized_conformers,
                        "retained_conformers": record.retained_conformers,
                    }
                )
        print(f"Wrote zero-retained conformer CSV to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
