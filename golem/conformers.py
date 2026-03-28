"""RDKit conformer generation utilities for offline 3D targets."""

from __future__ import annotations

import hashlib
import importlib
import logging
import multiprocessing as mp
import time
from dataclasses import dataclass
from queue import Empty

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

from golem.config import ConformerConfig

logger = logging.getLogger(__name__)
_MP_CONTEXT = mp.get_context(
    "fork" if "fork" in mp.get_all_start_methods() else "spawn"
)


@dataclass
class ConformerEnsemble:
    """Retained conformers and their relative energies."""

    mol: Chem.Mol
    conformer_ids: list[int]
    relative_energies_kcal: np.ndarray


def _molecule_seed(smiles: str, seed: int) -> int:
    digest = hashlib.sha256(smiles.encode()).digest()
    return (seed ^ int.from_bytes(digest[:4], byteorder="little")) & 0x7FFFFFFF


def _optimize_conformers(mol: Chem.Mol, method: str) -> dict[int, float] | None:
    if method == "MMFF":
        if not AllChem.MMFFHasAllMoleculeParams(mol):
            return None
        runner = AllChem.MMFFOptimizeMoleculeConfs
    elif method == "UFF":
        runner = AllChem.UFFOptimizeMoleculeConfs
    else:
        raise ValueError(f"Unsupported optimization method: {method!r}")

    try:
        results = runner(mol, numThreads=0)
    except Exception:
        logger.debug("Conformer optimization failed", exc_info=True)
        return None

    energies = {
        conformer.GetId(): float(energy)
        for conformer, (_, energy) in zip(mol.GetConformers(), results, strict=False)
        if np.isfinite(energy)
    }
    return energies or None


def _timed_out(started_at: float, timeout_seconds: int) -> bool:
    return timeout_seconds > 0 and (time.monotonic() - started_at) >= timeout_seconds


def _is_timeout_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "timed out" in message or "timeout" in message


def _call_by_name(
    module_name: str,
    qualname: str,
    args: tuple,
    kwargs: dict,
):
    target = importlib.import_module(module_name)
    for part in qualname.split("."):
        target = getattr(target, part)
    return target(*args, **kwargs)


def _timeout_worker(
    result_queue,
    module_name: str,
    qualname: str,
    args: tuple,
    kwargs: dict,
) -> None:
    try:
        result_queue.put(("ok", _call_by_name(module_name, qualname, args, kwargs)))
    except BaseException as exc:  # pragma: no cover - defensive process boundary
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_with_timeout(
    module_name: str,
    qualname: str,
    *,
    args: tuple = (),
    kwargs: dict | None = None,
    timeout_seconds: float,
):
    kwargs = kwargs or {}
    if timeout_seconds <= 0:
        return _call_by_name(module_name, qualname, args, kwargs)

    result_queue = _MP_CONTEXT.Queue(maxsize=1)
    process = _MP_CONTEXT.Process(
        target=_timeout_worker,
        args=(result_queue, module_name, qualname, args, kwargs),
    )
    process.daemon = True
    process.start()
    process.join(timeout_seconds)

    try:
        if process.is_alive():
            process.terminate()
            process.join(1.0)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(1.0)
            raise TimeoutError(f"{module_name}.{qualname} exceeded {timeout_seconds}s")

        try:
            status, payload = result_queue.get(timeout=1.0)
        except Empty as exc:
            raise RuntimeError(
                f"{module_name}.{qualname} exited without returning a result"
            ) from exc
    finally:
        result_queue.close()
        result_queue.join_thread()

    if status == "error":
        raise RuntimeError(payload)
    return payload


def _serialise_ensemble(
    ensemble: ConformerEnsemble | None,
) -> tuple[bytes, list[int], list[float]] | None:
    if ensemble is None:
        return None
    return (
        ensemble.mol.ToBinary(),
        list(ensemble.conformer_ids),
        ensemble.relative_energies_kcal.tolist(),
    )


def _deserialise_ensemble(
    payload: tuple[bytes, list[int], list[float]] | None,
) -> ConformerEnsemble | None:
    if payload is None:
        return None
    mol_binary, conformer_ids, relative_energies_kcal = payload
    return ConformerEnsemble(
        mol=Chem.Mol(mol_binary),
        conformer_ids=list(conformer_ids),
        relative_energies_kcal=np.asarray(relative_energies_kcal, dtype=np.float64),
    )


def _generate_conformer_ensemble_inline(
    smiles: str,
    config: ConformerConfig,
    *,
    seed: int,
) -> tuple[ConformerEnsemble | None, str | None]:
    """Generate a low-energy, RMS-pruned conformer ensemble."""
    started_at = time.monotonic()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "invalid_smiles"

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = _molecule_seed(smiles, seed)
    params.pruneRmsThresh = -1.0
    params.timeout = max(int(config.timeout_seconds), 0)

    try:
        conf_ids = list(
            AllChem.EmbedMultipleConfs(mol, numConfs=config.n_generate, params=params)
        )
    except Exception as exc:
        logger.debug("Conformer embedding failed for %s", smiles, exc_info=True)
        return None, "timeout" if _is_timeout_error(exc) else "embedding_failed"
    if not conf_ids:
        if _timed_out(started_at, config.timeout_seconds):
            return None, "timeout"
        return None, "no_conformers"

    energies = _optimize_conformers(mol, "MMFF")
    if energies is None:
        energies = _optimize_conformers(mol, "UFF")
    if energies is None:
        if _timed_out(started_at, config.timeout_seconds):
            return None, "timeout"
        return None, "optimization_failed"

    min_energy = min(energies.values())
    candidates = [
        (conf_id, energy - min_energy)
        for conf_id, energy in sorted(energies.items(), key=lambda item: item[1])
        if energy - min_energy <= config.energy_window_kcal
    ]

    kept: list[tuple[int, float]] = []
    for conf_id, rel_energy in candidates:
        is_duplicate = False
        for kept_id, _ in kept:
            try:
                if rdMolAlign.GetBestRMS(mol, mol, prbId=conf_id, refId=kept_id) < config.prune_rms:
                    is_duplicate = True
                    break
            except Exception:
                logger.debug("RMS comparison failed for %s", smiles, exc_info=True)
                is_duplicate = True
                break
        if is_duplicate:
            continue
        kept.append((conf_id, rel_energy))
        if len(kept) >= config.n_keep:
            break

    if not kept:
        if _timed_out(started_at, config.timeout_seconds):
            return None, "timeout"
        return None, "rms_pruned"

    return (
        ConformerEnsemble(
            mol=mol,
            conformer_ids=[conf_id for conf_id, _ in kept],
            relative_energies_kcal=np.array(
                [rel_energy for _, rel_energy in kept],
                dtype=np.float64,
            ),
        ),
        None,
    )


def _generate_conformer_ensemble_payload(
    smiles: str,
    config: ConformerConfig,
    seed: int,
) -> tuple[tuple[bytes, list[int], list[float]] | None, str | None]:
    ensemble, failure_reason = _generate_conformer_ensemble_inline(
        smiles,
        config,
        seed=seed,
    )
    return _serialise_ensemble(ensemble), failure_reason


def generate_conformer_ensemble(
    smiles: str,
    config: ConformerConfig,
    *,
    seed: int,
) -> tuple[ConformerEnsemble | None, str | None]:
    """Generate a low-energy, RMS-pruned conformer ensemble."""
    timeout_seconds = float(max(config.timeout_seconds, 0))
    if timeout_seconds <= 0:
        return _generate_conformer_ensemble_inline(smiles, config, seed=seed)

    try:
        payload, failure_reason = _run_with_timeout(
            __name__,
            "_generate_conformer_ensemble_payload",
            args=(smiles, config, seed),
            timeout_seconds=timeout_seconds,
        )
    except TimeoutError:
        logger.debug(
            "Conformer generation exceeded %.1fs for %s",
            timeout_seconds,
            smiles,
        )
        return None, "timeout"
    except Exception:
        logger.debug("Conformer generation worker failed for %s", smiles, exc_info=True)
        return None, "worker_failed"

    return _deserialise_ensemble(payload), failure_reason
