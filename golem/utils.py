"""Utility functions: seeding, data splitting, DataLoader creation, SMILES loading."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Callable, Iterator, List, Sized, Tuple

import numpy as np
import torch
from torch.utils.data import Sampler
from torch_geometric.loader import DataLoader


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_data(
    n: int, fractions: List[float], seed: int = 42
) -> Tuple[np.ndarray, ...]:
    """Random split into len(fractions) subsets. Returns index arrays.

    Args:
        n: Total number of samples.
        fractions: List of fractions (must sum to 1.0). Supports 2 or 3 splits.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of index arrays, one per split.
    """
    assert len(fractions) in (2, 3), "Only 2 or 3 splits supported"
    assert abs(sum(fractions) - 1.0) < 1e-6, f"Fractions must sum to 1.0, got {sum(fractions)}"

    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)

    splits = []
    start = 0
    for i, frac in enumerate(fractions):
        if i == len(fractions) - 1:
            # Last split gets the remainder to avoid rounding issues
            splits.append(indices[start:])
        else:
            end = start + int(n * frac)
            splits.append(indices[start:end])
            start = end

    return tuple(splits)


class EpochSeededRandomSampler(Sampler[int]):
    """Match DataLoader shuffle order from the legacy global-RNG path.

    Each iterator draw consumes one int64 from ``seed_source`` to seed a fresh
    epoch-local generator, mirroring PyTorch's ``RandomSampler(generator=None)``
    behavior without depending on the process-global RNG state.
    """

    def __init__(self, data_source: Sized, seed_source: torch.Generator):
        self.data_source = data_source
        self.seed_source = seed_source

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if n <= 0:
            return iter(())

        seed = int(
            torch.empty((), dtype=torch.int64).random_(generator=self.seed_source).item()
        )
        generator = torch.Generator()
        generator.manual_seed(seed)
        return iter(torch.randperm(n, generator=generator).tolist())

    def __len__(self) -> int:
        return len(self.data_source)


def make_loader(
    dataset: list,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    sampler: Sampler | None = None,
    generator: torch.Generator | None = None,
    worker_init_fn: Callable[[int], None] | None = None,
) -> DataLoader:
    """Create a PyG DataLoader.

    Args:
        dataset: List of PyG Data objects.
        batch_size: Batch size.
        shuffle: Whether to shuffle.
        num_workers: Number of data loading workers.
        sampler: Optional explicit sampler. When provided, ``shuffle`` is ignored.
        generator: Optional generator for DataLoader worker seeding.
        worker_init_fn: Optional worker initialiser.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )


def seed_loader_worker(worker_id: int) -> None:
    """Seed Python, NumPy, and Torch RNGs inside a DataLoader worker."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def load_smiles(path: str) -> List[str]:
    """Load SMILES from file. Auto-detects format by extension.

    - .smi: one SMILES per line (ignores lines starting with # and empty lines)
    - .csv: reads 'SMILES' column from CSV

    Args:
        path: Path to SMILES file.

    Returns:
        List of SMILES strings.

    Raises:
        ValueError: For unsupported file extensions.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".smi":
        smiles = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    # Take first whitespace-delimited token (SMILES may be followed by name)
                    smiles.append(line.split()[0])
        return smiles

    elif ext == ".csv":
        import pandas as pd

        df = pd.read_csv(path)
        if "SMILES" not in df.columns:
            raise ValueError(
                f"CSV file {path} does not have a 'SMILES' column. "
                f"Available columns: {list(df.columns)}"
            )
        return df["SMILES"].dropna().tolist()

    else:
        raise ValueError(
            f"Unsupported file extension '{ext}'. Use .smi or .csv"
        )
