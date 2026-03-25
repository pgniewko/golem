import os
import re
import subprocess

from setuptools import find_packages, setup


def _normalize_prerelease(ver):
    """Convert common pre-release tag formats to PEP 440."""
    ver = re.sub(r"[-.]?alpha[.-]?", "a", ver)
    ver = re.sub(r"[-.]?beta[.-]?", "b", ver)
    ver = re.sub(r"[-.]?rc[.-]?", "rc", ver)
    return ver


def _get_version_from_git():
    """Get version from git describe.

    At install time importlib.metadata can return stale data from a previous
    installation, so setup.py must resolve the version directly from git.
    """
    try:
        repo_dir = os.path.dirname(os.path.abspath(__file__))
        result = subprocess.run(
            ["git", "describe", "--tags", "--long"],
            capture_output=True,
            text=True,
            cwd=repo_dir,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr)

        desc = result.stdout.strip().lstrip("v")
        parts = desc.rsplit("-", 2)
        if len(parts) != 3 or not parts[2].startswith("g"):
            raise RuntimeError(f"Cannot parse git describe output: {desc!r}")

        ver, distance, sha = parts[0], parts[1], parts[2][1:]
        ver = _normalize_prerelease(ver)
        if int(distance) == 0:
            return ver
        return f"{ver}.dev{distance}+{sha}"
    except Exception:
        return "0+unknown"


setup(
    name="golem",
    version=_get_version_from_git(),
    description="Descriptor pretraining for Graph Transformers on molecular descriptors",
    packages=find_packages(include=["golem*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=1.13.0",
        "torch_geometric",
        "mordredcommunity",
        "rdkit",
        "gypsum-dl>=1.3.0",
        "numpy",
        "pandas",
        "pyyaml",
        "tqdm",
        "click",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
            "jupyter",
            "scikit-learn",
            "scipy",
            "matplotlib",
            "seaborn",
        ],
    },
    entry_points={"console_scripts": ["golem=golem.cli:main"]},
    include_package_data=True,
)
