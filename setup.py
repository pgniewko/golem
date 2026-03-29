from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from setuptools import find_packages, setup


def _load_version_utils():
    version_utils_path = Path(__file__).resolve().parent / "golem" / "_version_utils.py"
    spec = spec_from_file_location("golem_setup_version_utils", version_utils_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load version helpers from {version_utils_path}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_version_utils = _load_version_utils()

try:
    PACKAGE_VERSION = _version_utils._get_version_from_git()
except Exception:
    PACKAGE_VERSION = "0+unknown"


setup(
    name="golem",
    version=PACKAGE_VERSION,
    description="Descriptor pretraining for Graph Transformers on molecular descriptors",
    packages=find_packages(exclude=["tests*", "*.tests", "*.tests.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=1.13.0",
        "torch_geometric",
        "mordredcommunity",
        "molfeat",
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
    entry_points={
        "console_scripts": [
            "golem=golem.cli:main",
        ],
    },
    include_package_data=True,
)
