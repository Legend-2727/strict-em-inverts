"""Minimal install-script for the strict-em-inverts package.

Run `pip install -e .` from the repo root to make the analysis scripts
importable as `from strict_em_inverts.<module> import ...` (the top-level
`src/` directory is mapped to the package name).
"""
from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent

setup(
    name="strict-em-inverts",
    version="0.1.0",
    description=(
        "Code + evaluation tuples for the EMNLP 2026 paper "
        "'Strict-EM Inverts Cross-Lingual VLM Rankings: An "
        "Architecture-Specific Measurement Artifact'."
    ),
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Anonymous",
    license="MIT",
    package_dir={"strict_em_inverts": "src"},
    packages=["strict_em_inverts"] + [
        f"strict_em_inverts.{p}" for p in find_packages("src")
    ],
    python_requires=">=3.10",
    install_requires=(ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines(),
)
