"""Print basic environment checks for the budget-recovery workflow."""

from __future__ import annotations

import importlib.metadata as metadata
import os
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from saferai_budget_recovery import config


def main() -> None:
    print(f"Python version: {platform.python_version()}")
    for package in ("pandas", "numpy", "scipy", "pytest", "matplotlib"):
        try:
            version = metadata.version(package)
        except metadata.PackageNotFoundError:
            version = "NOT INSTALLED"
        print(f"{package}: {version}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Raw CSV exists: {config.RAW_DATA_PATH.exists()} ({config.RAW_DATA_PATH})")
    print(
        f"Processed data exists: {config.PROCESSED_DATA_PATH.exists()} "
        f"({config.PROCESSED_DATA_PATH})"
    )
    print(
        f"Fitted SOTA data exists: {config.SOTA_BETA_FITS_PATH.exists()} "
        f"({config.SOTA_BETA_FITS_PATH})"
    )


if __name__ == "__main__":
    main()
