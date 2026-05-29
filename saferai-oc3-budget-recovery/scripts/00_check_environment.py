"""Print basic environment checks for the data-cleaning workflow."""

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
    plan_path = PROJECT_ROOT.parent / "saferai_mixture_fragility_allocation_plan_v8.md"
    print(f"v8 plan exists: {plan_path.exists()} ({plan_path})")


if __name__ == "__main__":
    main()

