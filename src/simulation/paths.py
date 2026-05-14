"""Centralized filesystem paths used by the simulation package."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]

PLOTS_DIR = PROJECT_ROOT / "plots"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
TI_RESULTS_DIR = PROJECT_ROOT / "TISimResults"

NEURON_ASSETS_DIR = PACKAGE_ROOT / "neuron" / "assets"
HOC_DIR = NEURON_ASSETS_DIR / "hoc"
CELLS_DIR = NEURON_ASSETS_DIR / "cells"
MECHANISMS_DIR = NEURON_ASSETS_DIR / "mechanisms"
SAVE_STATE_DIR = CELLS_DIR / "SaveState"


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def project_path(*parts: str) -> Path:
    """Return an absolute path rooted at the project directory."""
    return PROJECT_ROOT.joinpath(*parts)
