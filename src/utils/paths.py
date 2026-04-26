# Purpose: Define stable project paths for data loading and tests.
# Design: Resolve paths from this file location instead of relying on the current directory.
# Workflow: Build project, data, and Titanic CSV paths as pathlib.Path constants.
#
# Michael Garcia
# michael@mandedesign.studio
# https://mandedesign.studio
#
# License: MIT

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "src" / "data"
TITANIC_DATASET = DATA_DIR / "raw" / "Titanic-Dataset.csv"
