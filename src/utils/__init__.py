# Purpose: Expose shared filesystem paths used across the project.
# Design: Re-export path constants from paths.py for shorter imports.
# Workflow: Import from src.utils when code needs the project root or dataset path.
#
# Michael Garcia
# michael@mandedesign.studio
# https://mandedesign.studio
#
# License: MIT

"""Shared project path helpers."""

from src.utils.paths import DATA_DIR, PROJECT_ROOT, TITANIC_DATASET

__all__ = ["DATA_DIR", "PROJECT_ROOT", "TITANIC_DATASET"]
