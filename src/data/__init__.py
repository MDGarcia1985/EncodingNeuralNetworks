# Purpose: Expose the data loading helpers used by the app, model code, and tests.
# Design: Re-export the feature list and dataset preparation function from load_data.py.
# Workflow: Import from src.data when callers need prepared Titanic training data.
#
# Michael Garcia
# michael@mandedesign.studio
# https://mandedesign.studio
#
# License: MIT

"""Data loading and preprocessing helpers."""

from src.data.load_data import FEATURES, load_and_prepare_data

__all__ = ["FEATURES", "load_and_prepare_data"]
