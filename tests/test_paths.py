# Purpose: Verify shared path constants point to real project resources.
# Design: Check pathlib.Path values without depending on the shell's current directory.
# Workflow: Log resolved paths and assert that the project, data directory, and CSV exist.
#
# Michael Garcia
# michael@mandedesign.studio
# https://mandedesign.studio
#
# License: MIT

from src.utils import DATA_DIR, PROJECT_ROOT, TITANIC_DATASET
from tests.log_utils import run_logged_test


# Purpose: Confirm project path constants resolve to existing locations.
# Design: Validate the root, data folder, and Titanic CSV name.
# Workflow: Build log lines, assert path existence and filename, then return details.
def test_project_paths_resolve_to_existing_locations():
    # Purpose: Perform path assertions and return printable details.
    # Design: Keep each resolved path visible for troubleshooting.
    # Workflow: Record paths, assert they exist, and return the recorded values.
    def check():
        results = [
            f"Project root: {PROJECT_ROOT}",
            f"Data directory: {DATA_DIR}",
            f"Titanic dataset: {TITANIC_DATASET}",
            f"Dataset exists: {TITANIC_DATASET.exists()}",
        ]

        assert PROJECT_ROOT.exists()
        assert DATA_DIR.exists()
        assert TITANIC_DATASET.exists()
        assert TITANIC_DATASET.name == "Titanic-Dataset.csv"
        return results

    run_logged_test(check)
