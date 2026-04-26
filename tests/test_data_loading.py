# Purpose: Verify Titanic data loading and feature preparation.
# Design: Use the real dataset path and log clear shape, feature, and encoder results.
# Workflow: Run data checks through the shared logging wrapper so pytest and the log agree.
#
# Michael Garcia
# michael@mandedesign.studio
# https://mandedesign.studio
#
# License: MIT

from src.data import FEATURES, load_and_prepare_data
from src.utils import TITANIC_DATASET
from tests.log_utils import run_logged_test


# Purpose: Confirm the configured feature list matches the model input contract.
# Design: Compare FEATURES to the expected ordered column names.
# Workflow: Log expected and actual values, then assert exact equality.
def test_features_list_exists():
    # Purpose: Perform the feature-list assertion and return printable details.
    # Design: Keep the logged values close to the assertion they explain.
    # Workflow: Build result lines, assert the expected feature order, and return the lines.
    def check():
        expected = ["Pclass", "Sex", "Age", "SibSp"]
        results = [
            f"Expected features: {expected}",
            f"Actual features: {FEATURES}",
        ]

        assert FEATURES == expected
        return results

    run_logged_test(check)


# Purpose: Confirm loaded data splits are usable for training and evaluation.
# Design: Validate shapes, label alignment, missing-age handling, and encoder classes.
# Workflow: Load the dataset, collect result details, then assert each data contract.
def test_load_and_prepare_data_returns_expected_training_parts():
    # Purpose: Perform the data-loading assertions and return printable details.
    # Design: Keep dataset shape and encoder information visible in the log.
    # Workflow: Load prepared data, build result lines, assert contracts, and return details.
    def check():
        X_train, X_test, y_train, y_test, sex_encoder = load_and_prepare_data(
            TITANIC_DATASET
        )
        results = [
            f"Dataset path: {TITANIC_DATASET}",
            f"X_train shape: {X_train.shape}",
            f"X_test shape: {X_test.shape}",
            f"y_train shape: {y_train.shape}",
            f"y_test shape: {y_test.shape}",
            f"Sex classes: {list(sex_encoder.classes_)}",
        ]

        assert X_train.shape[1] == len(FEATURES)
        assert X_test.shape[1] == len(FEATURES)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        assert not X_train["Age"].isna().any()
        assert not X_test["Age"].isna().any()
        assert set(sex_encoder.classes_) == {"female", "male"}
        return results

    run_logged_test(check)
