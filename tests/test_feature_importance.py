# Purpose: Verify feature importance helpers return complete and readable results.
# Design: Train real models from the prepared dataset and inspect their importance output.
# Workflow: Compute importance values, log the output, and assert expected feature coverage.
#
# Michael Garcia
# michael@mandedesign.studio
# https://mandedesign.studio
#
# License: MIT

from src.data import FEATURES, load_and_prepare_data
from src.model import (
    decision_tree_importance,
    format_importance_text,
    knn_importance,
    train_decision_tree,
    train_knn,
)
from src.utils import TITANIC_DATASET
from tests.log_utils import run_logged_test


# Purpose: Confirm Decision Tree importance includes every configured feature.
# Design: Use the model's built-in importance values from a fitted tree.
# Workflow: Train the tree, compute sorted importance, log it, and assert coverage.
def test_decision_tree_importance_returns_all_features():
    # Purpose: Perform Decision Tree importance assertions and return printable details.
    # Design: Keep count and raw importance tuples visible for debugging.
    # Workflow: Load data, train, calculate importance, assert completeness, and return details.
    def check():
        X_train, X_test, y_train, y_test, _ = load_and_prepare_data(TITANIC_DATASET)
        model = train_decision_tree(X_train, y_train)
        importance = decision_tree_importance(FEATURES, model)
        results = [
            f"Feature count: {len(FEATURES)}",
            f"Importance count: {len(importance)}",
            f"Importance values: {importance}",
        ]

        assert len(importance) == len(FEATURES)
        assert {name for name, _ in importance} == set(FEATURES)
        assert all(score >= 0 for _, score in importance)
        return results

    run_logged_test(check)


# Purpose: Confirm KNN permutation importance and formatting produce usable output.
# Design: Use permutation scores for KNN and the shared text formatter.
# Workflow: Train KNN, calculate importance, format it, log both forms, and assert coverage.
def test_knn_importance_and_formatter_return_readable_results():
    # Purpose: Perform KNN importance assertions and return printable details.
    # Design: Check both raw importance tuples and the GUI-facing formatted string.
    # Workflow: Load data, train, calculate, format, assert completeness, and return details.
    def check():
        X_train, X_test, y_train, y_test, _ = load_and_prepare_data(TITANIC_DATASET)
        model = train_knn(X_train, y_train)
        importance = knn_importance(FEATURES, model, X_test, y_test)
        formatted = format_importance_text(importance)
        results = [
            f"Importance count: {len(importance)}",
            f"Importance values: {importance}",
            f"Formatted importance: {formatted}",
        ]

        assert len(importance) == len(FEATURES)
        assert {name for name, _ in importance} == set(FEATURES)
        assert all(name in formatted for name in FEATURES)
        return results

    run_logged_test(check)
