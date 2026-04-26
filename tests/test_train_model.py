# Purpose: Verify model training helpers fit estimators and produce valid accuracy scores.
# Design: Use the prepared Titanic train/test split for both Decision Tree and KNN.
# Workflow: Train each model, score it, log the result, and assert prediction counts.
#
# Michael Garcia
# michael@mandedesign.studio
# https://mandedesign.studio
#
# License: MIT

from src.data import load_and_prepare_data
from src.model import calculate_accuracy, train_decision_tree, train_knn
from src.utils import TITANIC_DATASET
from tests.log_utils import run_logged_test


# Purpose: Confirm the Decision Tree trainer returns a fitted, scorable model.
# Design: Train on the prepared split and evaluate against held-out test labels.
# Workflow: Fit, score, predict, log details, and assert score range and count.
def test_train_decision_tree_fits_and_scores():
    # Purpose: Perform Decision Tree training assertions and return printable details.
    # Design: Keep model type, accuracy, and prediction count visible in the log.
    # Workflow: Load data, train, score, assert output contracts, and return details.
    def check():
        X_train, X_test, y_train, y_test, _ = load_and_prepare_data(TITANIC_DATASET)
        model = train_decision_tree(X_train, y_train)
        accuracy = calculate_accuracy(model, X_test, y_test)
        predictions = model.predict(X_test)
        results = [
            f"Decision Tree model: {type(model).__name__}",
            f"Accuracy: {accuracy:.4f}",
            f"Prediction count: {len(predictions)}",
        ]

        assert 0 <= accuracy <= 1
        assert len(predictions) == len(y_test)
        return results

    run_logged_test(check)


# Purpose: Confirm the KNN trainer returns a fitted, scorable model.
# Design: Train on the prepared split and evaluate against held-out test labels.
# Workflow: Fit, score, predict, log details, and assert score range and count.
def test_train_knn_fits_and_scores():
    # Purpose: Perform KNN training assertions and return printable details.
    # Design: Keep model type, accuracy, and prediction count visible in the log.
    # Workflow: Load data, train, score, assert output contracts, and return details.
    def check():
        X_train, X_test, y_train, y_test, _ = load_and_prepare_data(TITANIC_DATASET)
        model = train_knn(X_train, y_train)
        accuracy = calculate_accuracy(model, X_test, y_test)
        predictions = model.predict(X_test)
        results = [
            f"KNN model: {type(model).__name__}",
            f"Accuracy: {accuracy:.4f}",
            f"Prediction count: {len(predictions)}",
        ]

        assert 0 <= accuracy <= 1
        assert len(predictions) == len(y_test)
        return results

    run_logged_test(check)
