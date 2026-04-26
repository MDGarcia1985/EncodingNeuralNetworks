# Purpose: Verify passenger profile construction and model prediction behavior.
# Design: Use prepared training data and a fitted Decision Tree for realistic prediction checks.
# Workflow: Build a passenger row, run prediction, log details, and assert expected contracts.
#
# Michael Garcia
# michael@mandedesign.studio
# https://mandedesign.studio
#
# License: MIT

from src.data import FEATURES, load_and_prepare_data
from src.model import (
    build_passenger_profile,
    predict_survival,
    train_decision_tree,
)
from src.utils import TITANIC_DATASET
from tests.log_utils import run_logged_test


# Purpose: Confirm GUI-style passenger inputs become model-ready feature rows.
# Design: Reuse the fitted sex encoder and compare the DataFrame to FEATURES.
# Workflow: Build one passenger profile, log shape and values, then assert the contract.
def test_build_passenger_profile_matches_training_features():
    # Purpose: Perform passenger profile assertions and return printable details.
    # Design: Keep row shape, columns, and values visible in the log.
    # Workflow: Load the encoder, build the row, assert values, and return details.
    def check():
        *_, sex_encoder = load_and_prepare_data(TITANIC_DATASET)
        passenger = build_passenger_profile(3, "male", 25, 0, sex_encoder)
        results = [
            f"Passenger shape: {passenger.shape}",
            f"Passenger columns: {list(passenger.columns)}",
            f"Passenger values: {passenger.to_dict('records')}",
        ]

        assert passenger.shape == (1, len(FEATURES))
        assert list(passenger.columns) == FEATURES
        assert passenger.iloc[0]["Pclass"] == 3
        assert passenger.iloc[0]["Sex"] == 1
        assert passenger.iloc[0]["Age"] == 25
        assert passenger.iloc[0]["SibSp"] == 0
        return results

    run_logged_test(check)


# Purpose: Confirm model prediction returns a binary survival value.
# Design: Use a fitted Decision Tree and one prepared passenger row.
# Workflow: Train, predict, log the result, and assert it is either 0 or 1.
def test_predict_survival_returns_binary_prediction():
    # Purpose: Perform prediction assertions and return printable details.
    # Design: Keep the predicted value and allowed values visible.
    # Workflow: Train a model, build a profile, predict, assert binary output, and return details.
    def check():
        X_train, _, y_train, _, sex_encoder = load_and_prepare_data(TITANIC_DATASET)
        model = train_decision_tree(X_train, y_train)
        passenger = build_passenger_profile(3, "male", 25, 0, sex_encoder)
        prediction = predict_survival(model, passenger)
        results = [
            f"Prediction: {prediction}",
            f"Allowed predictions: [0, 1]",
        ]

        assert prediction in [0, 1]
        return results

    run_logged_test(check)
