# Purpose: Convert user passenger inputs into model features and return survival predictions.
# Design: Keep GUI input conversion separate from the trained estimator calls.
# Workflow: Encode input values into a one-row DataFrame, then pass it to the selected model.
#
# Michael Garcia
# michael@mandedesign.studio
# https://mandedesign.studio
#
# License: MIT

import pandas as pd
from src.data.load_data import FEATURES


# Purpose: Build a single passenger row that matches the training feature columns.
# Design: Reuse the fitted sex encoder so GUI input uses the same encoding as training data.
# Workflow: Convert user inputs to numeric values, assemble a DataFrame, and return it.
def build_passenger_profile(pclass, sex_text, age, sibsp, sex_encoder):
    """
    Converts GUI input into the dataframe format required by scikit-learn.
    """

    sex_number = sex_encoder.transform([sex_text.lower()])[0]

    passenger = pd.DataFrame([[
        int(pclass),
        sex_number,
        int(age),
        int(sibsp)
    ]], columns=FEATURES)

    return passenger


# Purpose: Run a fitted model against one prepared passenger profile.
# Design: Keep prediction access small so the GUI does not call estimator methods directly.
# Workflow: Call model.predict and return the first prediction value.
def predict_survival(model, passenger):
    """
    Returns the model prediction.

    1 = survived
    0 = did not survive
    """

    return model.predict(passenger)[0]
