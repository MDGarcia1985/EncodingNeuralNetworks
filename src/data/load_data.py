# Purpose: Load the Titanic CSV and prepare model-ready training and test data.
# Design: Keep data cleaning, feature selection, label encoding, and splitting in one helper.
# Workflow: Read the CSV, fill missing ages, encode sex labels, split data, and return the pieces.
#
# Michael Garcia
# michael@mandedesign.studio
# https://mandedesign.studio
#
# License: MIT

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


FEATURES = ["Pclass", "Sex", "Age", "SibSp"]


def load_and_prepare_data(csv_path):
    """
    Loads the Titanic dataset and prepares it for training.

    Note:
    This file owns the data work. Do not put GUI code or model training code here.
    """

    df = pd.read_csv(csv_path)

    df["Age"] = df["Age"].fillna(df["Age"].median())

    sex_encoder = LabelEncoder()
    df["Sex"] = sex_encoder.fit_transform(df["Sex"])

    X = df[FEATURES]
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test, sex_encoder
