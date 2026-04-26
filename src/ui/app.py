# Purpose: Provide the Tkinter interface for training models and predicting survival.
# Design: Build a compact GUI around the data loader, model helpers, and feature summaries.
# Workflow: Load data, train models, render inputs, and update prediction labels on button click.
#
# Michael Garcia
# michael@mandedesign.studio
# https://mandedesign.studio
#
# License: MIT

import tkinter as tk
from tkinter import ttk

from src.data.load_data import load_and_prepare_data, FEATURES
from src.model.train_model import (
    train_decision_tree,
    train_knn,
    calculate_accuracy
)
from src.model.predict import (
    build_passenger_profile,
    predict_survival
)
from src.model.feature_importance import (
    decision_tree_importance,
    knn_importance,
    format_importance_text
)
from src.utils.paths import TITANIC_DATASET


X_train, X_test, y_train, y_test, sex_encoder = load_and_prepare_data(TITANIC_DATASET)

decision_tree_model = train_decision_tree(X_train, y_train)
knn_model = train_knn(X_train, y_train)

decision_tree_accuracy = calculate_accuracy(decision_tree_model, X_test, y_test)
knn_accuracy = calculate_accuracy(knn_model, X_test, y_test)


# Purpose: Handle the Predict button click and display both model outcomes.
# Design: Read current widget values, build one passenger profile, and query each model.
# Workflow: Convert inputs, predict with Decision Tree and KNN, then update result labels.
def predict():
    passenger = build_passenger_profile(
        pclass_entry.get(),
        sex_entry.get(),
        age_entry.get(),
        sibsp_entry.get(),
        sex_encoder
    )

    decision_tree_result = predict_survival(decision_tree_model, passenger)
    knn_result = predict_survival(knn_model, passenger)

    if decision_tree_result == 1:
        dt_result_label.config(
            text="Using DT: Survived",
            fg="green",
            font=("Arial", 20, "bold")
        )
    else:
        dt_result_label.config(
            text="Using DT: Did not survive",
            fg="red",
            font=("Arial", 20, "bold")
        )

    if knn_result == 1:
        knn_result_label.config(
            text="Using KNN: Survived",
            fg="green",
            font=("Arial", 20, "bold")
        )
    else:
        knn_result_label.config(
            text="Using KNN: Did not survive",
            fg="red",
            font=("Arial", 20, "bold")
        )


# Purpose: Add one labeled input row to the GUI form.
# Design: Use a read-only combobox when choices are fixed and an entry when free typing is needed.
# Workflow: Create a row frame, add the label, then attach the appropriate input widget.
def add_field(label_text, variable, options=None):
    row = tk.Frame(frame, bg="blue")
    row.pack(fill="x", pady=4)

    tk.Label(
        row,
        text=label_text,
        width=14,
        anchor="w",
        bg="blue",
        fg="white",
        font=("Arial", 12)
    ).pack(side=tk.LEFT)

    if options:
        ttk.Combobox(
            row,
            textvariable=variable,
            values=options,
            state="readonly",
            width=15
        ).pack(side=tk.LEFT)
    else:
        tk.Entry(
            row,
            textvariable=variable,
            width=18
        ).pack(side=tk.LEFT)


root = tk.Tk()
root.title("Titanic Survival Prediction using ML")
root.geometry("420x480")
root.config(bg="blue")

tk.Label(
    root,
    text="Titanic Survival Predictor",
    bg="blue",
    fg="white",
    font=("Arial", 24, "bold")
).pack(pady=20)

tk.Label(
    root,
    text=f"Decision Tree Accuracy: {decision_tree_accuracy:.1%}",
    bg="blue",
    fg="white",
    font=("Arial", 14)
).pack(pady=10)

tk.Label(
    root,
    text=f"KNN Accuracy: {knn_accuracy:.1%}",
    bg="blue",
    fg="white",
    font=("Arial", 14)
).pack(pady=10)

frame = tk.Frame(root, bg="blue")
frame.pack(pady=15)

pclass_entry = tk.StringVar(value="3")
sex_entry = tk.StringVar(value="male")
age_entry = tk.StringVar(value="25")
sibsp_entry = tk.StringVar(value="0")

add_field("Passenger Class (1-3)", pclass_entry, options=["1", "2", "3"])
add_field("Sex (male/female):", sex_entry, options=["male", "female"])
add_field("Age:", age_entry)
add_field(
    "Siblings/Spouses:",
    sibsp_entry,
    options=["0", "1", "2", "3", "4", "5", "6", "7", "8"]
)

tk.Button(
    root,
    text="Predict Survival",
    command=predict,
    font=("Arial", 14, "bold"),
    fg="green",
    bg="white",
    activebackground="lightgreen",
    activeforeground="black"
).pack(pady=15)

dt_result_label = tk.Label(
    root,
    text="Enter a passenger profile and click predict.",
    bg="blue",
    fg="white",
    font=("Arial", 14)
)
dt_result_label.pack(pady=10)

knn_result_label = tk.Label(
    root,
    text="Enter a passenger profile and click predict.",
    bg="blue",
    fg="white",
    font=("Arial", 14)
)
knn_result_label.pack(pady=10)

dt_imp = decision_tree_importance(FEATURES, decision_tree_model)
dt_imp_text = format_importance_text(dt_imp)

tk.Label(
    root,
    text=f"Decision Tree - Top factors for survival: {dt_imp_text}",
    bg="blue",
    fg="white",
    font=("Arial", 10)
).pack(pady=10)

knn_imp = knn_importance(FEATURES, knn_model, X_test, y_test)
knn_imp_text = format_importance_text(knn_imp)

tk.Label(
    root,
    text=f"KNN - Top factors for survival: {knn_imp_text}",
    bg="blue",
    fg="white",
    font=("Arial", 10)
).pack(pady=10)

if __name__ == "__main__":
    root.mainloop()
