############################
# IMPORTS
############################
from cProfile import label
import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


############################
# DATA LOADING & PREPROCESSING
############################
df = pd.read_csv('Titanic-Dataset.csv')
df['Age'] = df['Age'].fillna(df['Age'].median())

sex_encoder = LabelEncoder()
df["Sex"] = sex_encoder.fit_transform(df["Sex"])

features = ['Pclass', 'Sex', 'Age', 'SibSp']
X = df[features]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


############################
# MODEL INITIALIZATION & TRAINING
############################
decision_tree_model = DecisionTreeClassifier(max_depth=4, random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=5)

decision_tree_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)

decision_tree_accuracy = accuracy_score(y_test, decision_tree_model.predict(X_test))
knn_accuracy = accuracy_score(y_test, knn_model.predict(X_test))


############################
# PREDICTION FUNCTION
############################
def predict():
    user_sex_text = sex_entry.get().lower()
    user_sex_number = sex_encoder.transform([user_sex_text])[0]

    passenger = pd.DataFrame([[
        int(pclass_entry.get()),
        user_sex_number,
        int(age_entry.get()),
        int(sibsp_entry.get())
    ]], columns=features)

    decision_tree_result = decision_tree_model.predict(passenger)[0]
    knn_result = knn_model.predict(passenger)[0]

    if decision_tree_result == 1:
        dt_result_label.config(text="Using DT: Survived", fg="green", font=("Arial", 20, "bold"))
    else:
        dt_result_label.config(text="Usign DT: Did not survive", fg="red", font=("Arial", 20, "bold"))

    if knn_result == 1:
        knn_result_label.config(text="Using KNN: Survived", fg="green", font=("Arial", 20, "bold"))
    else:
        knn_result_label.config(text="Using KNN: Did not survive", fg="red", font=("Arial", 20, "bold"))


############################
# GUI SETUP
############################
root = tk.Tk()
root.title("Titanic Survival Prediction using ML")
root.geometry("420x480")
root.config(bg="blue")

tk.Label(root, text="Titaninc Survival Predictor", bg="blue", fg="white",
        font=("Arial", 24, "bold")).pack(pady=20)

tk.Label(root, text=f"Decision Tree Accuracy: {decision_tree_accuracy:.1%}", bg="blue", fg="white",
        font=("Arial", 14)).pack(pady=10)

tk.Label(root, text=f"KNN Accuracy: {knn_accuracy:.1%}", bg="blue", fg="white",
        font=("Arial", 14)).pack(pady=10)


############################
# INPUT FIELDS
############################
frame = tk.Frame(root, bg="blue")
frame.pack(pady=15)

def add_field(label_text, variable, options=None, default=""):
    row = tk.Frame(frame, bg="blue")
    row.pack(fill="x", pady=4)
    tk.Label(row, text=label_text, width=14, anchor='w', bg="blue", fg="white",
             font=("Arial", 12)).pack(side=tk.LEFT)
    if options:
        ttk.Combobox(row, textvariable=variable, values=options,
                     state="readonly", width=15).pack(side=tk.LEFT)
    else:
        tk.Entry(row, textvariable=variable,
                 width=18).pack(side=tk.LEFT)

pclass_entry = tk.StringVar(value="3")
sex_entry = tk.StringVar(value="male")
age_entry = tk.StringVar(value="25")
sibsp_entry = tk.StringVar(value="0")

add_field("Passenger Class (1-3)", pclass_entry, options=["1", "2", "3"])
add_field("Sex (male/female):", sex_entry, options=["male", "female"])
add_field("Age: ", age_entry)
add_field("Siblings/Spouses:", sibsp_entry, options=["0", "1", "2", "3", "4", "5", "6", "7", "8"])


############################
# ACTION BUTTONS & OUTPUT
############################
tk.Button(root, text="Predict Survival", command=predict, font=("Arial", 14, "bold"),
          fg="green", bg="white", activebackground="lightgreen", activeforeground="black").pack(pady=15)

dt_result_label = tk.Label(root, text="Enter a passenger profile and click predict.", bg="blue",
                           fg="white", font=("Arial", 14))
dt_result_label.pack(pady=10)

knn_result_label = tk.Label(root, text="Enter a passenger profile and click predict.", bg="blue",
                           fg="white", font=("Arial", 14))
knn_result_label.pack(pady=10)


############################
# FEATURE IMPORTANCE DISPLAY
############################
dt_imp = sorted(zip(features, decision_tree_model.feature_importances_), key=lambda x: -x[1])
dt_imp_text = "  |  ".join(f"{n}: {s:.0%}" for n, s in dt_imp)

tk.Label(root, text=f"Decision Tree - Top factors for survival: {dt_imp_text}", bg="blue",
        fg="white", font=("Arial", 10)).pack(pady=10)

from sklearn.inspection import permutation_importance
knn_perm_importance = permutation_importance(knn_model, X_test, y_test, n_repeats=10, random_state=42)
knn_imp = sorted(zip(features, knn_perm_importance.importances_mean), key=lambda x: -x[1])
knn_imp_text = "  |  ".join(f"{n}: {s:.0%}" for n, s in knn_imp)

tk.Label(root, text=f"KNN - Top factors for survival: {knn_imp_text}", bg="blue",
        fg="white", font=("Arial", 10)).pack(pady=10)


############################
# APPLICATION ENTRY POINT
############################
root.mainloop()