# Purpose: Train Titanic survival models and score their predictions.
# Design: Keep each model factory small and share one accuracy helper across classifiers.
# Workflow: Fit a selected classifier with training data, then evaluate it against test labels.
#
# Michael Garcia
# michael@mandedesign.studio
# https://mandedesign.studio
#
# License: MIT

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# Purpose: Build the Decision Tree classifier used by the GUI and tests.
# Design: Use a shallow, deterministic tree so results are easy to inspect and reproduce.
# Workflow: Create the classifier, fit it on training data, and return the fitted model.
def train_decision_tree(X_train, y_train):
    """
    Trains the Decision Tree model.
    """

    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    return model


# Purpose: Build the KNN classifier used as a second model comparison.
# Design: Use scikit-learn's KNeighborsClassifier with a fixed neighbor count.
# Workflow: Create the classifier, fit it on training data, and return the fitted model.
def train_knn(X_train, y_train):
    """
    Trains the KNN model.
    """

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    return model


# Purpose: Measure how well a fitted model predicts held-out labels.
# Design: Keep scoring model-agnostic by using the estimator's predict method.
# Workflow: Generate predictions for X_test and compare them to y_test with accuracy_score.
def calculate_accuracy(model, X_test, y_test):
    """
    Calculates model accuracy against test data.
    """

    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)
