# Purpose: Summarize which passenger features most influence each survival model.
# Design: Use tree-native importance for Decision Trees and permutation importance for KNN.
# Workflow: Compute feature scores, sort them by impact, and format them for display.
#
# Michael Garcia
# michael@mandedesign.studio
# https://mandedesign.studio
#
# License: MIT

from sklearn.inspection import permutation_importance


# Purpose: Report feature importance scores from a fitted Decision Tree model.
# Design: Pair each configured feature name with the model's built-in importance value.
# Workflow: Zip names to scores, sort highest first, and return the ordered list.
def decision_tree_importance(features, decision_tree_model):
    """
    Gets feature importance from the Decision Tree model.
    """

    return sorted(
        zip(features, decision_tree_model.feature_importances_),
        key=lambda x: -x[1]
    )


# Purpose: Estimate feature importance for a fitted KNN model.
# Design: Use permutation importance because KNN does not expose built-in feature weights.
# Workflow: Shuffle each feature repeatedly, measure accuracy impact, and return sorted scores.
def knn_importance(features, knn_model, X_test, y_test):
    """
    KNN does not have built-in feature importance.

    This uses permutation importance, which tests how much accuracy drops
    when each feature is randomly shuffled.
    """

    result = permutation_importance(
        knn_model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42
    )

    return sorted(
        zip(features, result.importances_mean),
        key=lambda x: -x[1]
    )


# Purpose: Turn feature importance tuples into compact GUI text.
# Design: Keep formatting separate from scoring so both models share one display path.
# Workflow: Convert each score to a percentage string and join the feature summaries.
def format_importance_text(importance_values):
    """
    Turns feature importance values into readable text for the GUI.
    """

    return "  |  ".join(
        f"{name}: {score:.0%}"
        for name, score in importance_values
    )
