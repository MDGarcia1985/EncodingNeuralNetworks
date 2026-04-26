# Purpose: Explain what the Titanic survival prediction project does and how to use it.
# Design: Provide setup, dataset placement, run commands, tests, and package export notes in one place.
# Workflow: Read this first, install dependencies, place the dataset, run the app, then run pytest.
#
# Michael Garcia
# michael@mandedesign.studio
# https://mandedesign.studio
#
# License: MIT

# Titanic Survival Prediction App

A Python machine learning project that predicts Titanic passenger survival with two
scikit-learn models and a Tkinter GUI.

The app trains a Decision Tree and KNN classifier from the Titanic dataset, compares
their accuracy, predicts survival from a passenger profile, and displays feature
importance results.

## Project Structure

```text
.
|-- scripts/
|   `-- run_app.py
|-- src/
|   |-- data/
|   |   |-- raw/
|   |   |   `-- Titanic-Dataset.csv
|   |   `-- load_data.py
|   |-- model/
|   |   |-- feature_importance.py
|   |   |-- predict.py
|   |   `-- train_model.py
|   |-- ui/
|   |   `-- app.py
|   `-- utils/
|       `-- paths.py
|-- tests/
|   |-- data/
|   |   `-- testLog.txt
|   |-- test_data_loading.py
|   |-- test_feature_importance.py
|   |-- test_paths.py
|   |-- test_predict.py
|   `-- test_train_model.py
|-- requirements.txt
`-- README.md
```

## Requirements

- Python 3.11 or newer
- pandas
- scikit-learn
- pytest
- Tkinter, included with many Python installs but not all virtual environments

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

Place the Titanic CSV file here:

```text
src/data/raw/Titanic-Dataset.csv
```

The project path helper in `src/utils/paths.py` points the app and tests to that
location.

## Run The App

Preferred module command:

```bash
python -m src.ui.app
```

Script command:

```bash
python scripts/run_app.py
```

If either command reports `ModuleNotFoundError: No module named 'tkinter'`, install
or use a Python build that includes Tkinter.

## Run Tests

Run the full pytest suite:

```bash
python -m pytest -q
```

The suite covers:

- Data loading and feature selection
- Project path resolution
- Passenger profile creation and prediction
- Decision Tree and KNN training
- Feature importance formatting

Each test appends a result block to:

```text
tests/data/testLog.txt
```

The log records the test number, date, 24-hour time, pass/fail status, and
individual result printouts.

## Package Exports

The `src` subpackages expose their common helpers through `__init__.py` files:

- `src.data`: `FEATURES`, `load_and_prepare_data`
- `src.model`: training, prediction, accuracy, and feature-importance helpers
- `src.utils`: `PROJECT_ROOT`, `DATA_DIR`, `TITANIC_DATASET`
