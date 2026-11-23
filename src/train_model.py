"""
train_model.py
Stages 3 and 4: preprocessing + model building.

Run from the project folder:
    python src/train_model.py
It saves:
    models/all_models.joblib   (the 3 trained models + outlier limits)
    outputs/results.json       (all numbers used by the dashboard and the report)
    outputs/model_comparison.csv
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

# Make "import data_utils" work whether we run from the project root or from src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import (ALL_FEATURES, TARGET, apply_outlier_caps, build_preprocessor,
                        clean_data, get_outlier_bounds, load_data)

RANDOM_STATE = 42   # fixed number so results are the same every time
TEST_SIZE = 0.20    # 80% training, 20% testing

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join(PROJECT_ROOT, "data", "heart.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_FILE = os.path.join(MODELS_DIR, "all_models.joblib")
RESULTS_FILE = os.path.join(OUTPUTS_DIR, "results.json")


def get_models():
    """The three classification algorithms required by the assignment."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        # max_depth limits the tree so that it does not simply memorise the training data
        "Decision Tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=5,
                                                random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8,
                                                min_samples_leaf=3, random_state=RANDOM_STATE),
    }


def get_feature_names(fitted_pipeline):
    """Names of the columns after encoding (e.g. 'Age', 'ChestPainType_ASY')."""
    names = fitted_pipeline.named_steps["prep"].get_feature_names_out()
    return [n.split("__", 1)[1] for n in names]  # remove the 'num__' / 'cat__' prefix


def train_and_evaluate(csv_path=DEFAULT_CSV, verbose=True):
    """Run the complete preprocessing + training + evaluation and save everything."""

    def say(text):
        if verbose:
            print(text)

    # ---------------- Load and clean ----------------
    df = load_data(csv_path)
    df, info = clean_data(df)
    say("Rows after cleaning: %d (duplicates removed: %d)" % (info["rows_after"], info["duplicates_removed"]))
    say("Zeros treated as missing: " + str(info["zeros_treated_as_missing"]))

    X = df[ALL_FEATURES]
    y = df[TARGET]

    # ---------------- Split FIRST (prevents data leakage) ----------------
    # stratify=y keeps the same disease / no-disease ratio in both parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    say("Training rows: %d   Testing rows: %d" % (len(X_train), len(X_test)))

    # ---------------- Outliers: limits come from the training data only ----------------
    bounds = get_outlier_bounds(X_train)
    X_train = apply_outlier_caps(X_train, bounds)
    X_test = apply_outlier_caps(X_test, bounds)

    # ---------------- Train and evaluate every model ----------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    trained = {}
    results = {}
    rows = []

    for name, classifier in get_models().items():
        # A Pipeline = preprocessing + model together. fit() learns the medians,
        # scaling and encoding from the TRAINING data only.
        pipe = Pipeline([("prep", build_preprocessor()), ("model", classifier)])

        # 5-fold cross-validation on the TRAINING data (used to choose the best model)
        cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "train_accuracy": float(accuracy_score(y_train, pipe.predict(X_train))),
            "cv_f1_mean": float(cv_f1.mean()),
            "cv_f1_std": float(cv_f1.std()),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),  # [[TN, FP], [FN, TP]]
        }
        trained[name] = pipe
        results[name] = metrics
        rows.append({"Model": name, **{k: v for k, v in metrics.items() if k != "confusion_matrix"}})
        say("%-20s accuracy=%.3f precision=%.3f recall=%.3f f1=%.3f (CV f1=%.3f)" % (
            name, metrics["accuracy"], metrics["precision"], metrics["recall"],
            metrics["f1"], metrics["cv_f1_mean"]))

    # ---------------- Choose the best model ----------------
    # Rule: highest cross-validated F1 on the TRAINING data (the test set is not used to choose).
    best_name = max(results, key=lambda n: results[n]["cv_f1_mean"])
    say("Best model (highest cross-validated F1): " + best_name)

    # Majority-class baseline: accuracy of always predicting the most common class
    baseline_accuracy = float(max(y_test.mean(), 1 - y_test.mean()))

    # Feature importance from the Random Forest (for the dashboard / report)
    rf = trained["Random Forest"]
    importance = pd.Series(rf.named_steps["model"].feature_importances_,
                           index=get_feature_names(rf)).sort_values(ascending=False)

    output = {
        "dataset": {
            **info,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
            "test_class_counts": {str(k): int(v) for k, v in y_test.value_counts().sort_index().items()},
        },
        "outlier_bounds": bounds,
        "models": results,
        "best_model": best_name,
        "selection_rule": "Highest mean 5-fold cross-validated F1-score on the training data",
        "baseline_accuracy": baseline_accuracy,
        "feature_importance": {k: float(v) for k, v in importance.items()},
        "settings": {"random_state": RANDOM_STATE, "test_size": TEST_SIZE},
    }

    # ---------------- Save ----------------
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    joblib.dump({"models": trained, "bounds": bounds, "best_model": best_name}, MODEL_FILE)
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    pd.DataFrame(rows).round(4).to_csv(os.path.join(OUTPUTS_DIR, "model_comparison.csv"), index=False)
    say("Saved: " + MODEL_FILE)
    say("Saved: " + RESULTS_FILE)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the heart disease models")
    parser.add_argument("--data", default=DEFAULT_CSV, help="path to heart.csv")
    args = parser.parse_args()
    train_and_evaluate(args.data)
