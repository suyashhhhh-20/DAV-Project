"""
data_utils.py
Helper functions used by the training script, the EDA script and the dashboard.
Everything about loading, cleaning and preparing the heart disease data lives here.
"""

import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------------------------------------------------
# Column names of the Kaggle "Heart Failure Prediction" dataset
# ----------------------------------------------------------------------
TARGET = "HeartDisease"  # 1 = heart disease, 0 = no heart disease

NUMERIC_FEATURES = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
CATEGORICAL_FEATURES = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns where a value of 0 is medically impossible, so 0 really means "missing"
ZERO_MEANS_MISSING = ["RestingBP", "Cholesterol"]

# Columns on which we will cap extreme values (outliers)
OUTLIER_COLUMNS = ["RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]

# Friendly names used on charts and in the dashboard
CHEST_PAIN_NAMES = {
    "TA": "Typical Angina",
    "ATA": "Atypical Angina",
    "NAP": "Non-Anginal Pain",
    "ASY": "Asymptomatic",
}


def load_data(csv_path):
    """Read the CSV file and make sure it has the columns we expect."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            "Dataset not found at: " + str(csv_path) + "\n"
            "Download heart.csv from "
            "https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction "
            "and save it as data/heart.csv (see README.md, Step 4)."
        )
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]  # remove accidental spaces

    missing_columns = [c for c in ALL_FEATURES + [TARGET] if c not in df.columns]
    if missing_columns:
        raise ValueError("The CSV is missing these columns: " + str(missing_columns))
    return df


def clean_data(df):
    """
    Basic cleaning:
      1. remove exact duplicate rows
      2. treat impossible zeros (RestingBP = 0, Cholesterol = 0) as missing values (NaN)
    Returns the cleaned DataFrame and a small dictionary describing what was done.
    """
    df = df.copy()
    rows_before = len(df)

    # 1. Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    duplicates_removed = rows_before - len(df)

    # 2. Zeros that really mean "missing"
    zeros_found = {}
    for col in ZERO_MEANS_MISSING:
        zeros_found[col] = int((df[col] == 0).sum())
        df[col] = df[col].replace(0, np.nan)

    info = {
        "rows_before": int(rows_before),
        "duplicates_removed": int(duplicates_removed),
        "rows_after": int(len(df)),
        "zeros_treated_as_missing": zeros_found,
        "missing_after_cleaning": {c: int(v) for c, v in df.isnull().sum().items() if v > 0},
    }
    return df, info


def get_outlier_bounds(train_features):
    """
    Work out the IQR limits for each outlier column using ONLY the training data.
    lower = Q1 - 1.5 * IQR      upper = Q3 + 1.5 * IQR
    """
    bounds = {}
    for col in OUTLIER_COLUMNS:
        q1 = train_features[col].quantile(0.25)
        q3 = train_features[col].quantile(0.75)
        iqr = q3 - q1
        bounds[col] = [float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)]
    return bounds


def apply_outlier_caps(features, bounds):
    """Cap (clip) values outside the limits. Missing values (NaN) stay as they are."""
    features = features.copy()
    for col, (lower, upper) in bounds.items():
        features[col] = features[col].clip(lower=lower, upper=upper)
    return features


def build_preprocessor():
    """
    Build the preprocessing steps:
      numeric columns     -> fill missing values with the median -> scale (StandardScaler)
      categorical columns -> fill missing with most common value -> one-hot encoding
    These steps are LEARNED from the training data only when we call .fit().
    """
    numeric_steps = Pipeline([
        ("fill_missing", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_steps = Pipeline([
        ("fill_missing", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_steps, NUMERIC_FEATURES),
        ("cat", categorical_steps, CATEGORICAL_FEATURES),
    ])


def predict_one_patient(model, bounds, patient):
    """
    Predict for ONE patient entered in the dashboard.
    patient is a dictionary such as {"Age": 54, "Sex": "M", ...}
    Returns (predicted_class, probability_of_heart_disease).
    """
    row = pd.DataFrame([patient])[ALL_FEATURES]
    for col in ZERO_MEANS_MISSING:           # 0 entered by the user = value unknown
        row[col] = row[col].replace(0, np.nan)
    row = apply_outlier_caps(row, bounds)     # same capping as the training data
    probability = float(model.predict_proba(row)[0, 1])
    return int(probability >= 0.5), probability
