"""
preprocessing.py
================

Week 4 — from a cleaned table to a model-ready, leak-safe recipe.

The Week 2/3 `preprocess()` (dropna + get_dummies + one split) is gone. In its
place: cleaning that keeps every row, a training-only duplicate drop, the
feature/target split, the locked-test-set split, and a `build_preprocessor()`
factory whose fitted steps (imputer/encoder/scaler) live *inside* the model
Pipeline — so they are re-learned on the training rows of every split and never
see held-out rows.

`flag_invalid_values()` now lives here (it's a cleaning rule); `src/data_diagnostics.py`
is deleted — its other functions were EDA-only and stay in the notebooks.

Three kinds of step, and where each may run:
  - stateless rule (clean_dataset, split_features_target): anywhere, incl. new data
  - training-data decision (drop_duplicate_rows): training file only, before the split
  - fitted step (impute/encode/scale): inside the Pipeline, fit on train rows only

Requires scikit-learn >= 1.9 and category_encoders (see requirements.txt).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder, OrdinalEncoder, TargetEncoder, StandardScaler, MinMaxScaler, RobustScaler,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from category_encoders import CountEncoder


# --------------------------------------------------------------------------- #
# Cleaning (stateless rules — run on training data AND on data to predict)
# --------------------------------------------------------------------------- #
def flag_invalid_values(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """Convert impossible-but-present values to NaN, in place, per domain rule.

    ``rules`` maps a column to ``{"min": ..., "max": ...}`` (either bound optional).
    An age of -3 or a decile score of 15 is "missing" once this runs, even though
    ``.isna()`` would never have caught it. Returns a per-column violation report.
    """
    report_rows = []
    for column, bounds in rules.items():
        if column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        lower_ok = numeric >= bounds["min"] if "min" in bounds else pd.Series(True, index=numeric.index)
        upper_ok = numeric <= bounds["max"] if "max" in bounds else pd.Series(True, index=numeric.index)
        violations = numeric.notna() & ~(lower_ok & upper_ok)
        report_rows.append({"column": column, "rule": bounds, "violations": int(violations.sum())})
        df.loc[violations, column] = np.nan
    return pd.DataFrame(report_rows)


def _canonicalize_categories(df: pd.DataFrame, columns_and_maps: dict, placeholder_tokens: set) -> pd.DataFrame:
    """Map spelling variants of a category to one canonical label; placeholders -> NaN.
    Returns a copy — does not mutate the input."""
    out = df.copy()
    for col, mapping in columns_and_maps.items():
        if col not in out.columns:
            continue
        cleaned = out[col].astype(str).str.strip()
        out[col] = cleaned.str.lower().map(mapping).fillna(cleaned)
        out.loc[out[col].astype(str).str.strip().isin(placeholder_tokens), col] = np.nan
    return out


def clean_dataset(df: pd.DataFrame, diagnostics_config: dict) -> pd.DataFrame:
    """Apply the Week 3 diagnosis: coerce numeric-text columns, invalid/placeholder
    values -> NaN, canonicalize category spelling, drop redundant columns.

    ROW-PRESERVING (Week 4): every input row comes out, same order. Removing
    duplicate rows is a training-only decision and lives in `drop_duplicate_rows()`
    — at prediction time every row needs a prediction (a Kaggle submission needs
    one per id). Target-agnostic: safe on label-free inference data.
    """
    out = df.copy()
    tokens = set(diagnostics_config.get("placeholder_tokens", []))

    for col in diagnostics_config.get("numeric_text_columns", []):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col].replace(list(tokens), np.nan), errors="coerce")

    flag_invalid_values(out, diagnostics_config.get("validity_rules", {}))

    out = _canonicalize_categories(out, diagnostics_config.get("canonical_categories", {}), tokens)

    drop_cols = [c for c in diagnostics_config.get("redundant_columns", []) if c in out.columns]
    out = out.drop(columns=drop_cols)
    return out


def drop_duplicate_rows(df: pd.DataFrame, id_column: str = None) -> pd.DataFrame:
    """TRAINING DATA ONLY. Drop exact duplicate rows and repeated ids (keep first),
    so the same person isn't counted twice or split across dev/test. Run BEFORE
    `split_dev_test()`. Never call on data you're predicting for."""
    out = df.drop_duplicates()
    if id_column and id_column in out.columns:
        out = out.drop_duplicates(subset=id_column, keep="first")
    return out


# --------------------------------------------------------------------------- #
# Features, target, and the locked test set
# --------------------------------------------------------------------------- #
def add_missingness_indicators(df: pd.DataFrame, mnar_indicator_sources: list) -> pd.DataFrame:
    """Add a `<col>_was_missing` flag for each MNAR column, BEFORE imputation — so the
    pattern survives a median/mode fill that can't carry it. Target-agnostic."""
    out = df.copy()
    for col in mnar_indicator_sources:
        if col in out.columns:
            out[f"{col}_was_missing"] = out[col].isna().astype(int)
    return out


def split_features_target(df: pd.DataFrame, data_config: dict, mnar_indicator_sources: list):
    """Return (X, y, extras). `y` is None and `extras` has no target when called on
    label-free data. `race` and `score_text` are kept in `extras` for the fairness
    audit/comparison, never as features."""
    target = data_config["target"]
    sensitive_attr = data_config["sensitive_attr"]
    drop_columns = data_config.get("drop_columns", [])

    df = add_missingness_indicators(df, mnar_indicator_sources)
    y = df[target] if target in df.columns else None

    extras_cols = [c for c in [sensitive_attr, "score_text"] if c in df.columns]
    extras = df[extras_cols].copy() if extras_cols else None

    always_drop = set(drop_columns) | {target, sensitive_attr}
    feature_cols = [c for c in df.columns if c not in always_drop]
    X = df[feature_cols]
    return X, y, extras


def split_dev_test(X, y, extras, test_size: float, random_state: int):
    """Carve out the LOCKED TEST SET (Week 4 — replaces split_train_test).

    Stratified split of X, y and extras together (kept row-aligned). The development
    set is everything we may learn from and compare on (CV splits it again into
    train/validation folds). The locked test set is never used to fit, tune, compare
    or choose — its size/seed live in config.yaml's `test_set` and never change again.
    """
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = train_test_split(
        X, y, extras, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_dev, X_test, y_dev, y_test, extras_dev, extras_test


# --------------------------------------------------------------------------- #
# The recipe: build_preprocessor() — fitted INSIDE the model Pipeline
# --------------------------------------------------------------------------- #
_SCALERS = {"none": "passthrough", "standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}

_ENCODERS = {
    "onehot": lambda seed: OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    "ordinal": lambda seed: OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
    "count": lambda seed: CountEncoder(handle_unknown=0, handle_missing=0),
    # requires scikit-learn >= 1.9 (cv accepts a splitter). On older sklearn use:
    #   TargetEncoder(target_type="binary", cv=5, shuffle=True, random_state=seed)
    "target": lambda seed: TargetEncoder(target_type="binary", cv=StratifiedKFold(5, shuffle=True, random_state=seed)),
}


def build_preprocessor(preprocessing_config: dict) -> ColumnTransformer:
    """Factory: a leak-safe ColumnTransformer for the chosen encoder/scaler pair
    (read from config, not hardcoded). Nothing is fit here — fitting happens on the
    training rows of each fold, because this object sits inside the model Pipeline."""
    encoder_name = preprocessing_config["encoder"]
    scaler_name = preprocessing_config["scaler"]
    numeric_features = preprocessing_config["numeric_features"]
    categorical_features = preprocessing_config["categorical_features"]
    mnar_indicator_sources = preprocessing_config.get("mnar_indicator_sources", [])
    imputation = preprocessing_config.get("imputation", {})

    scaler_factory = _SCALERS[scaler_name]
    scaler = scaler_factory() if callable(scaler_factory) else scaler_factory
    encoder = _ENCODERS[encoder_name](preprocessing_config.get("random_state"))

    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=imputation.get("numeric_strategy", "median"))),
        ("scale", scaler),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=imputation.get("categorical_strategy", "most_frequent"))),
        ("encode", encoder),
    ])
    indicator_cols = [f"{c}_was_missing" for c in mnar_indicator_sources]

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
        ("indicators", "passthrough", indicator_cols),
    ])
