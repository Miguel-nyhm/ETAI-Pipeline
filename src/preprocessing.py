"""
preprocessing.py
================
 
Week 3 — the cleaning step. This runs *before* the split and before any
fit-dependent preprocessing (encoding, scaling, imputation), which is what
keeps it leak-safe: cleaning here is dataset-wide but target-agnostic, so it
depends on no label and can run on unlabeled inference data too.
 
For now this file holds only `clean_dataset` (+ its category helper). Next week
the fit-dependent preprocessing (ColumnTransformer, encoder/scaler, the
train/test split) grows into this same file.
 
`clean_dataset` applies the diagnosis from data_diagnostics.py:
    placeholder-token -> NaN, domain-rule violations -> NaN (via
    flag_invalid_values), category spelling normalized, duplicates dropped,
    redundant columns removed.
"""
from __future__ import annotations
 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


from src.data_diagnostics import flag_invalid_values
 
 
def _canonicalize_categories(
    df: pd.DataFrame, columns_and_maps: dict, placeholder_tokens: set
) -> pd.DataFrame:
    """Map spelling variants of a category to one canonical label.
 
    Normalizes whitespace/case, maps known variants (e.g. ``MALE`` -> ``Male``),
    leaves genuinely new labels untouched, and folds any placeholder token into
    NaN. Returns a copy — does not mutate the input.
    """
    out = df.copy()
    for col, mapping in columns_and_maps.items():
        if col not in out.columns:
            continue
        cleaned = out[col].astype(str).str.strip()
        out[col] = cleaned.str.lower().map(mapping).fillna(cleaned)
        out.loc[out[col].astype(str).str.strip().isin(placeholder_tokens), col] = np.nan
    return out
 
 
def clean_dataset(df: pd.DataFrame, diagnostics_config: dict) -> pd.DataFrame:
    """Turn the raw file into a cleaned dataset, using the diagnosis config.
 
    Order matters:
      1. coerce numeric-looking text columns (placeholder token -> NaN, then numeric)
      2. domain-rule violations -> NaN  (reuses flag_invalid_values)
      3. canonicalize category spelling (and category placeholders -> NaN)
      4. drop exact-duplicate rows, then rows with a repeated id
      5. drop the perfectly redundant columns
 
    Target-agnostic and fit-free — nothing here learns a statistic from the
    data, so it is safe to run on the whole dataset before the split.
    """
    out = df.copy()
    tokens = set(diagnostics_config.get("placeholder_tokens", []))
 
    # 1. numeric columns that loaded as text because of a placeholder token
    for col in diagnostics_config.get("numeric_text_columns", []):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col].replace(list(tokens), np.nan), errors="coerce")
 
    # 2. impossible values -> NaN (mutates `out` in place, report ignored here)
    flag_invalid_values(out, diagnostics_config.get("validity_rules", {}))
 
    # 3. category spelling / category placeholders -> NaN
    out = _canonicalize_categories(out, diagnostics_config.get("canonical_categories", {}), tokens)
 
    # 4. duplicates: exact rows, then repeated ids
    out = out.drop_duplicates()
    id_column = diagnostics_config.get("id_column")
    if id_column and id_column in out.columns:
        out = out.drop_duplicates(subset=id_column, keep="first")
 
    # 5. redundant columns (multicollinearity)
    drop_cols = [c for c in diagnostics_config.get("redundant_columns", []) if c in out.columns]
    out = out.drop(columns=drop_cols)
 
    return out


def preprocess(
    df: pd.DataFrame,
    target: str,
    sensitive_attr: str,
    drop_columns: list,
    test_size: float,
    random_state: int,
):
    # naive: just drop rows with any missing values
    df = df.dropna()

    y = df[target]

    # kept aside for fairness auditing after training -- never used as a model input
    extras = df[[sensitive_attr, "score_text"]].copy()

    columns_to_exclude = [target, sensitive_attr] + [
        c for c in drop_columns if c in df.columns
    ]
    X = df.drop(columns=columns_to_exclude)

    # naive: one-hot encode all non-numeric columns, no further thought
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test, extras_train, extras_test = train_test_split(
        X, y, extras, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, extras_test
