"""
data_diagnostics.py
===================

Week 3 (EDA) additions to the pipeline: the three diagnostic techniques from
the EDA notebook, as reusable, config-driven functions.

    1. Missingness *mechanism*  -> test_missingness_mechanism
    2. Invalid values (domain rules) -> flag_invalid_values
    3. Duplicates, two ways     -> find_duplicates

plus one private helper (_cramers_v). Nothing is hardcoded to COMPAS: the
column lists, validity rules and id column are passed in, normally from the
`diagnostics` block of config.yaml.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def _cramers_v(confusion_matrix: pd.DataFrame) -> float:
    """Bias-corrected Cramer's V effect size for a chi-square test of association.

    Ranges in [0, 1]: ~0 = no association, ~1 = near-perfect. At ~7,000 rows a
    chi-square p-value goes tiny for trivial associations, so this effect size —
    not the p-value — is what says whether an association is big enough to care.
    """
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.to_numpy().sum()
    if n == 0:
        return 0.0
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2_corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    return 0.0 if denom <= 0 else float(np.sqrt(phi2_corr / denom))


def test_missingness_mechanism(df: pd.DataFrame, target_col: str, candidate_predictors: list) -> pd.DataFrame:
    """Test *why* a column is missing (MCAR vs MAR/MNAR).

    For ``target_col``'s missing-value indicator, cross-tabulate against every
    column in ``candidate_predictors``, run a chi-square test and report
    Cramer's V. One row per predictor, strongest association first.

    Reading the verdict: a max Cramer's V well under 0.1 across all predictors
    reads as MCAR (no pattern -> safe to impute simply). Roughly >= 0.2 reads as
    MAR/MNAR, and the top predictor usually points at why.

    The model's target must NOT be in ``candidate_predictors`` — testing against
    it would leak the label into how we clean the data.
    """
    indicator = df[target_col].isna()
    rows = []
    for predictor in candidate_predictors:
        if predictor == target_col or predictor not in df.columns:
            continue
        sub = pd.DataFrame({"missing": indicator, "predictor": df[predictor]}).dropna(subset=["predictor"])
        if sub["predictor"].nunique() < 2 or sub["missing"].nunique() < 2:
            continue
        table = pd.crosstab(sub["missing"], sub["predictor"])
        _, p, _, _ = chi2_contingency(table)
        rows.append({"predictor": predictor, "cramers_v": round(_cramers_v(table), 3), "p_value": p, "n": len(sub)})
    if not rows:
        return pd.DataFrame(columns=["predictor", "cramers_v", "p_value", "n"])
    return pd.DataFrame(rows).sort_values("cramers_v", ascending=False).reset_index(drop=True)


def flag_invalid_values(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """Detect values that are present but impossible, using domain rules.

    ``.isna()`` only catches actual NaN. It says nothing about an age of -3 or a
    decile score of 15 on a 1..10 scale. A rule encodes what a column *means*.

    ``rules`` maps a column to a bounds dict with optional ``min`` / ``max``
    (structured, not an eval'd string). Violations are converted to NaN **in
    place** on ``df`` — "impossible but present" is still missing — and a report
    of how many were found per column is returned.
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


def find_duplicates(df: pd.DataFrame, id_column: str | None = None) -> dict:
    """Two duplicate checks that catch different failures.

    ``.duplicated()`` catches rows identical in *every* column. Repeated ids
    catch the same case entered twice with one field typo'd differently — which
    the exact-row check misses. They can disagree in general, so both are
    reported rather than trusting either alone.
    """
    result = {"exact_row_duplicates": int(df.duplicated().sum())}
    if id_column and id_column in df.columns:
        result["repeated_ids"] = int(df[id_column].duplicated().sum())
    return result