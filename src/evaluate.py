"""
evaluate.py — Week 4: honest evaluation by cross-validation.

Replaces the Week 2/3 single-split `evaluate()`. The model is judged by stratified
k-fold CV of the *whole* pipeline (preprocessing + model) on the development set, so
every fitted step is re-learned inside each fold and the validation fold never
influences its own preprocessing. The classification report and fairness audit are
computed on out-of-fold predictions (every dev row predicted by a model that didn't
train on it), leaving the locked test set untouched.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, classification_report


def cross_validate_pipeline(pipeline, X, y, cv, scoring: str = "accuracy", n_jobs: int = 1):
    """Fit a fresh copy of `pipeline` on each fold's training part, score on its
    validation part. Because the pipeline contains the preprocessing, imputation
    medians / encoder statistics / scaler stats are re-learned inside every fold.

    Returns (fold_scores, y_oof):
      - fold_scores: one row per fold -- train, validation, and the gap (a large,
        consistent gap = overfitting).
      - y_oof: out-of-fold predictions. Each row is predicted by the one fold model
        that did NOT train on it, so all are "unseen" -- what the classification
        report and fairness check use. Taken from the same fits as the scores.
    """
    scores = cross_validate(
        pipeline, X, y, cv=cv, scoring=scoring,
        return_train_score=True, return_estimator=True, return_indices=True, n_jobs=n_jobs,
    )
    fold_scores = pd.DataFrame({
        "fold": range(1, len(scores["test_score"]) + 1),
        "train": scores["train_score"],
        "validation": scores["test_score"],
    })
    fold_scores["gap"] = fold_scores["train"] - fold_scores["validation"]

    y_oof = np.empty(len(X), dtype=np.asarray(y).dtype)
    for model, val_idx in zip(scores["estimator"], scores["indices"]["test"]):
        y_oof[val_idx] = model.predict(X.iloc[val_idx])
    return fold_scores, y_oof


def cv_report(fold_scores: pd.DataFrame, scoring: str = "accuracy") -> str:
    """Per-fold table + mean +/- std, as text (printed, and saved to results/)."""
    lines = [
        f"Cross-validation ({len(fold_scores)} stratified folds, metric: {scoring})",
        "",
        fold_scores.to_string(index=False, float_format=lambda v: f"{v:.3f}"),
        "",
    ]
    for col in ["train", "validation", "gap"]:
        sign = "+" if col == "gap" else ""
        lines.append(f"{col.capitalize():<11s} mean = {fold_scores[col].mean():{sign}.3f}   "
                     f"std = {fold_scores[col].std(ddof=1):.3f}")
    text = "\n".join(lines)
    print(text)
    return text


def oof_classification_report(y_true, y_pred) -> str:
    """Classification report on the out-of-fold predictions."""
    text = "Classification report (out-of-fold predictions, development set):\n" + \
        classification_report(y_true, y_pred, zero_division=0)
    print(text)
    return text


def fairness_report(y_true, y_pred, extras: pd.DataFrame, sensitive_attr: str = "race") -> str:
    """
    Deliberately simple fairness check -- not a substitute for a real audit, just
    enough to show "accurate" and "fair" are not the same thing.

    For each race group, the false positive rate (did NOT reoffend but was predicted
    to) for our own model (out-of-fold predictions) and for COMPAS's own score
    (score_text != "Low" counts as a "high risk" prediction), on the same rows.
    """
    df = extras.copy()
    df["y_true"] = pd.Series(y_true).values
    df["y_pred_model"] = y_pred
    df["y_pred_compas"] = (df["score_text"] != "Low").astype(int)

    lines = [
        "False positive rate by race (development set, out-of-fold)",
        "(share of people who did NOT reoffend, but were predicted to)",
        "",
    ]
    for label, col in [("Our model", "y_pred_model"), ("COMPAS's own score", "y_pred_compas")]:
        lines.append(f"  {label}:")
        for group, g in df.groupby(sensitive_attr):
            negatives = g[g["y_true"] == 0]
            if len(negatives) == 0:
                continue
            fpr = (negatives[col] == 1).mean()
            lines.append(f"    {group:<20s} FPR = {fpr:.2f}  (n={len(negatives)})")
        lines.append("")

    text = "\n".join(lines)
    print(text)
    return text
