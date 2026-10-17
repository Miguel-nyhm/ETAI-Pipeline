"""
tuning.py — Week 5: hyperparameter tuning (Optuna) + nested cross-validation.

Tuning is a decision, so it follows the same rules as every other decision:
development set only, every candidate scored by CV of the whole Pipeline, same
folds for every trial, a seeded sampler. The honest score for the *tuning
procedure* comes from nested CV (inner loop chooses, outer loop judges).
"""
import numpy as np
import pandas as pd
import optuna
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import get_scorer

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _make_objective(pipeline, X, y, search_space, inner_cv, scoring):
    """Build the Optuna objective: read each hyperparameter's range from the
    search space, set it on a clone of the Pipeline, score it by inner CV."""
    def objective(trial):
        params = {}
        for name, spec in search_space.items():
            kind = spec["type"]
            if kind == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"], log=spec.get("log", False))
            elif kind == "float":
                params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
            elif kind == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])
            else:
                raise ValueError(f"Unknown search-space type: {kind}")
        candidate = clone(pipeline).set_params(**params)
        return cross_val_score(candidate, X, y, cv=inner_cv, scoring=scoring).mean()
    return objective


def tune_pipeline(pipeline, X, y, tuning_cfg, scoring="accuracy"):
    """Run an Optuna study over the search space (config `tuning` section).
    Each trial clones the Pipeline, sets the candidate's hyperparameters, and
    scores it by inner stratified k-fold CV on X, y. Returns the *unfitted*
    winning pipeline and the study."""
    inner_cv = StratifiedKFold(
        n_splits=tuning_cfg["n_splits"], shuffle=True, random_state=tuning_cfg["random_state"]
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=tuning_cfg["random_state"]),
    )
    study.optimize(
        _make_objective(pipeline, X, y, tuning_cfg["search_space"], inner_cv, scoring),
        n_trials=tuning_cfg["n_trials"],
    )
    best_pipeline = clone(pipeline).set_params(**study.best_params)
    return best_pipeline, study


def nested_cross_validate(pipeline, X, y, outer_cv, tuning_cfg, scoring="accuracy"):
    """Honest evaluation of the whole tuning procedure.

    Outer loop judges, inner loop (inside tune_pipeline) chooses. Each outer fold
    runs a complete tuning on its outer-train, refits the winner, and scores it on
    the outer-validation rows — which never influenced the choice. Same return
    shape as cross_validate_pipeline (fold table + out-of-fold predictions), so the
    downstream reports work unchanged.
    """
    scorer = get_scorer(scoring)
    rows = []
    y_oof = np.empty(len(X), dtype=np.asarray(y).dtype)
    for fold, (tr, va) in enumerate(outer_cv.split(X, y), start=1):
        X_tr, y_tr = X.iloc[tr], y.iloc[tr]
        X_va, y_va = X.iloc[va], y.iloc[va]
        best_pipe, study = tune_pipeline(pipeline, X_tr, y_tr, tuning_cfg, scoring)
        model = best_pipe.fit(X_tr, y_tr)
        y_oof[va] = model.predict(X_va)
        rows.append({
            "fold": fold,
            "train": scorer(model, X_tr, y_tr),
            "validation": scorer(model, X_va, y_va),
            "inner_best": study.best_value,
            **study.best_params,
        })
    fold_scores = pd.DataFrame(rows)
    fold_scores["gap"] = fold_scores["train"] - fold_scores["validation"]
    return fold_scores, y_oof


def tuning_report(fold_scores: pd.DataFrame, scoring: str = "accuracy") -> str:
    """Per-fold nested-CV table + mean/std, as text (printed and saved)."""
    show = [c for c in ["fold", "train", "validation", "gap", "inner_best"] if c in fold_scores] + \
           [c for c in fold_scores.columns if c.startswith("model__")]
    lines = [
        f"Nested cross-validation ({len(fold_scores)} outer folds, metric: {scoring})",
        "",
        fold_scores[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"),
        "",
    ]
    for col in ["train", "validation", "gap", "inner_best"]:
        if col in fold_scores:
            sign = "+" if col == "gap" else ""
            lines.append(f"{col.capitalize():<12s} mean = {fold_scores[col].mean():{sign}.3f}   "
                         f"std = {fold_scores[col].std(ddof=1):.3f}")
    text = "\n".join(lines)
    print(text)
    return text
