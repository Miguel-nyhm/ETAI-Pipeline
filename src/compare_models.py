"""
compare_models.py — build the week-5 "Ensembles" comparison table for every model.

Optional convenience on top of the course pipeline. `main.py` runs ONE model (the course
design) and prints its default CV, its tuned nested CV and its chosen hyperparameters. The
README asks for one table with a row per model (`### Ensembles`); getting it by hand means
editing `config.model.type` and rerunning `main.py` once per model. This script does exactly
those runs in one go, on the SAME folds, and prints the table ready to paste into the README.

    python compare_models.py

For every model it reports the Default CV (its `DEFAULTS` params) and, when the model has a
search space in `config.tuning.search_spaces` and tuning is enabled, the honest Tuned nested
CV, the best-trial tuning score, and the hyperparameters chosen on the whole development set.
Logistic regression has no search space, so its tuned columns are "—" (a default-only
baseline), matching the notebook's template. The locked test set is never touched.
"""
import os
from datetime import datetime

import yaml
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.data import load_data
from src.preprocessing import (
    clean_dataset, drop_duplicate_rows, split_features_target, split_dev_test,
)
from src.model import build_pipeline
from src.evaluate import cross_validate_pipeline
from src.tuning import nested_cross_validate, tune_pipeline


# Row order and display names for the README table (the notebook's `### Ensembles` rows).
MODELS = [
    ("logistic_regression",   "Logistic regression"),
    ("decision_tree",         "Decision tree"),
    ("random_forest",         "Random forest"),
    ("oblique_random_forest", "Oblique random forest"),
    ("adaboost",              "AdaBoost"),
]

# Default (non-tuned) params per model: the baseline each "Default: CV" column is measured at,
# and the fixed params the tuned runs build on (seeds, n_jobs, counts the search space leaves
# alone). These mirror the notebook's MODEL_CONFIGS; a model not listed uses {}.
DEFAULTS = {
    "logistic_regression":   {"max_iter": 2000},
    "decision_tree":         {"random_state": 42},
    "random_forest":         {"n_estimators": 200, "random_state": 42, "n_jobs": -1},
    "oblique_random_forest": {"n_estimators": 200, "n_components": 10, "random_state": 42, "n_jobs": -1},
    "adaboost":              {"max_depth": 1, "n_estimators": 200, "learning_rate": 1.0, "random_state": 42},
}

DASH = "—"


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def fmt_score(mean: float, std: float, gap: float) -> str:
    """'0.683 ± 0.012 (+0.031)' — mean ± std of the validation score, with the train-val gap."""
    return f"{mean:.3f} ± {std:.3f} ({gap:+.3f})"


def fmt_params(params: dict) -> str:
    """Chosen hyperparameters as 'name=value, ...', with the leading 'model__' step dropped."""
    parts = []
    for key, value in params.items():
        name = key[len("model__"):] if key.startswith("model__") else key
        shown = f"{value:.4g}" if isinstance(value, float) else value
        parts.append(f"{name}={shown}")
    return ", ".join(parts)


def main():
    config = load_config()

    df = load_data(config["data"]["path"])
    df = clean_dataset(df, config["diagnostics"])
    df = drop_duplicate_rows(df, config["diagnostics"].get("id_column"))
    X, y, extras = split_features_target(
        df, config["data"], config["preprocessing"]["mnar_indicator_sources"]
    )
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = split_dev_test(
        X, y, extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"],
    )

    cv_cfg, tun = config["cv"], config["tuning"]
    scoring = cv_cfg.get("scoring", "accuracy")
    n_jobs = cv_cfg.get("n_jobs", 1)
    tuning_on = tun.get("enabled", True)
    spaces = tun["search_spaces"]

    # outer = the usual CV folds (shared by every row); inner = the tuning folds.
    outer_cv = StratifiedKFold(
        n_splits=cv_cfg["n_splits"], shuffle=cv_cfg.get("shuffle", True),
        random_state=cv_cfg.get("random_state") if cv_cfg.get("shuffle", True) else None,
    )
    inner_cv = StratifiedKFold(
        n_splits=tun["n_splits"], shuffle=True, random_state=tun["random_state"]
    )

    rows = []
    for name, label in MODELS:
        pipeline = build_pipeline(config["preprocessing"], {"type": name, "params": DEFAULTS.get(name, {})})

        # Default column: CV at the baseline params (the first table main.py prints).
        fs_default, _ = cross_validate_pipeline(pipeline, X_dev, y_dev, outer_cv, scoring, n_jobs)
        default_cell = fmt_score(
            fs_default["validation"].mean(), fs_default["validation"].std(ddof=1), fs_default["gap"].mean()
        )

        space = spaces.get(name)
        if tuning_on and space:
            print(f"Tuning {name} ...")
            nested, _ = nested_cross_validate(
                pipeline, X_dev, y_dev, outer_cv, inner_cv, scoring,
                space, tun["n_trials"], tun["random_state"], n_jobs,
            )
            _, study = tune_pipeline(
                pipeline, X_dev, y_dev, inner_cv, scoring,
                space, tun["n_trials"], tun["random_state"], n_jobs,
            )
            tuned_cell = fmt_score(
                nested["validation"].mean(), nested["validation"].std(ddof=1), nested["gap"].mean()
            )
            best_cell = f"{study.best_value:.3f}"
            chosen_cell = fmt_params(study.best_params)
            sort_key = nested["validation"].mean()
        else:
            print(f"{name}: default only (no search space)" if not space else f"{name}: tuning disabled")
            tuned_cell = best_cell = chosen_cell = DASH
            sort_key = fs_default["validation"].mean()

        rows.append({
            "Model": label,
            f"Default: CV mean ± std (gap)": default_cell,
            f"Tuned: nested CV mean ± std (gap)": tuned_cell,
            "Tuning score (best trial)": best_cell,
            "Chosen hyperparameters": chosen_cell,
            "_sort": sort_key,
        })

    table = pd.DataFrame(rows).drop(columns="_sort")

    # Markdown, in the notebook's row order, ready to paste under "## Model evaluation".
    header = "| " + " | ".join(table.columns) + " |"
    sep = "| " + " | ".join("---" for _ in table.columns) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in table.itertuples(index=False)]
    note = (f"<!-- nested CV, metric: {scoring}, {tun['n_trials']} trials, "
            f"{cv_cfg['n_splits']} outer x {tun['n_splits']} inner folds -->")
    md = "### Ensembles\n" + note + "\n" + "\n".join([header, sep, *body]) + "\n"

    # Which model to report, by the honest number (tuned where tuned, else default).
    best = max(rows, key=lambda r: r["_sort"])
    md += f"\n_Best honest {scoring}: **{best['Model']}** ({best['_sort']:.3f})._\n"

    print("\n" + md)

    results_dir = config.get("output", {}).get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"readme_table_{ts}.md")
    with open(path, "w") as f:
        f.write(md)
    print(f"README table saved to {path}")


if __name__ == "__main__":
    main()
