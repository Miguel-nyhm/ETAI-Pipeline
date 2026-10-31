"""
compare_models.py -- fill the scoreboard with every model in one run.

Optional convenience on top of the course pipeline. `main.py` runs ONE model and appends its
row to the scoreboard (the course design: A4 is "run main.py once per model"). This script does
those runs in one go, on the SAME folds: each model untuned, and -- if it has a search space in
`config.tuning.search_spaces` and tuning is enabled -- tuned with nested CV, scored on the
selection metric. Every run is appended to `output.scoreboard` with the same schema main.py
writes, so the two are interchangeable. The locked test set is never touched.

    python compare_models.py                 # all models below
    python compare_models.py adaboost dummy  # only these

Each model's default (untuned) params come from DEFAULTS; a model not listed uses {}.
"""
import sys
from datetime import datetime

import yaml
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.data import load_data
from src.preprocessing import (
    clean_dataset, drop_duplicate_rows, split_features_target, split_dev_test,
)
from src.model import build_pipeline
from src.evaluate import build_scorers, cross_validate_pipeline
from src.tuning import nested_cross_validate, tune_pipeline
from src.results import save_scores


# Row order and default (untuned) params per model. These mirror config.yaml's model examples.
MODELS = ["dummy", "logistic_regression", "naive_bayes", "decision_tree", "random_forest",
          "oblique_random_forest", "adaboost", "gradient_boosting", "hist_gradient_boosting"]
DEFAULTS = {
    "dummy": {},
    "logistic_regression": {"max_iter": 2000},
    "naive_bayes": {},
    "decision_tree": {"random_state": 42},
    "random_forest": {"n_estimators": 200, "random_state": 42, "n_jobs": -1},
    "oblique_random_forest": {"n_estimators": 200, "n_components": 10, "random_state": 42, "n_jobs": -1},
    "adaboost": {"max_depth": 1, "n_estimators": 200, "learning_rate": 1.0, "random_state": 42},
    "gradient_boosting": {"random_state": 42},
    "hist_gradient_boosting": {"random_state": 42},
}


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def prepare(config):
    """Load, clean and split once; return everything the per-model scoring needs."""
    df = load_data(config["data"]["path"])
    df = clean_dataset(df, config["diagnostics"])
    df = drop_duplicate_rows(df, config["diagnostics"].get("id_column"))
    X, y, extras = split_features_target(df, config["data"],
                                         config["preprocessing"].get("mnar_indicator_sources", []))
    X_dev, X_test, y_dev, y_test, ed, et = split_dev_test(
        X, y, extras, test_size=config["test_set"]["size"], random_state=config["test_set"]["random_state"])
    cv_cfg, tun = config["cv"], config.get("tuning", {})
    scorers = build_scorers(cv_cfg["metrics"])
    selection_metric = cv_cfg["selection_metric"]
    n_jobs = cv_cfg.get("n_jobs", 1)
    outer_cv = StratifiedKFold(n_splits=cv_cfg["n_splits"], shuffle=cv_cfg.get("shuffle", True),
                               random_state=cv_cfg.get("random_state") if cv_cfg.get("shuffle", True) else None)
    inner_cv = StratifiedKFold(n_splits=tun.get("n_splits", 5), shuffle=True,
                               random_state=tun.get("random_state", 42))
    floor_scores, _ = cross_validate_pipeline(
        build_pipeline(config["preprocessing"], {"type": "dummy"}), X_dev, y_dev, outer_cv,
        {selection_metric: scorers[selection_metric]}, selection_metric, n_jobs)
    floor = floor_scores["validation"].mean()
    return dict(config=config, X_dev=X_dev, y_dev=y_dev, scorers=scorers, selection_metric=selection_metric,
                n_jobs=n_jobs, outer_cv=outer_cv, inner_cv=inner_cv, floor=floor, tun=tun)


def score_model(ctx, name: str) -> dict:
    """Run one model (tuned if it has a search space and tuning is on) and build its scoreboard row."""
    config, selection_metric = ctx["config"], ctx["selection_metric"]
    pipe = build_pipeline(config["preprocessing"], {"type": name, "params": DEFAULTS.get(name, {})})
    untuned_scores, _ = cross_validate_pipeline(pipe, ctx["X_dev"], ctx["y_dev"], ctx["outer_cv"],
                                                ctx["scorers"], selection_metric, ctx["n_jobs"])
    space = (ctx["tun"].get("search_spaces") or {}).get(name)
    tuned = bool(ctx["tun"].get("enabled", True) and space)

    if tuned:
        final_scores, _ = nested_cross_validate(
            pipe, ctx["X_dev"], ctx["y_dev"], ctx["outer_cv"], ctx["inner_cv"], ctx["scorers"],
            selection_metric, space, ctx["tun"]["n_trials"], ctx["tun"]["random_state"], ctx["n_jobs"])
        _, study = tune_pipeline(pipe, ctx["X_dev"], ctx["y_dev"], ctx["inner_cv"],
                                 ctx["scorers"][selection_metric], space,
                                 ctx["tun"]["n_trials"], ctx["tun"]["random_state"], ctx["n_jobs"])
        tuned_params = str(study.best_params)
    else:
        final_scores, tuned_params = untuned_scores, ""

    return {"run": datetime.now().strftime("%Y%m%d_%H%M%S_") + name, "data": config["data"]["path"],
            "model": name, "tuned": tuned, "encoder": config["preprocessing"]["encoder"],
            "scaler": config["preprocessing"]["scaler"], "params": str(DEFAULTS.get(name, {})),
            "tuned_params": tuned_params, "metric": selection_metric,
            "mean": final_scores["validation"].mean(), "std": final_scores["validation"].std(ddof=1),
            "untuned": untuned_scores["validation"].mean(), "floor": ctx["floor"],
            **{f"fold_{i}": s for i, s in enumerate(final_scores["validation"], start=1)}}


def main():
    config = load_config()
    names = sys.argv[1:] or MODELS
    ctx = prepare(config)
    board_path = config.get("output", {}).get("scoreboard", "results/runs.csv")
    for name in names:
        print(f"Scoring {name} ...", flush=True)
        runs = save_scores(board_path, score_model(ctx, name))

    same = runs[(runs["data"] == config["data"]["path"]) & (runs["metric"] == ctx["selection_metric"])]
    board = (same.sort_values("mean", ascending=False)
             [["model", "tuned", "mean", "untuned", "floor"]]
             .rename(columns={"mean": ctx["selection_metric"]})
             .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nScoreboard ({ctx['selection_metric']}, best first)\n{board}")
    print(f"\nFull board: {board_path}")


if __name__ == "__main__":
    main()
