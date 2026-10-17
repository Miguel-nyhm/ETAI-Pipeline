"""
Entry point for the predictive pipeline.

Run with:
    python main.py

Week 5 flow:
    load -> clean -> drop duplicates -> features/target -> lock the test set
    -> build_pipeline (preprocessing + model)
    -> nested CV of the whole tuning procedure (honest number + out-of-fold preds)
    -> reports (OOF classification report + fairness)
    -> tune once on all development rows for the kept hyperparameters, refit
    -> save results

The locked test set is set aside and NOT touched here.
"""
import yaml
from sklearn.model_selection import StratifiedKFold

from src.data import load_data
from src.preprocessing import (
    clean_dataset,
    drop_duplicate_rows,
    split_features_target,
    split_dev_test,
)
from src.model import build_pipeline
from src.evaluate import oof_classification_report, fairness_report
from src.tuning import nested_cross_validate, tuning_report, tune_pipeline
from src.results import save_run


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    # --- load + clean + de-duplicate (training data only) ---
    df = load_data(config["data"]["path"])
    df = clean_dataset(df, config["diagnostics"])
    df = drop_duplicate_rows(df, config["diagnostics"].get("id_column"))

    # --- features / target / extras, then lock the test set ---
    X, y, extras = split_features_target(
        df, config["data"], config["preprocessing"]["mnar_indicator_sources"]
    )
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = split_dev_test(
        X, y, extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"],
    )

    # --- preprocessing + model as one estimator ---
    pipeline = build_pipeline(config["preprocessing"], config["model"])

    # --- nested CV: honest score for the tuning procedure ---
    cv_cfg = config["cv"]
    scoring = cv_cfg.get("scoring", "accuracy")
    outer_cv = StratifiedKFold(
        n_splits=cv_cfg["n_splits"],
        shuffle=cv_cfg.get("shuffle", True),
        random_state=cv_cfg.get("random_state") if cv_cfg.get("shuffle", True) else None,
    )
    fold_scores, y_oof = nested_cross_validate(
        pipeline, X_dev, y_dev, outer_cv, config["tuning"], scoring
    )

    report = tuning_report(fold_scores, scoring)
    report += "\n" + oof_classification_report(y_dev, y_oof)
    report += "\n" + fairness_report(
        y_dev, y_oof, extras_dev, sensitive_attr=config["data"]["sensitive_attr"]
    )

    # --- the hyperparameters we keep: tune once on ALL development rows, refit ---
    best_pipeline, study = tune_pipeline(pipeline, X_dev, y_dev, config["tuning"], scoring)
    best_pipeline.fit(X_dev, y_dev)
    report += (f"\nKept hyperparameters (tuned on all development rows): "
               f"{study.best_params}   inner-best {scoring} = {study.best_value:.3f}\n")

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"\nFull results saved to {path}")


if __name__ == "__main__":
    main()
