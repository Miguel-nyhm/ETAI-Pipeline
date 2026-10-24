"""
Entry point for the predictive pipeline.

Run with:
    python main.py

Week 5 flow (single model = config["model"]):
    load -> clean -> drop duplicates -> features/target -> lock the test set
    -> build_pipeline (preprocessing + model)
    -> default-parameter CV   (where week 4 left off)
    -> nested CV of the tuning procedure   (the HONEST tuned score + out-of-fold preds)
    -> tune once on all development rows    (the hyperparameters we keep)
    -> reports (tuning + OOF classification report + fairness)
    -> refit the tuned model on all development rows -> save results

To compare several *tuned* models in one shot, run `python compare_models.py`.
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
from src.evaluate import (
    cross_validate_pipeline,
    cv_report,
    oof_classification_report,
    fairness_report,
)
from src.tuning import nested_cross_validate, tune_pipeline, tuning_report
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

    # --- CV splitters: outer = the usual cv; inner = the tuning folds ---
    cv_cfg, tun = config["cv"], config["tuning"]
    scoring = cv_cfg.get("scoring", "accuracy")
    n_jobs = cv_cfg.get("n_jobs", 1)
    outer_cv = StratifiedKFold(
        n_splits=cv_cfg["n_splits"], shuffle=cv_cfg.get("shuffle", True),
        random_state=cv_cfg.get("random_state") if cv_cfg.get("shuffle", True) else None,
    )
    inner_cv = StratifiedKFold(
        n_splits=tun["n_splits"], shuffle=True, random_state=tun["random_state"]
    )
    search_space = tun["search_spaces"].get(config["model"]["type"])

    # --- 1. default-parameter CV (where week 4 left off) ---
    fs_default, y_oof = cross_validate_pipeline(pipeline, X_dev, y_dev, outer_cv, scoring, n_jobs)
    report = "=== Default hyperparameters (CV) ===\n" + cv_report(fs_default, scoring)

    # Tuning is skipped when it is off in the config, or when this model has no search space
    # (e.g. logistic_regression): we then report and deploy the default model.
    do_tune = tun.get("enabled", True) and search_space is not None
    best_pipeline = pipeline            # default unless tuning replaces it
    # y_oof above is from the default model; tuning overwrites it with the tuned OOF predictions.

    if do_tune:
        # --- 2. nested CV: the honest tuned score + out-of-fold predictions ---
        nested_scores, y_oof = nested_cross_validate(
            pipeline, X_dev, y_dev, outer_cv, inner_cv, scoring,
            search_space, tun["n_trials"], tun["random_state"], n_jobs,
        )
        report += "\n\n=== Tuned (nested CV) ===\n" + cv_report(
            nested_scores[["fold", "train", "validation", "gap"]], scoring
        )

        # --- 3. the hyperparameters we keep: tune once on ALL development rows ---
        best_pipeline, study = tune_pipeline(
            pipeline, X_dev, y_dev, inner_cv, scoring,
            search_space, tun["n_trials"], tun["random_state"], n_jobs,
        )
        report += "\n\n" + tuning_report(study, nested_scores, scoring)

    # --- 4. reports on the out-of-fold predictions (tuned if we tuned, else default) ---
    report += "\n" + oof_classification_report(y_dev, y_oof)
    report += "\n" + fairness_report(
        y_dev, y_oof, extras_dev, sensitive_attr=config["data"]["sensitive_attr"]
    )

    # --- 5. the model you'd deploy: tuned pipeline refit on ALL development rows ---
    best_pipeline.fit(X_dev, y_dev)

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"\nFull results saved to {path}")


if __name__ == "__main__":
    main()
