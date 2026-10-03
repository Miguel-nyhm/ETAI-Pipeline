"""
Entry point for the predictive pipeline.

Run with:
    python main.py

Week 4 flow:
    load config -> load data -> clean (row-preserving) -> drop duplicate rows (train only)
    -> features/target/extras -> carve off the LOCKED test set
    -> (preprocessing + model) as one Pipeline
    -> stratified k-fold CV on the development set (preprocessing re-fit inside each fold)
    -> report: fold table + out-of-fold classification report + fairness audit
    -> refit the final model on all development rows -> save results

The locked test set is set aside and deliberately NOT touched here -- it is reserved
for the single final assessment in a later week.
"""
import yaml
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

from src.data import load_data
from src.preprocessing import (
    clean_dataset,
    drop_duplicate_rows,
    split_features_target,
    split_dev_test,
    build_preprocessor,
)
from src.model import build_model
from src.evaluate import (
    cross_validate_pipeline,
    cv_report,
    oof_classification_report,
    fairness_report,
)
from src.results import save_run


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    # --- load + clean (keeps every row) + drop duplicates (training data only) ---
    df = load_data(config["data"]["path"])
    df = clean_dataset(df, config["diagnostics"])
    df = drop_duplicate_rows(df, config["diagnostics"].get("id_column"))

    # --- features / target / extras, then carve off the locked test set ---
    X, y, extras = split_features_target(
        df, config["data"], config["preprocessing"]["mnar_indicator_sources"]
    )
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = split_dev_test(
        X, y, extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"],
    )

    # --- preprocessing + model = one estimator (nothing fitted yet) ---
    pipeline = Pipeline([
        ("prep", build_preprocessor(config["preprocessing"])),
        ("model", build_model(config["model"])),
    ])

    # --- stratified k-fold CV on the development set ---
    cv_cfg = config["cv"]
    shuffle = cv_cfg.get("shuffle", True)
    cv = StratifiedKFold(
        n_splits=cv_cfg["n_splits"],
        shuffle=shuffle,
        random_state=cv_cfg.get("random_state") if shuffle else None,
    )
    scoring = cv_cfg.get("scoring", "accuracy")
    fold_scores, y_oof = cross_validate_pipeline(
        pipeline, X_dev, y_dev, cv, scoring, n_jobs=cv_cfg.get("n_jobs", 1)
    )

    # --- report: fold table + out-of-fold classification report + fairness audit ---
    report = cv_report(fold_scores, scoring)
    report += "\n" + oof_classification_report(y_dev, y_oof)
    report += "\n" + fairness_report(
        y_dev, y_oof, extras_dev, sensitive_attr=config["data"]["sensitive_attr"]
    )

    # --- the model you'd actually deploy: same pipeline refit on ALL development rows ---
    pipeline.fit(X_dev, y_dev)

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"\nFull results saved to {path}")


if __name__ == "__main__":
    main()
