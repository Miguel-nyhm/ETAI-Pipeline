"""
Saving each run's results to disk.

Two outputs:
  - save_run():    the full human-readable report of one run, to a timestamped .txt in results/.
  - save_scores(): one row per run appended to a CSV scoreboard (week 7), so every model's
                   numbers accumulate in one table you can sort and compare.

Printing to the terminal is fine while you're watching it happen, but it's gone the moment you
scroll past it. These write it down instead, so you can reopen a run or compare two side by side.
"""
import os

import pandas as pd


def save_run(results_dir: str, run: str, config: dict, report_text: str) -> str:
    """
    Writes one run's full report to `results_dir/run_<run>.txt` (the run id is passed in by
    main.py, so the .txt and the scoreboard row share it), created if needed. Returns the path.
    """
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"run_{run}.txt")

    cv = config.get("cv", {})
    header = (
        f"Run: {run}\n"
        f"Model: {config['model']['type']}  params={config['model'].get('params')}\n"
        f"Selection metric: {cv.get('selection_metric')}   metrics: {cv.get('metrics')}\n"
        f"Test size: {config['test_set']['size']}  "
        f"random_state: {config['test_set']['random_state']}\n"
        + "=" * 60 + "\n\n"
    )

    with open(path, "w") as f:
        f.write(header + report_text)
    return path


def save_scores(path: str, row: dict) -> pd.DataFrame:
    """Appends one run to the CSV at `path` (created if needed) and returns all runs."""
    runs = pd.read_csv(path, dtype={"run": str}) if os.path.exists(path) else pd.DataFrame()
    runs = pd.concat([runs, pd.DataFrame([row])], ignore_index=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    runs.to_csv(path, index=False)
    return runs
