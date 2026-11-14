## Student Number and Name

Number:20260658

Name:Miguel Guerreiro

## WEEK 9 — Beyond COMPAS

This week the pipeline had to leave COMPAS's binary target behind. Of the teacher's two options (new dataset / new kind of target) I took **Option B: a new kind of target** — predicting COMPAS's own three-level risk label `score_text` (**Low / Medium / High**) instead of binary `two_year_recid`. Same data, same framework; the point was to find every place the pipeline had quietly assumed *two* classes and make it class-count-agnostic, without breaking the binary run.

**How to run it.** The multiclass setup lives in its own config so the original stays untouched:
```bash
python main.py config_multiclass.yaml      # 3-class score_text, writes to results_multiclass/
python main.py                             # original binary run, unchanged
```
`main.py` now reads the config path as an argument (defaulting to `config.yaml`) and maps the string labels to integers 0/1/2 via a new `data.target_labels` key. `config_multiclass.yaml` switches the target to `score_text`, drops `decile_score` and `two_year_recid` (both leak the answer), uses one-hot encoding (which sidesteps per-class target encoding), and ranks on `roc_auc_ovr` with macro-averaged metrics. Everything writes to `results_multiclass/` so the two targets never overwrite each other.

**What broke, why, and how it was fixed.** I changed one thing at a time and ran after each, logging every crash. Three of the bugs were *silent* — they don't raise an error, they just compute the wrong thing on three classes:

| # | What broke | Why | Fix |
|---|---|---|---|
| 1 | XGBoost refused to fit — `ValueError` on string labels | `XGBClassifier` needs integer classes 0/1/2, not `"Low"/"Medium"/"High"` | map the target through `data.target_labels` in `main.py` |
| 2 | *(silent)* out-of-fold probabilities came out as a single column | `predict_proba(...)[:, 1]` hard-codes "the positive class" — fine for 2 classes, wrong for 3 | branch: 1-D `[:, 1]` when 2 classes, the full probability matrix when more (`evaluate.py`, `tuning.py`) |
| 3 | `ValueError: Shape of passed values (3,3), indices imply (2,2)` | the confusion matrix was built for a fixed 2×2 | `argmax` over the probability matrix + labels read from the data (`oof_classification_report`) |
| 4 | `ValueError: expected a 1d array, got an array with shape (5771, 3)` in the fairness report | FPR was defined as "predicted positive while truly negative" — a binary idea | redefine FPR as "predicted the top class while not truly in it", `argmax` the predictions, drop the COMPAS-score comparison (COMPAS has no 3-class column to compare against) |
| 5 | *(silent)* the feature selector's L1 voter only looked at class 0 | `lasso.coef_[0]` is the whole coefficient vector when binary, but only the *first class's* row when multiclass | `np.abs(lasso.coef_).max(axis=0)` — a feature counts if it's non-zero for *any* class |
| 6 | `ValueError: 'liblinear' solver does not support multiclass` | `liblinear` is binary-only | branch the solver: `liblinear` for 2 classes, `saga` for 3+ |
| 6b | *(silent)* `FutureWarning` + "penalty=l1 with l1_ratio=0.0" — L1 was being overridden to L2 | `penalty="l1"` is deprecated in scikit-learn 1.8 and was silently ignored | set `l1_ratio=1.0` instead, the supported way to ask for pure L1 |

The thread running through all of these is a single buried assumption: that a classification problem has *one positive class*. It hides in the probability slice (`[:, 1]`), the 0.5 threshold, the 2×2 confusion matrix, the fairness FPR, and even the L1 selector's `coef_[0]`. Four of those crash loudly on three classes; two don't — they'd have handed me a plausible-looking but wrong number. The hardest part of "beyond COMPAS" wasn't reading the error messages, it was trusting that the *silent* spots existed and going to find them.

**Does it work?** XGBoost on the three-level target, same five folds, floor is the majority-class dummy:

<!-- selection metric: roc_auc_ovr · onehot · 30 trials · 5 outer x 5 inner folds · random_state=42 -->
| XGBoost (multiclass) | roc_auc_ovr | accuracy | macro-F1 |
|---|---|---|---|
| No-model floor (dummy) | 0.500 | — | — |
| Default hyperparameters | 0.749 ± 0.011 | 0.619 | 0.508 |
| Tuned (nested CV) | **0.757 ± 0.009** | **0.630** | 0.519 |

roc_auc_ovr clears the 0.75 target and accuracy sits right at 0.63, both comfortably above the 0.500 floor. Macro-F1 (0.519) is the one number that stays low, and the confusion matrix says exactly why — it's one class dragging the average down:

| true ↓ / predicted → | Low | Medium | High | recall | F1 |
|---|---|---|---|---|---|
| **Low** (3118) | 2816 | 207 | 95 | 0.90 | 0.78 |
| **Medium** (1531) | 929 | 349 | 253 | 0.23 | **0.30** |
| **High** (1122) | 384 | 267 | 471 | 0.42 | 0.49 |

**Medium is by far the hardest class** — recall 0.23, meaning the model catches barely one truly-Medium person in four, and pushes 61% of them (929 of 1531) into Low. That's the squishy-middle problem: Low and High sit at the ends of the risk scale with distinctive feature signatures (few priors + older vs many priors + young), but Medium has no profile of its own, so the model — rewarded for overall accuracy — mostly collapses it onto the nearest confident class. Because macro-F1 weights all three classes equally, that one weak class alone holds the macro score under the Low/High average. The fairness gap also survives the reframing: the predicted-High false-positive rate is 0.11 for African-American defendants against 0.05 for Caucasian — the same ~2× disparity the binary model showed.

One honest detail on the scoreboard: the tuned run's line reads `vote(min_votes=1)`, not the `2` in the config. That isn't a mismatch — nested CV *tunes* `min_votes` alongside the model and chose 1 (keep more features), which is the whole point of tuning it honestly rather than fixing it by hand.

**The binary run is untouched — proven, not assumed.** Every fix branches on the number of classes, so the original COMPAS pipeline had to come out byte-for-byte identical. It does: re-running the binary feature-selection config after all six changes reproduces the pre-Week-9 run to full floating-point precision — every fold, every tuned hyperparameter (mean `0.7249663763989602`; folds `0.72386 / 0.70295 / 0.73152 / 0.72335 / 0.73415`, identical in both runs). That byte-identical match is the real proof that "make it handle three classes" didn't quietly cost anything on two.

## WEEK 8

Added **XGBoost** (`XGBClassifier`) and **feature selection** as a learned Pipeline step. The selector (`VoteSelector`) keeps the features that at least `min_votes` of three models back — a random forest and gradient boosting (above-average importance) and an L1 logistic regression (non-zero coefficient). It is fitted on each fold's training rows only (never before the split), and `min_votes` is tuned together with the model under nested CV. XGBoost adds brakes the earlier boosters don't have — `gamma` (minimum gain to split), `min_child_weight` (minimum leaf weight) and `reg_alpha`/`reg_lambda` (L1/L2) — plus native missing-value handling.

**Does feature selection pay off?** XGBoost tuned, with and without selection, on the same five folds:

<!-- selection metric: roc_auc · 30 trials · 5 outer x 5 inner folds · random_state=42 -->
| XGBoost (tuned, nested CV) | ROC-AUC | Features kept | Chosen hyperparameters (abridged) |
|---|---|---|---|
| With selection (`min_votes` tuned → 1) | 0.725 ± 0.013 | 2 of 10: `age`, `priors_count` | max_depth=3, lr=0.23, min_child_weight=1.8, gamma=4.0, reg_lambda=0.08, reg_alpha=0.05, subsample=0.57, colsample=0.63, n_estimators=92 |
| Without selection | 0.728 ± 0.015 | all 10 | max_depth=7, lr=0.11, min_child_weight=28, gamma=4.1, reg_lambda=0.24, subsample=0.85, colsample=0.56, n_estimators=94 |

Corrected paired t-test (Nadeau–Bengio), with − without: **mean difference −0.003, wins 2/5 folds, t = −0.89, p = 0.42**.

Feature selection drops 8 of the 10 columns and the tuned score falls only from 0.728 to 0.725 — a 0.003 gap, smaller than the ~0.015 fold-to-fold std, and the corrected test confirms it isn't significant (p = 0.42). So the two-feature model (`age` + `priors_count`) is statistically **as good as** the full one. Even with `min_votes` tuned down to 1, only those two features got any votes at all — the other eight carry almost no signal. I'd keep selection as a legitimate option when a simpler, more explainable model matters (two inputs instead of ten, at no measurable cost in ROC-AUC), but I wouldn't claim it improves accuracy: on COMPAS the signal is just "more priors, younger," which XGBoost finds with or without the selector. One telling detail — without selection, XGBoost leans harder on its own brakes to tame the noise columns (`min_child_weight` jumps from ~2 to 28, `max_depth` from 3 to 7), regularization doing the job the selector would have done up front.

## WEEK 7

Evaluation moved from a single accuracy number to **eight metrics computed from out-of-fold probabilities**, with **ROC-AUC** as the selection metric (what tuning maximises and what the board ranks by). Every model so far is scored on the same five folds, untuned and tuned (nested CV), against a no-model floor (a dummy). `results/runs.csv` keeps one row per run.

<!-- selection metric: roc_auc · 8 metrics · 5 outer x 5 inner folds · 30 trials · random_state=42 -->
| Model | Untuned ROC-AUC | Tuned ROC-AUC (nested) | Chosen hyperparameters |
|---|---|---|---|
| Dummy (floor) | 0.500 | — | — |
| Logistic regression | 0.718 | — | — |
| Naive Bayes | 0.685 | — | — |
| Decision tree | 0.610 | 0.717 ± 0.017 | max_depth=5, min_samples_leaf=115, criterion=gini |
| Random forest | 0.684 | 0.725 ± 0.015 | max_depth=24, min_samples_leaf=34, max_features=0.5 |
| Oblique random forest | 0.685 | 0.726 ± 0.015 | tree__max_depth=12, tree__min_samples_leaf=87, project__n_components=8 |
| AdaBoost | 0.723 | 0.727 ± 0.015 | estimator__max_depth=4, n_estimators=55, learning_rate=0.167 |
| Gradient boosting | 0.727 | 0.727 ± 0.014 | n_estimators=426, learning_rate=0.012, max_depth=3, min_samples_leaf=2, subsample=0.766 |
| HistGradientBoosting | 0.717 | 0.728 ± 0.016 | max_iter=92, learning_rate=0.095, max_leaf_nodes=6, min_samples_leaf=84, l2_regularization=0.0008 |

**Selection metric — ROC-AUC.** I rank on ROC-AUC because it scores how well the model *orders* people by risk across every threshold, rather than its accuracy at one arbitrary 0.5 cut-off, and because it isn't distorted by the class imbalance (~45% reoffend) the way raw accuracy is. It's also the one metric every model exposes through `predict_proba`, so the whole board compares on equal terms; the operating threshold (recall vs precision) is a separate, later choice.

The three boosters sit at the top (HistGradientBoosting 0.728, gradient boosting and AdaBoost 0.727), but their lead over tuned logistic regression (0.718) and the random forests (0.725–0.726) is smaller than the ~0.015 fold-to-fold std, so no model is convincingly ahead — I'd report **HistGradientBoosting at 0.728 ± 0.016**, while noting a plain logistic regression is within noise of it and far cheaper to fit and explain. Tuning helps the high-variance models most (the lone decision tree jumps 0.610 → 0.717, the forests ~0.68 → ~0.73) and barely moves the already-regularised boosters. Every model clears the 0.500 floor comfortably; Naive Bayes (0.685) is the weakest real model, as its "features independent given the class" assumption is wrong for these correlated counts.

## WEEK 5/6

<!-- nested CV, accuracy, 30 trials, 5 outer x 5 inner folds, random_state=42 -->
| Model | Default: CV mean ± std (gap) | Tuned: nested CV mean ± std (gap) | Tuning score (best trial) | Chosen hyperparameters |
|---|---|---|---|---|
| Logistic regression | 0.672 ± 0.013 (+0.003) | — | — | — |
| Decision tree | 0.610 ± 0.018 (+0.085) | 0.672 ± 0.017 (+0.013) | 0.679 | max_depth=6, min_samples_leaf=101, criterion=gini |
| Random forest | 0.650 ± 0.019 (+0.082) | 0.681 ± 0.015 (+0.009) | 0.683 | max_depth=22, min_samples_leaf=42, max_features=0.5 |
| Oblique random forest | 0.647 ± 0.011 (+0.114) | 0.679 ± 0.013 (+0.013) | 0.683 | estimator__tree__max_depth=6, estimator__tree__min_samples_leaf=3, estimator__project__n_components=5 |
| AdaBoost | 0.676 ± 0.012 (+0.006) | 0.682 ± 0.017 (+0.004) | 0.684 | estimator__max_depth=4, n_estimators=220, learning_rate=0.0251 |

AdaBoost gives the best honest score (0.682 tuned nested CV), with the random forest (0.681) and oblique forest (0.679) just behind; all three beat the tuned decision tree and logistic regression (~0.672). But every lead is smaller than the ~0.013–0.017 fold-to-fold std, so no model is convincingly ahead — I'd report **AdaBoost at 0.682 ± 0.017**, noting its edge over logistic regression is within the noise. The chosen hyperparameters match how each model controls overfitting: the lone decision tree must stay shallow (`max_depth=6`) or it overfits, whereas the random forest can afford deep trees (`max_depth=22`) because averaging over bootstrap samples tames the variance; AdaBoost prefers slightly deeper base trees (depth 4) but a very small learning rate (0.025), i.e. many small corrective steps. The oblique forest does **not** beat the ordinary one here: COMPAS's features are counts and encoded categories where the signal is mostly "more priors, younger" — something axis-aligned splits already capture — so random linear combinations buy nothing.


## WEEK 4

Evaluation moved from a single train/test split (weeks 2–3) to stratified 5-fold cross-validation, with a 20% test set now locked away. The difference is reliability: a single split gave one number that swung several points depending on the seed, while CV reports a stable mean ± std over five folds. Under CV, logistic regression scores 0.672 ± 0.013 (gap +0.003), the decision tree 0.610 ± 0.018 (gap +0.085), and the new random forest 0.650 ± 0.018 (gap +0.083), all comfortably above the dummy baseline's 0.549. Logistic regression remains the best model and generalizes cleanly; the tree and forest overfit (large train–validation gaps) and the forest, though more powerful in principle, doesn't beat the simple linear model out of the box. The headline is that cross-validation didn't change which model wins, but it tells us how much to trust that answer — and that the single-split scores from earlier weeks were one lucky (or unlucky) draw, not a dependable estimate.

## WEEK 3

After this week's changes we see a drop in test accuracy in both logistic regression and decision trees. This is actually a healthier result since after cleaning the data, we have dropped duplicate rows which were probably inflating the week 2's test accuracy. We also have fewer rows after dropping all the new NaN rows. And we are still missing the preprocessing of the data which actually might improve the accuracy. After canonicalizing, we do get real groups now and clean comparisons between our model's fpr and COMPAS's fpr scores.

Logistic regression: test accuracy 0.655 (near zero train-test gap)
Decision tree: test accuracy 0.607 (0.192 gap)

Logistic regression still is the best model and the decision tree still overfits. Cleaning the data does not fix overfitting.

## WEEK 2

Logistic regression is the better model, and the decision trees model overfits. We got a 0.829 train accuracy on the trees but a test accuracy of 0.626. On the other hand logistic regression sits at 0.679 train/0.678 test with almost no gap.

The tree grows until it memorises the training data, so the high training score doesn't survive contact with unseen data. Logistic regression is a simpler and more stable model, so it generalises.


## Baseline Predictive Pipeline -- ETAI

This is the **starting point** for your semester project: a small but *complete* predictive pipeline -- every piece a real project needs (entry point, config, data loading, preprocessing, model, evaluation), just kept as simple as possible for now.

The task: predict two-year recidivism using ProPublica's COMPAS
dataset -- the data behind a real 2016 investigation into a risk-
assessment algorithm actually used by US courts to help inform bail and sentencing decisions. See `data/README.md` for the full problem description and a complete data dictionary before you start.

It has some **deliberately weak spots**. Part of your work this
semester is finding them and making them better -- see the pipeline progress table below, which tracks what changes and why as the weeks
go on.

## Project structure

```
.
├── main.py                # entry point: run the whole pipeline (optional config path as 1st argument)
├── config.yaml             # all tunable settings live here (binary target: two_year_recid)
├── config_multiclass.yaml  # week 9: 3-class score_text target (Low/Medium/High)
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── preprocessing.py    # cleaning + train/test split
│   ├── model.py             # model construction
│   ├── evaluate.py         # accuracy metrics + fairness check
│   └── results.py          # saves each run's report to disk
├── results/                # created automatically -- one file per run (not tracked in git)
├── results_multiclass/     # week 9: same, for the 3-class runs (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md            # problem description + full data dictionary
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |
| 6–7 | Boosting & probability-based metrics | `gradient_boosting`, `hist_gradient_boosting` and `naive_bayes` added to `src/model.py`; evaluation rewritten to score out-of-fold **probabilities** on eight metrics with `roc_auc` as the selection metric (`src/evaluate.py`, `src/tuning.py`); a no-model floor (dummy) and a persistent scoreboard at `results/runs.csv` (`src/results.py`, `main.py`); `compare_models.py` fills the whole board in one run |
| 8 | XGBoost & feature selection | `xgboost` (`XGBClassifier`) added to `src/model.py`; a learned `VoteSelector` step between preprocessing and the model, `build_pipeline` now taking a selection config and a cache `memory` folder; `corrected_paired_test` added to `src/evaluate.py`; `config.yaml` gains the `xgboost` search space and a `feature_selection` section; `main.py` tunes `select__min_votes` with the model, prints the features kept, and records the selection setting on the scoreboard (`src/results.py`) |
| 9 | Beyond COMPAS — a new kind of target | `main.py` reads the config path as a command-line argument and maps string labels to integers via a new `data.target_labels` key; `config_multiclass.yaml` added for the 3-class `score_text` target (one-hot encoding, `roc_auc_ovr`, macro metrics, output to `results_multiclass/`); `src/evaluate.py`, `src/tuning.py` and `src/model.py` made class-count-agnostic (full probability matrix, `argmax` confusion matrix, top-class fairness FPR, multiclass-safe L1 selector with the `saga` solver) — every change branches on the number of classes so the binary run stays byte-identical |

## Environment setup

You only need to do this once per machine.

### macOS / Linux
```bash
python3 -m venv venv                 # creates an isolated Python environment in a folder called "venv"
source venv/bin/activate             # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```

### Windows -- PowerShell
```powershell
python -m venv venv                  # creates an isolated Python environment in a folder called "venv"
venv\Scripts\activate                # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```
If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows -- cmd.exe
Same three steps as above, just with cmd's own activation command:
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Once the environment is active you'll see `(venv)` at the start of your prompt. To leave it later, run `deactivate` (same command on every OS).

### Every time after the first

Creating the environment and installing packages only needs to happen once, ever. Every other time you sit down to work -- a new terminal window, the next practical class, tomorrow -- you don't repeat any of the steps above. From the project's root folder, you just need to:

**macOS / Linux**
```bash
source venv/bin/activate
python main.py
```

**Windows**
```powershell
venv\Scripts\activate
python main.py
```

That's it -- activate, then run. If you don't see `(venv)` at the start of your prompt, the environment isn't active and `python main.py` may use the wrong Python (or fail to find a package) entirely.

## Running the pipeline

With the environment active (see above), from the project's root
folder, on any OS:
```bash
python main.py
```

This loads `config.yaml`, loads and preprocesses the data, trains the model, and prints:
- **train accuracy and test accuracy, side by side.** Comparing the two is how you catch overfitting: if the model looks much better on the data it was trained on than on data it's never seen, it has memorised rather than learned something that generalises. 
- a classification report on the test set
- a false-positive-rate-by-race comparison between our model and
  COMPAS's own score

All of this is also saved to a timestamped file in `results/` (e.g.`results/run_20260916_143012.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Dataset

See `data/README.md`.
