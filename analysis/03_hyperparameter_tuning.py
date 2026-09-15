# %% [markdown-style header — see module docstring below for the full narrative]
"""
================================================================================
RADCURE ML — Milestone 3A/3B: Controlled Hyperparameter Tuning
================================================================================

Author:      Carlo Babo
Project:     Applied Machine Learning — RADCURE Clinical Dataset

--------------------------------------------------------------------------------
WHAT THIS SCRIPT COVERS
--------------------------------------------------------------------------------
Milestone 2 (frozen) compared five DEFAULT-hyperparameter model families
under a fixed 5-fold stratified cross-validation design and shortlisted
three for tuning: Logistic Regression (best default), Random Forest
(second-best on the primary metric), and RBF SVC (weaker ROC-AUC than
Random Forest but a better Average Precision, with `C`/`gamma` left at
scikit-learn defaults). This script runs one controlled `GridSearchCV`
per shortlisted family, under the IDENTICAL fixed CV design, and compares
each tuned result against its own frozen Milestone-2 default. It does not
tune a decision threshold, calibrate probabilities, or attempt stacking,
and it does not touch the held-out test set anywhere below.

--------------------------------------------------------------------------------
MILESTONES 1 AND 2 ARE FROZEN -- NOTHING BELOW CHANGES THEM
--------------------------------------------------------------------------------
This script does not redefine, and does not import as a module, Milestone 1
(`analysis/01_project_foundation.py`) or Milestone 2
(`analysis/02_baseline_modeling.py`). It reconstructs the confirmed
deterministic modelling cohort and the confirmed 80/20 split using the SAME
reusable functions those milestones used (`cleaning.*`, `target.*`,
`config.*`, `preprocessing.build_preprocessor`), because that reconstruction
is fully deterministic given a fixed `random_state` -- it is not a new
design decision. Frozen and unchanged here: target definition, target
eligibility, cohort definition, the PRIMARY predictor set, deterministic
cleaning rules, the primary 80/20 stratified split (random_state=42), the
preprocessing specification, and held-out test-set policy (reconstructed
ONLY to verify its confirmed size and class counts, then never touched
again).

--------------------------------------------------------------------------------
GOVERNING RULES FOR THIS MILESTONE
--------------------------------------------------------------------------------
  - Exactly three model families are tuned, each over a FIXED, pre-declared
    grid (Section 3): Logistic Regression (10 combinations), Random Forest
    (96), RBF SVC (60). No grid is changed after seeing a result.
  - The SAME fixed StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    and the SAME three metrics as Milestone 2 (roc_auc PRIMARY,
    average_precision SECONDARY, balanced_accuracy SUPPLEMENTARY) are used.
  - `GridSearchCV(..., refit="roc_auc")`: hyperparameter selection is made
    by the PRIMARY metric only. Average Precision and Balanced Accuracy for
    a model are always read off the SAME selected row -- never
    independently re-optimised per metric.
  - No threshold tuning, no probability calibration, no
    `SVC(probability=True)`, no stacking. No test-set metric of any kind.
  - Search-boundary hits on continuous/log-scale hyperparameters (Logistic
    `C`; SVC `C`, `gamma`) are flagged for review, not automatically
    followed by an expanded search.

Confirmed project environment: `ml` conda environment
(/home/c/miniconda3/envs/ml/bin/python) -- pandas 3.0.5, numpy 2.4.6,
scikit-learn 1.9.0, openpyxl 3.1.5, pytest 9.1.1.
"""

# %%
# =============================================================================
# SECTION 1 — Scope and frozen Milestone-2 benchmark
# =============================================================================
# Objective:
#   Import only what this milestone needs, and print the frozen
#   Milestone-2 default-hyperparameter results as the fixed comparison
#   benchmark every tuned result below is measured against.

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sklearn.model_selection import train_test_split  # noqa: E402

from radcure import cleaning, config, leakage, target, tuning  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 100)


def section(title: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n{title}\n{bar}")


# Frozen Milestone-2 default-hyperparameter benchmark (confirmed values;
# not recomputed here -- Milestone 2 is frozen and is not re-run).
DEFAULT_RESULTS: dict[str, dict[str, float]] = {
    "Logistic Regression": {
        "roc_auc_mean": 0.7897, "roc_auc_std": 0.0270,
        "average_precision_mean": 0.4404, "average_precision_std": 0.0310,
        "balanced_accuracy_mean": 0.5879, "balanced_accuracy_std": 0.0253,
    },
    "Random Forest": {
        "roc_auc_mean": 0.7554, "roc_auc_std": 0.0237,
        "average_precision_mean": 0.3815, "average_precision_std": 0.0333,
        "balanced_accuracy_mean": 0.5783, "balanced_accuracy_std": 0.0201,
    },
    "RBF SVC": {
        "roc_auc_mean": 0.7285, "roc_auc_std": 0.0239,
        "average_precision_mean": 0.4135, "average_precision_std": 0.0300,
        "balanced_accuracy_mean": 0.5324, "balanced_accuracy_std": 0.0066,
    },
}

section("SECTION 1 — Scope and frozen Milestone-2 benchmark")
print("Milestone 3A/3B: controlled hyperparameter tuning of the three Milestone-2")
print("shortlisted model families (Logistic Regression, Random Forest, RBF SVC).")
print("\nFrozen Milestone-2 DEFAULT-hyperparameter benchmark (not recomputed here):")
print(pd.DataFrame(DEFAULT_RESULTS).T.round(4).to_string())


# %%
# =============================================================================
# SECTION 2 — Reconstruct confirmed training data
# =============================================================================
# Objective:
#   Deterministically reproduce the exact X_train/y_train/X_test/y_test
#   from Milestones 1-2, using the SAME reusable functions, WITHOUT
#   repeating their audit narrative or diagnostic output.
#
# Rationale:
#   Every step below is a pure function of the immutable raw file plus
#   fixed constants (`config.PRIMARY_FEATURES`, `config.RANDOM_STATE`,
#   `config.TEST_SIZE`) -- given the same inputs it reproduces the
#   confirmed Milestone-1 split configuration and counts. This is
#   reconstruction of an already-confirmed result, not a new design
#   decision, so it is deliberately compact (identical in substance to
#   `02_baseline_modeling.py` Sections 2-3, duplicated here rather than
#   imported, per project convention that analysis scripts do not import
#   one another).

section("SECTION 2 — Reconstructing confirmed training data")

raw_df = cleaning.load_raw_clinical()
target_df = target.build_target(raw_df)
eligible_mask = target_df[config.ELIGIBILITY_COLUMN]

clinical_df = raw_df.copy()
clinical_df["ECOG PS"] = cleaning.clean_ecog_ps(raw_df["ECOG PS"])
clinical_df["Smoking PY"] = cleaning.clean_smoking_py(raw_df["Smoking PY"])
clinical_df["Smoking Status"] = cleaning.clean_smoking_status(raw_df["Smoking Status"])
clinical_df["Ds Site"] = cleaning.clean_ds_site(raw_df["Ds Site"])
clinical_df["T"] = cleaning.clean_t_category(raw_df["T"])
clinical_df["N"] = cleaning.clean_n_category(raw_df["N"])
clinical_df = clinical_df.join(target_df)

cohort_df = clinical_df.loc[eligible_mask].reset_index(drop=True)
X = cohort_df[config.PRIMARY_FEATURES].copy()
y = cohort_df[config.TARGET_COLUMN].astype(int)
patient_id = cohort_df[config.PATIENT_ID_COLUMN].copy()

leakage.assert_no_excluded_column_in_frame(X.columns)

X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, patient_id,
    test_size=config.TEST_SIZE,
    stratify=y,
    random_state=config.RANDOM_STATE,
)

print(f"Eligible cohort reconstructed: n = {len(X)}")
print(f"Train: n={len(X_train)}  events={int(y_train.sum())}  non-events={int((y_train == 0).sum())}")
print(f"Test : n={len(X_test)}  events={int(y_test.sum())}  non-events={int((y_test == 0).sum())}  "
      f"(reconstructed for verification ONLY -- see integrity check below)")

# --- Integrity check against the confirmed Milestone-1/2 counts ------------
assert list(X_train.columns) == config.PRIMARY_FEATURES
assert len(X_train) == config.EXPECTED_TRAIN_N
assert int(y_train.sum()) == config.EXPECTED_TRAIN_EVENTS
assert int((y_train == 0).sum()) == config.EXPECTED_TRAIN_NONEVENTS
assert len(X_test) == config.EXPECTED_TEST_N
assert int(y_test.sum()) == config.EXPECTED_TEST_EVENTS
assert int((y_test == 0).sum()) == config.EXPECTED_TEST_NONEVENTS
assert set(id_train) & set(id_test) == set()

print(f"[OK] Training set reconstructed exactly: n={config.EXPECTED_TRAIN_N}, "
      f"events={config.EXPECTED_TRAIN_EVENTS}, non-events={config.EXPECTED_TRAIN_NONEVENTS}.")
print(f"[OK] Held-out test set reconstructed exactly: n={config.EXPECTED_TEST_N}, "
      f"events={config.EXPECTED_TEST_EVENTS}, non-events={config.EXPECTED_TEST_NONEVENTS}.")
print("[OK] Train/test patient_id sets are disjoint.")

# ------------------------------------------------------------------------------
# >>> FROM THIS POINT ON, X_test / y_test / id_test ARE NOT REFERENCED AGAIN. <<<
# They were reconstructed only to verify the frozen split's identity above.
# No predictor distribution in X_test is inspected, nothing is fitted on
# it, nothing is predicted from it, and no test-set metric is computed
# anywhere in this script.
# ------------------------------------------------------------------------------
del X_test, y_test, id_test

# Result / Interpretation:
#   All six confirmed counts match exactly, and the train/test patient
#   identifiers are disjoint -- the reconstruction reproduces the confirmed
#   Milestone-1 split configuration and counts.
# Decision:
#   `X_test`/`y_test`/`id_test` are deleted from the namespace as a hard
#   guard against accidental later use; only `X_train`/`y_train` are used
#   from here on.
# Next step:
#   Fix the tuning design and search spaces (Section 3) before any search
#   is run.


# %%
# =============================================================================
# SECTION 3 — Tuning design and search spaces
# =============================================================================
# Objective:
#   Fix, in one place and before any search is run, the CV design, the
#   scoring metrics, the selection rule, and the exact search spaces for
#   all three model families.
#
# Rationale:
#   Reusing the IDENTICAL StratifiedKFold and metric set as Milestone 2
#   (imported from `radcure.modeling`, not re-implemented in
#   `radcure.tuning`) makes the tuned results directly comparable to the
#   frozen defaults -- any change in mean score reflects the hyperparameter
#   search, not a change in how models are scored. `refit="roc_auc"` means
#   hyperparameter selection uses the PRIMARY metric only; Average
#   Precision and Balanced Accuracy are read off the SAME selected
#   configuration afterwards, never independently optimised.

section("SECTION 3 — Tuning design and search spaces")

cv = tuning.build_cv_splitter()
print(f"CV splitter: StratifiedKFold(n_splits={cv.get_n_splits()}, shuffle={cv.shuffle}, "
      f"random_state={cv.random_state}) -- identical to Milestone 2.")
print(f"Scoring: {tuning.CV_SCORING}")
print(f"Selection rule: refit={tuning.REFIT_METRIC!r} (PRIMARY metric).")

grids = {
    "Logistic Regression": tuning.LOGISTIC_REGRESSION_PARAM_GRID,
    "Random Forest": tuning.RANDOM_FOREST_PARAM_GRID,
    "RBF SVC": tuning.RBF_SVC_PARAM_GRID,
}
for name, grid in grids.items():
    print(f"\n{name} search space ({tuning.grid_size(grid)} combinations):")
    for param, values in grid.items():
        print(f"   {param:28s} {values}")

assert tuning.grid_size(tuning.LOGISTIC_REGRESSION_PARAM_GRID) == 10
assert tuning.grid_size(tuning.RANDOM_FOREST_PARAM_GRID) == 96
assert tuning.grid_size(tuning.RBF_SVC_PARAM_GRID) == 60
print("\n[OK] Grid sizes match the confirmed design: Logistic Regression=10, "
      "Random Forest=96, RBF SVC=60.")

print(
    "\nParallelism note (Random Forest): RandomForestClassifier is fixed with "
    "n_jobs=-1 (parallelises tree-building across all cores on EACH fit). "
    "GridSearchCV for Random Forest therefore uses n_jobs=1 (sequential over "
    "the 96 x 5 = 480 fits) to avoid the two levels of parallelism "
    "oversubscribing the same cores simultaneously. Logistic Regression and "
    "RBF SVC fit a single model per call with no internal parallelism, so "
    "their GridSearchCV uses n_jobs=-1 instead."
)

results: dict[str, dict[str, object]] = {}
searches: dict[str, object] = {}

# Next step:
#   Run the Logistic Regression search (Section 4).


# %%
# =============================================================================
# SECTION 4 — Logistic Regression tuning
# =============================================================================
# Objective:
#   Search `C` (inverse regularisation strength) and `class_weight` for
#   Logistic Regression, keeping L2 regularisation and the scikit-learn
#   default solver unchanged from Milestone 2.

section("SECTION 4 — Logistic Regression tuning")

t0 = time.time()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    lr_search = tuning.build_logistic_regression_search()
    lr_search.fit(X_train, y_train)
elapsed = time.time() - t0
searches["Logistic Regression"] = lr_search
results["Logistic Regression"] = tuning.summarize_best(lr_search)

print(f"Fitted {tuning.grid_size(tuning.LOGISTIC_REGRESSION_PARAM_GRID)} candidates "
      f"x {cv.get_n_splits()} folds in {elapsed:.1f}s.")
print(f"Best params: {lr_search.best_params_}")
s = results["Logistic Regression"]
print(f"   roc_auc            mean={s['roc_auc_mean']:.4f}  std={s['roc_auc_std']:.4f}")
print(f"   average_precision  mean={s['average_precision_mean']:.4f}  std={s['average_precision_std']:.4f}")
print(f"   balanced_accuracy  mean={s['balanced_accuracy_mean']:.4f}  std={s['balanced_accuracy_std']:.4f}")
if caught:
    for w in caught:
        print(f"   [WARNING] {w.category.__name__}: {w.message}")
else:
    print("   [OK] No warnings raised during the search.")

lr_flags = tuning.flag_boundary_hits(
    lr_search.best_params_, tuning.LOGISTIC_REGRESSION_PARAM_GRID, ("classifier__C",)
)
if lr_flags:
    print("   [BOUNDARY FLAG]", *lr_flags)
else:
    print("   [OK] Selected C is not at a grid boundary.")

# Next step:
#   Run the Random Forest search (Section 5).


# %%
# =============================================================================
# SECTION 5 — Random Forest tuning
# =============================================================================
# Objective:
#   Search `max_depth`, `min_samples_leaf`, `max_features` and
#   `class_weight` for Random Forest, with `n_estimators=500`,
#   `random_state=42` and `n_jobs=-1` fixed.
#
# Runtime note:
#   96 candidates x 5 folds = 480 individual 500-tree forest fits, run
#   sequentially at the GridSearchCV level (see Section 3's parallelism
#   note) -- this is the slowest of the three searches in this milestone.

section("SECTION 5 — Random Forest tuning")

t0 = time.time()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    rf_search = tuning.build_random_forest_search()
    rf_search.fit(X_train, y_train)
elapsed = time.time() - t0
searches["Random Forest"] = rf_search
results["Random Forest"] = tuning.summarize_best(rf_search)

print(f"Fitted {tuning.grid_size(tuning.RANDOM_FOREST_PARAM_GRID)} candidates "
      f"x {cv.get_n_splits()} folds in {elapsed:.1f}s.")
print(f"Best params: {rf_search.best_params_}")
s = results["Random Forest"]
print(f"   roc_auc            mean={s['roc_auc_mean']:.4f}  std={s['roc_auc_std']:.4f}")
print(f"   average_precision  mean={s['average_precision_mean']:.4f}  std={s['average_precision_std']:.4f}")
print(f"   balanced_accuracy  mean={s['balanced_accuracy_mean']:.4f}  std={s['balanced_accuracy_std']:.4f}")
if caught:
    for w in caught:
        print(f"   [WARNING] {w.category.__name__}: {w.message}")
else:
    print("   [OK] No warnings raised during the search.")

print(f"   (Descriptive note, not a formal boundary flag per Section 11's scope: "
      f"max_depth={rf_search.best_params_['classifier__max_depth']}, "
      f"min_samples_leaf={rf_search.best_params_['classifier__min_samples_leaf']}, "
      f"max_features={rf_search.best_params_['classifier__max_features']!r}.)")

# Next step:
#   Run the RBF SVC search (Section 6).


# %%
# =============================================================================
# SECTION 6 — RBF SVC tuning
# =============================================================================
# Objective:
#   Search `C`, `gamma` and `class_weight` for the RBF SVC. `probability`
#   is never set (stays at its scikit-learn default), so ROC-AUC and
#   Average Precision continue to be scored via `decision_function`,
#   matching Milestone 2's baseline configuration.

section("SECTION 6 — RBF SVC tuning")

t0 = time.time()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    svc_search = tuning.build_rbf_svc_search()
    svc_search.fit(X_train, y_train)
elapsed = time.time() - t0
searches["RBF SVC"] = svc_search
results["RBF SVC"] = tuning.summarize_best(svc_search)

print(f"Fitted {tuning.grid_size(tuning.RBF_SVC_PARAM_GRID)} candidates "
      f"x {cv.get_n_splits()} folds in {elapsed:.1f}s.")
print(f"Best params: {svc_search.best_params_}")
s = results["RBF SVC"]
print(f"   roc_auc            mean={s['roc_auc_mean']:.4f}  std={s['roc_auc_std']:.4f}")
print(f"   average_precision  mean={s['average_precision_mean']:.4f}  std={s['average_precision_std']:.4f}")
print(f"   balanced_accuracy  mean={s['balanced_accuracy_mean']:.4f}  std={s['balanced_accuracy_std']:.4f}")
if caught:
    for w in caught:
        print(f"   [WARNING] {w.category.__name__}: {w.message}")
else:
    print("   [OK] No warnings raised during the search.")

svc_flags = tuning.flag_boundary_hits(
    svc_search.best_params_, tuning.RBF_SVC_PARAM_GRID, ("classifier__C", "classifier__gamma")
)
selected_gamma = svc_search.best_params_["classifier__gamma"]
if isinstance(selected_gamma, str):
    print(f"   [NOTE] Selected gamma={selected_gamma!r} is a scikit-learn heuristic, "
          f"not a numeric grid value -- it has no 'edge' to flag.")
if svc_flags:
    print("   [BOUNDARY FLAG]", *svc_flags)
else:
    print("   [OK] No numeric C/gamma selection is at a grid boundary.")
assert not (hasattr(svc_search.best_estimator_.named_steps["classifier"], "predict_proba"))
print("   [OK] Selected SVC has no predict_proba (probability was never enabled).")

first_stage_svc = results["RBF SVC"]

# Result / Interpretation:
#   The first-stage search selected C at the UPPER edge of its grid, which
#   means the grid -- not the data -- may have limited the result. That is
#   resolved by exactly one local refinement in Section 6b.
# Next step:
#   Run the single, final SVC refinement (Section 6b).


# %%
# =============================================================================
# SECTION 6b — RBF SVC boundary refinement (second and FINAL stage)
# =============================================================================
# Objective:
#   Resolve the first-stage upper-boundary hit on `C` with ONE local
#   second-stage search: `C` extended upward past 100, `gamma` refined
#   locally around the selected 0.001.
#
# Methodological rationale and its cost:
#   A selected value sitting on the edge of its grid means the search space,
#   not the data, may have bounded the result -- so the first-stage SVC
#   number cannot be read as "the best this model family can do". One
#   local extension is the proportionate response.
#
#   The cost is stated openly: this is a SECOND consultation of the same
#   training folds for the same model family, and it is applied to only one
#   of the three families. The SVC therefore receives more post-hoc
#   adaptation opportunity than Logistic Regression and Random Forest,
#   whose grids were searched once each. Any SVC-vs-LR/RF gap below
#   requires especially cautious reading in light of that asymmetry, and it
#   is one more reason the training-CV numbers here are model-selection
#   estimates rather than generalization estimates (see Section 9b).
#
#   Pre-declared stopping rule: there is NO third stage. If C=3000.0 (the
#   new upper edge) is selected, that is FLAGGED as a residual, accepted
#   limitation and the search stops anyway.

section("SECTION 6b — RBF SVC boundary refinement (second and FINAL stage)")

print("Refinement grid (second and final stage):")
for param, values in tuning.RBF_SVC_REFINEMENT_PARAM_GRID.items():
    print(f"   {param:28s} {values}")
assert tuning.grid_size(tuning.RBF_SVC_REFINEMENT_PARAM_GRID) == 24
print(f"[OK] Refinement grid size = {tuning.grid_size(tuning.RBF_SVC_REFINEMENT_PARAM_GRID)} combinations "
      f"(4 C x 3 gamma x 2 class_weight).")

t0 = time.time()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    svc_refined_search = tuning.build_rbf_svc_refinement_search()
    svc_refined_search.fit(X_train, y_train)
svc_refine_elapsed = time.time() - t0
searches["RBF SVC (refined)"] = svc_refined_search
svc_refined = tuning.summarize_best(svc_refined_search)

print(f"\nFitted 24 candidates x {cv.get_n_splits()} folds in {svc_refine_elapsed:.1f}s.")
print(f"Refined best params: {svc_refined_search.best_params_}")
print(f"   roc_auc            mean={svc_refined['roc_auc_mean']:.4f}  std={svc_refined['roc_auc_std']:.4f}")
print(f"   average_precision  mean={svc_refined['average_precision_mean']:.4f}  "
      f"std={svc_refined['average_precision_std']:.4f}")
print(f"   balanced_accuracy  mean={svc_refined['balanced_accuracy_mean']:.4f}  "
      f"std={svc_refined['balanced_accuracy_std']:.4f}")
if caught:
    for w in caught:
        print(f"   [WARNING] {w.category.__name__}: {w.message}")
else:
    print("   [OK] No warnings raised during the refinement search.")

# --- Boundary flags for the SECOND stage -------------------------------------
refined_flags = tuning.flag_boundary_hits(
    svc_refined_search.best_params_,
    tuning.RBF_SVC_REFINEMENT_PARAM_GRID,
    ("classifier__C", "classifier__gamma"),
)
if refined_flags:
    for flag in refined_flags:
        print(f"   [BOUNDARY FLAG] {flag}")
    print("   [STOP] Per the pre-declared rule, the search STOPS here regardless -- "
          "no third-stage SVC search is run. Any flag above is recorded as a residual, "
          "accepted limitation of the reported SVC configuration.")
else:
    print("   [OK] Neither refined C nor refined gamma is at an edge of the refinement grid.")

# --- Grid-local flags vs. the UNION of both searched grids --------------------
# `flag_boundary_hits` only ever sees ONE grid, so it cannot know that the
# refinement grid's lower C edge (100.0) is the first-stage grid's UPPER
# edge -- i.e. that the region below 100 was already searched in Section 6
# and found worse. Whether a flag is a genuine open question is therefore
# judged against the UNION of everything searched, which is done here.
selected_C = svc_refined_search.best_params_["classifier__C"]
union_C = sorted(
    set(tuning.RBF_SVC_PARAM_GRID["classifier__C"])
    | set(tuning.RBF_SVC_REFINEMENT_PARAM_GRID["classifier__C"])
)
print(f"\nCombined C range searched across BOTH stages: {union_C}")
if selected_C == max(union_C):
    print(f"   [OPEN] Selected C={selected_C} is still the largest value ever searched; "
          f"larger C values remain untested. Accepted as a residual limitation.")
elif selected_C == min(union_C):
    print(f"   [OPEN] Selected C={selected_C} is still the smallest value ever searched; "
          f"smaller C values remain untested. Accepted as a residual limitation.")
else:
    print(f"   [RESOLVED] Selected C={selected_C} is INTERIOR to the combined searched "
          f"range ({min(union_C)} ... {max(union_C)}): both smaller and larger C values "
          f"were evaluated and scored no better. The first-stage upper-boundary flag is "
          f"resolved -- the grid, not the data, was the earlier constraint, and extending "
          f"it did not change the selection.")

print(f"\nFirst-stage  -> refined  (same selection rule, refit='{tuning.REFIT_METRIC}'):")
for metric in tuning.CV_SCORING:
    before = first_stage_svc[f"{metric}_mean"]
    after = svc_refined[f"{metric}_mean"]
    print(f"   {metric:19s} {before:.4f} -> {after:.4f}   (delta {after - before:+.4f})")

# --- The refined configuration REPLACES the first-stage SVC -------------------
# Phase B: the refined configuration is the one that is frozen and carried
# forward. The first-stage row remains printed above and in Section 7 so the
# tuning history stays visible rather than being quietly overwritten.
results["RBF SVC"] = svc_refined
assert not hasattr(svc_refined_search.best_estimator_.named_steps["classifier"], "predict_proba")
print("\n[OK] Refined SVC has no predict_proba (probability was never enabled at either stage).")
print("[FROZEN] The refined RBF SVC configuration above is now frozen; no further SVC tuning.")

# Decision:
#   Proceed to the default-vs-tuned comparison (Section 7), where the SVC
#   row is the REFINED configuration.
# Next step:
#   Build the default-vs-tuned comparison table.


# %%
# =============================================================================
# SECTION 7 — Default-vs-tuned comparison
# =============================================================================
# Objective:
#   Compare each tuned model against its OWN frozen Milestone-2 default,
#   for all three metrics, as an observed difference -- not a formal
#   significance test.

section("SECTION 7 — Default vs. tuned comparison")

print("Note: the 'RBF SVC' tuned row below is the REFINED (second-stage)")
print("configuration from Section 6b, not the first-stage C=100 result.\n")

comparison_rows = []
for name in ["Logistic Regression", "Random Forest", "RBF SVC"]:
    default = DEFAULT_RESULTS[name]
    tuned = results[name]
    for metric in ["roc_auc", "average_precision", "balanced_accuracy"]:
        comparison_rows.append(
            {
                "model": name,
                "metric": metric,
                "default_mean": default[f"{metric}_mean"],
                "default_std": default[f"{metric}_std"],
                "tuned_mean": tuned[f"{metric}_mean"],
                "tuned_std": tuned[f"{metric}_std"],
                "delta": tuned[f"{metric}_mean"] - default[f"{metric}_mean"],
            }
        )
default_vs_tuned = pd.DataFrame(comparison_rows).set_index(["model", "metric"]).round(4)
print(default_vs_tuned.to_string())
print(
    "\nDeltas above are observed mean differences only. No formal significance "
    "test is performed; they should be read alongside each model's fold-to-fold "
    "standard deviation (both default and tuned), not treated as proof that a "
    "change is real."
)

# Next step:
#   Compare the three tuned models against each other (Section 8).


# %%
# =============================================================================
# SECTION 8 — Tuned model comparison
# =============================================================================
# Objective:
#   Rank the three tuned models against each other under the primary
#   metric, check whether Average Precision agrees, and discuss fold
#   variability descriptively -- exactly as Milestone 2 did for the
#   defaults, so the two milestones read consistently.

section("SECTION 8 — Tuned model comparison")

metric_columns = [f"{m}_{stat}" for m in tuning.CV_SCORING for stat in ("mean", "std")]
tuned_table = pd.DataFrame(
    {name: {col: results[name][col] for col in metric_columns} for name in results}
).T.round(4)
print(tuned_table.to_string())

ranking_auc = tuned_table["roc_auc_mean"].sort_values(ascending=False).index.tolist()
ranking_ap = tuned_table["average_precision_mean"].sort_values(ascending=False).index.tolist()
print(f"\nRanked by ROC-AUC (PRIMARY):      {ranking_auc}")
print(f"Ranked by Average Precision:      {ranking_ap}")
print(f"Same ranking? {'YES' if ranking_auc == ranking_ap else 'NO -- see interpretation below'}")

# Result / Interpretation (read against the actual numbers printed above
# and in Section 7):
#
#   No formal significance test is performed anywhere in this section.
#   Differences are reported as OBSERVED mean differences. The fold-to-fold
#   standard deviations are descriptive spread across 5 overlapping training
#   folds -- they are NOT standard errors and NOT confidence intervals, and
#   they are not used here as if they were.
#
#   Random Forest: the search produced an OBSERVED CV improvement in
#   ROC-AUC and Average Precision over its own default. The selected
#   configuration (bounded `max_depth`, `min_samples_leaf` above 1,
#   `max_features="sqrt"`) is more structurally regularized than the
#   unconstrained default AND achieved better CV metrics. Reduced
#   overfitting was NOT directly demonstrated -- that would require
#   comparing train-vs-validation gaps, which this script does not compute
#   (`return_train_score=False`). The two statements are kept separate.
#
#   Logistic Regression vs. first-stage SVC: these were essentially TIED on
#   ROC-AUC and on Average Precision. Differences in the third/fourth
#   decimal place are not treated as an advantage for either model.
#
#   SVC Balanced Accuracy: any large change in this metric is strongly
#   influenced by the selected `class_weight` and by the native hard
#   decision rule (`decision_function >= 0`), which moves the operating
#   point. It is not, on its own, evidence of better ranking quality --
#   ROC-AUC and Average Precision are the metrics that speak to ranking.
#
#   Ranking agreement between ROC-AUC and Average Precision is printed
#   explicitly above rather than smoothed over (the Milestone-2 defaults
#   disagreed; this comparison may not).
# Decision:
#   The provisional recommendation (Section 9) is based only on the tuned
#   training-CV numbers above; no test-set information is used.
# Next step:
#   State the model-selection caveat (Section 9b) and stop.


# %%
# =============================================================================
# SECTION 9b — What these numbers are, and what they are NOT
# =============================================================================
section("SECTION 9b — Status of these estimates")

print("""
GridSearchCV best scores, and every training-only model comparison in this
script, are part of MODEL SELECTION. They are NOT an unbiased final
generalization estimate.

Reasons, stated plainly:
  - Each reported score is the best of many candidates evaluated on the
    same 5 training folds. Selecting the maximum over 10, 96, 60 (and, for
    the SVC, a further 24) candidates biases that maximum optimistically.
  - The RBF SVC was searched TWICE (Sections 6 and 6b) while Logistic
    Regression and Random Forest were searched once. The SVC therefore
    received more post-hoc adaptation opportunity than the other two,
    which calls for especially cautious reading of any small numerical
    lead it shows over them.
  - Fold-to-fold standard deviations describe spread across overlapping
    training folds. They are not standard errors, not confidence
    intervals, and no significance claim is derived from them.

Full nested hyperparameter-tuning cross-validation, which would give an
approximately unbiased estimate of the whole tuning procedure, is
deliberately NOT introduced at this stage.

The untouched held-out test set remains reserved for the single final
locked model, and has not been used anywhere in this script.
""")


# %%
# =============================================================================
# SECTION 9 — Final model-family recommendation / STOP
# =============================================================================
section("SECTION 9 — Provisional recommendation and STOP before further work")

print("\nFull default-vs-tuned table:")
print(default_vs_tuned.to_string())
print("\nTuned models, ranked by ROC-AUC (PRIMARY metric):")
print(tuned_table.sort_values("roc_auc_mean", ascending=False).to_string())

print("""
Explicitly NOT done in this script:
  [ ] No decision threshold has been tuned or selected.
  [ ] No probability calibration has been performed.
  [ ] No stacking has been implemented (Section 13 of the Milestone-3
      design defers this to a later, separately-approved step, contingent
      on inspecting whether the three models' training-only out-of-fold
      errors are sufficiently complementary to justify it).
  [ ] Any search-boundary hit reported above (Sections 4/6) has been noted
      but NOT automatically followed by an expanded search -- that is a
      manual decision for review, not made here.
  [ ] The held-out test set was reconstructed ONLY to verify its size and
      class counts (Section 2), then deleted from the namespace and never
      referenced again -- no test-set metric of any kind was computed.

STOP. Awaiting review before any threshold optimisation, calibration,
stacking, or held-out test evaluation.
""")
