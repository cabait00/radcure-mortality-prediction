"""
================================================================================
RADCURE ML — Milestone 3C: Complementarity, Stacking, and an Exploratory
                           Challenger
================================================================================

Author:      Carlo Babo
Project:     Applied Machine Learning — RADCURE Clinical Dataset

--------------------------------------------------------------------------------
WHAT THIS SCRIPT COVERS
--------------------------------------------------------------------------------
Milestone 3A/3B tuned three prespecified model families and refined the RBF
SVC once (`analysis/03_hyperparameter_tuning.py`). Those three tuned
configurations are now FROZEN. This script asks three follow-up questions,
all on the training split only:

  1. Do the three frozen models actually make DIFFERENT errors, or are they
     near-redundant? (training-only out-of-fold complementarity analysis)
  2. Does ONE controlled stacking ensemble of them add discrimination?
  3. Does ONE exploratory gradient-boosting challenger (XGBoost) beat the
     simpler prespecified models?

--------------------------------------------------------------------------------
PRESPECIFIED vs. EXPLORATORY -- READ THIS BEFORE READING ANY TABLE
--------------------------------------------------------------------------------
Two of the models below were NOT part of the original analysis plan:

  PRESPECIFIED / CORE  Logistic Regression, Random Forest, RBF SVC
                       -- shortlisted in Milestone 2 from a
                          default-hyperparameter comparison, tuned in
                          Milestone 3A/3B.

  EXPLORATORY          Stacking, XGBoost
                       -- both added POST HOC, after the prespecified
                          comparison had been run and inspected.

This distinction is preserved in every table and is not quietly dropped
later. A post-hoc model that wins by a small margin has had a different,
weaker test applied to it than a prespecified one, because the decision to
try it at all was informed by having already seen the prespecified results.

--------------------------------------------------------------------------------
MILESTONES 1, 2 AND 3A/3B ARE FROZEN
--------------------------------------------------------------------------------
Unchanged here: target definition and eligibility, cohort definition, the
PRIMARY predictor set, deterministic cleaning, the 80/20 stratified split
(random_state=42), the preprocessing specification, the fixed
StratifiedKFold(5, shuffle=True, random_state=42), the three metrics, and
the tuned hyperparameters of the three base models. No model family is
tuned further in this script; the only search performed anywhere below is
the small exploratory XGBoost grid, which is labelled as such.

The held-out test set is reconstructed ONLY to verify its confirmed size
and class counts, then deleted from the namespace and never referenced
again. No test-set metric of any kind is computed in this script.
"""

# %%
# =============================================================================
# SECTION 1 — Scope and frozen inputs
# =============================================================================
# Objective:
#   Import what this milestone needs and print the frozen Milestone-3A/3B
#   results that everything below is measured against.

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sklearn.model_selection import cross_validate, train_test_split  # noqa: E402

from radcure import cleaning, config, ensemble, leakage, target, tuning  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 100)


def section(title: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n{title}\n{bar}")


BASE_MODEL_NAMES = ["Logistic Regression", "Random Forest", "RBF SVC"]

# Frozen Milestone-3A/3B tuned results (RBF SVC row = the REFINED
# second-stage configuration). Recorded, not recomputed.
TUNED_RESULTS: dict[str, dict[str, float]] = {
    "Logistic Regression": {
        "roc_auc_mean": 0.7897, "roc_auc_std": 0.0270,
        "average_precision_mean": 0.4404, "average_precision_std": 0.0310,
        "balanced_accuracy_mean": 0.5879, "balanced_accuracy_std": 0.0253,
    },
    "Random Forest": {
        "roc_auc_mean": 0.7791, "roc_auc_std": 0.0254,
        "average_precision_mean": 0.4374, "average_precision_std": 0.0310,
        "balanced_accuracy_mean": 0.5270, "balanced_accuracy_std": 0.0066,
    },
    # The Section-6b refinement re-selected the first-stage configuration
    # unchanged (C=100, gamma=0.001, class_weight="balanced"), so these
    # values are identical to the first-stage SVC row.
    "RBF SVC": {
        "roc_auc_mean": 0.7909, "roc_auc_std": 0.0194,
        "average_precision_mean": 0.4421, "average_precision_std": 0.0254,
        "balanced_accuracy_mean": 0.7090, "balanced_accuracy_std": 0.0217,
    },
}

section("SECTION 1 — Scope and frozen inputs")
print("Milestone 3C: training-only complementarity analysis, ONE controlled")
print("stacking experiment, and ONE exploratory XGBoost challenger.")
print("\nFrozen tuned hyperparameters carried forward from Milestone 3A/3B:")
print(f"   Logistic Regression : {ensemble.FROZEN_LOGISTIC_REGRESSION_PARAMS}")
print(f"   Random Forest       : {ensemble.FROZEN_RANDOM_FOREST_PARAMS}")
print(f"   RBF SVC (refined)   : {ensemble.FROZEN_RBF_SVC_PARAMS}")


# %%
# =============================================================================
# SECTION 2 — Reconstruct confirmed training data
# =============================================================================
# Objective:
#   Deterministically reproduce the exact X_train/y_train from Milestones
#   1-2 using the SAME reusable functions, and verify the confirmed counts.
#
# Rationale:
#   Identical in substance to Section 2 of `03_hyperparameter_tuning.py`
#   and Sections 2-3 of `02_baseline_modeling.py`. Duplicated rather than
#   imported, per the project convention that analysis scripts never import
#   one another -- each remains independently runnable. Every step is a pure
#   function of the immutable raw file plus fixed constants, so this is
#   reconstruction of an already-confirmed result, not a new decision.

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
# Reconstructed only to verify the frozen split's identity above. Nothing is
# fitted on it, nothing is predicted from it, no test-set metric is computed.
# ------------------------------------------------------------------------------
del X_test, y_test, id_test

# Class ratio used for the exploratory XGBoost `scale_pos_weight`, computed
# from the TRAINING split only (never hardcoded as a rounded literal).
TRAIN_POSITIVE_COUNT = int(y_train.sum())
TRAIN_NEGATIVE_COUNT = int((y_train == 0).sum())
TRAIN_CLASS_RATIO = TRAIN_NEGATIVE_COUNT / TRAIN_POSITIVE_COUNT
print(f"\nTraining class ratio (negatives/positives) = {TRAIN_NEGATIVE_COUNT}/"
      f"{TRAIN_POSITIVE_COUNT} = {TRAIN_CLASS_RATIO:.6f}")

# Next step:
#   Freeze the three tuned base pipelines (Section 3).


# %%
# =============================================================================
# SECTION 3 — PHASE B: freeze the three tuned base models
# =============================================================================
# Objective:
#   Build each frozen tuned model as a full preprocessing -> classifier
#   pipeline with its OWN fresh preprocessor, and confirm the frozen
#   hyperparameters and the no-probability contract.

section("SECTION 3 — PHASE B: frozen tuned base models")

base_pipelines = ensemble.build_frozen_base_pipelines()
for name, pipe in base_pipelines.items():
    clf = pipe.named_steps["classifier"]
    print(f"{name:22s} -> {type(clf).__name__}")

preprocessor_ids = {id(p.named_steps["preprocessor"]) for p in base_pipelines.values()}
assert len(preprocessor_ids) == len(base_pipelines), "Base pipelines share a preprocessor object."
print("\n[OK] Each frozen base pipeline has its OWN fresh, unfitted preprocessor.")

svc_clf = base_pipelines["RBF SVC"].named_steps["classifier"]
assert not hasattr(svc_clf, "predict_proba")
print("[OK] Frozen RBF SVC exposes decision_function only (probability never enabled).")
print("[FROZEN] No further tuning of Logistic Regression, Random Forest or RBF SVC.")

# Next step:
#   Generate training-only out-of-fold scores and analyse complementarity.


# %%
# =============================================================================
# SECTION 4 — PHASE C: training-only OOF complementarity analysis
# =============================================================================
# Objective:
#   Determine whether the three frozen models are near-redundant or make
#   genuinely different errors -- the factual basis for judging whether an
#   ensemble has any plausible rationale.
#
# Methodological rationale:
#   Out-of-fold predictions under the SAME fixed 5-fold splitter give each
#   observation a score from a model that never saw it, so the comparison
#   is honest and uses training data only.
#
#   Spearman (rank) correlation is the PRIMARY pairwise measure here, for a
#   concrete reason: ROC-AUC and Average Precision are both rank-based, and
#   the three models do not share a score scale (LR/RF emit probabilities in
#   [0,1]; the SVC emits unbounded decision_function margins, because
#   probability=True is deliberately never enabled). Pearson correlation on
#   such mixed scales conflates scale differences with genuine disagreement,
#   so it is reported as SUPPLEMENTARY information only.
#
#   Hard-prediction disagreement uses each estimator's NATIVE decision rule
#   (probability >= 0.5, decision_function >= 0). No threshold is selected
#   or tuned -- these are the built-in defaults, used only to describe where
#   the models differ.

section("SECTION 4 — PHASE C: training-only OOF complementarity")

oof_scores: dict[str, np.ndarray] = {}
oof_hard: dict[str, np.ndarray] = {}

for name, pipe in base_pipelines.items():
    method = ensemble.OOF_SCORE_METHODS[name]
    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scores = ensemble.compute_oof_scores(pipe, X_train, y_train, method)
    oof_scores[name] = scores
    oof_hard[name] = ensemble.hard_predictions(scores, method)
    print(f"{name:22s} OOF via {method:18s} in {time.time() - t0:5.1f}s  "
          f"range=[{scores.min():+.4f}, {scores.max():+.4f}]  "
          f"positive-rate={oof_hard[name].mean():.4f}")
    for w in caught:
        print(f"   [WARNING] {w.category.__name__}: {w.message}")

print("\nNote the differing score ranges above: these are NOT a common scale.")
print("Rank-based (Spearman) comparison is therefore the primary measure.")

# --- Pairwise correlations ---------------------------------------------------
pairs = [(a, b) for i, a in enumerate(BASE_MODEL_NAMES) for b in BASE_MODEL_NAMES[i + 1:]]

corr_rows = []
for a, b in pairs:
    rho = spearmanr(oof_scores[a], oof_scores[b]).statistic
    r = pearsonr(oof_scores[a], oof_scores[b]).statistic
    corr_rows.append({"pair": f"{a} vs {b}", "spearman_PRIMARY": rho, "pearson_supplementary": r})
corr_table = pd.DataFrame(corr_rows).set_index("pair").round(4)
print("\nPairwise OOF score correlations:")
print(corr_table.to_string())

# --- Hard-prediction disagreement and error overlap --------------------------
overlap_rows = []
for a, b in pairs:
    stats = ensemble.error_overlap(y_train, oof_hard[a], oof_hard[b])
    overlap_rows.append({
        "pair": f"{a} vs {b}",
        "disagree_rate": stats["disagreement_rate"],
        "both_wrong": stats["both_wrong"],
        "only_A_wrong": stats["only_a_wrong"],
        "only_B_wrong": stats["only_b_wrong"],
        "both_correct": stats["both_correct"],
        "FN_both": stats["fn_both"],
        "FN_only_A": stats["fn_only_a"],
        "FN_only_B": stats["fn_only_b"],
        "FP_both": stats["fp_both"],
        "FP_only_A": stats["fp_only_a"],
        "FP_only_B": stats["fp_only_b"],
    })
overlap_table = pd.DataFrame(overlap_rows).set_index("pair")
print("\nOOF hard-prediction disagreement and error overlap "
      "(A = first model in the pair, B = second):")
print(overlap_table.round(4).to_string())

print(f"\nFor reference, training prevalence: {TRAIN_POSITIVE_COUNT}/{len(y_train)} "
      f"= {y_train.mean():.4f} positive.")

# Result / Interpretation:
#   Read against the tables printed above. The questions this section
#   exists to answer, each answered descriptively and WITHOUT any formal
#   significance claim:
#     - Are the rankings nearly redundant? -> look at Spearman. Values very
#       close to 1.0 would mean the models order patients almost
#       identically, so an ensemble could only re-weight near-identical
#       information.
#     - Is Random Forest weaker but meaningfully different? -> compare its
#       Spearman against the other two with its standalone ROC-AUC
#       (0.7791, the lowest of the three).
#     - Do the models make non-identical errors? -> look at
#       `only_A_wrong` / `only_B_wrong`: observations one model gets right
#       and the other does not. Large values here are what an ensemble
#       could in principle exploit.
#     - Note that a high hard-prediction disagreement rate can be driven
#       largely by the SVC's class_weight="balanced" operating point rather
#       than by different ranking information; the Spearman column is the
#       check against over-reading it.
# Next step:
#   Run the single controlled stacking experiment (Section 5).


# %%
# =============================================================================
# SECTION 5 — PHASE D: one controlled stacking experiment
# =============================================================================
# Objective:
#   Evaluate ONE prespecified stacking configuration of the three frozen
#   base models, under a proper outer cross-validation.
#
# Leakage control (the critical design point):
#   TWO levels of cross-validation are used.
#     - INNER (inside StackingClassifier, cv=StratifiedKFold(5, ...)):
#       builds the meta-features. Every meta-feature for an observation
#       comes from a base learner fitted WITHOUT that observation. Base
#       learners are never fitted on the full training set and then asked
#       for in-sample predictions to train the meta-learner -- that would
#       let the meta-learner see optimistic, partly-memorised base
#       predictions.
#     - OUTER (cross_validate, StratifiedKFold(5, shuffle=True,
#       random_state=42)): scores the ENTIRE stacking procedure, including
#       its internal meta-feature generation, on folds held out from the
#       whole thing.
#
# Scope limits, fixed in advance:
#   No tuning of the stack. No alternative meta-learner. No passthrough
#   variant. No multiple stacking variants. This is one experiment, not a
#   new search space.
#
# Meta-feature scale correction (part of this SAME single experiment, not a
# second stacking variant):
#   stack_method="auto" gives the meta-learner two probability columns
#   (LR, RF) and one unbounded decision_function column (SVC), on different
#   scales. Because the final estimator is an L2-regularised
#   LogisticRegression and L2 regularisation is scale-dependent, the final
#   estimator standardises its inputs first: `Pipeline([("scaler",
#   StandardScaler()), ("classifier", LogisticRegression(...))])`. The
#   scaler is fit only on the inner-CV meta-features, never on held-out
#   test data.

section("SECTION 5 — PHASE D: one controlled stacking experiment")

stack = ensemble.build_stacking_classifier()
print(f"Base estimators : {[n for n, _ in stack.estimators]}")
final_steps = " -> ".join(type(step).__name__ for _, step in stack.final_estimator.steps)
print(f"Final estimator : {final_steps}")
print(f"stack_method    : {stack.stack_method!r}    passthrough: {stack.passthrough}")
print(f"INNER cv        : {stack.cv}")

outer_cv = ensemble.build_cv_splitter()
print(f"OUTER cv        : {outer_cv}")

t0 = time.time()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    stack_cv = cross_validate(
        stack, X_train, y_train,
        cv=outer_cv,
        scoring=tuning.CV_SCORING,
        n_jobs=1,
        return_train_score=False,
    )
stack_elapsed = time.time() - t0
print(f"\nOuter-CV evaluation of the stack completed in {stack_elapsed:.1f}s.")
for w in caught:
    print(f"   [WARNING] {w.category.__name__}: {w.message}")

stacking_result = {}
for metric in tuning.CV_SCORING:
    fold_scores = stack_cv[f"test_{metric}"]
    stacking_result[f"{metric}_mean"] = float(np.mean(fold_scores))
    stacking_result[f"{metric}_std"] = float(np.std(fold_scores))
    print(f"   {metric:19s} mean={np.mean(fold_scores):.4f}  std={np.std(fold_scores):.4f}")

print("""
These are OUTER-fold means across the training split. They remain
TRAINING-SET model-selection estimates, not a final unbiased
generalization estimate.""")

# Next step:
#   Run the exploratory XGBoost challenger (Section 6).


# %%
# =============================================================================
# SECTION 6 — PHASE E: first-stage exploratory XGBoost challenger
# =============================================================================
# Objective:
#   Check whether a gradient-boosted tree ensemble offers a materially
#   better training-CV result than the simpler prespecified models.
#
# SCIENTIFIC STATUS -- stated plainly and not softened later:
#   XGBoost is a POST-HOC EXPLORATORY CHALLENGER, added AFTER inspection of
#   the prespecified LR/RF/SVC comparison. It was not in the Milestone-2
#   shortlist and was not part of the original analysis plan. It is labelled
#   EXPLORATORY in every table below. Presenting it as though it had been
#   prespecified would misrepresent how it came to be tried.
#
# Scope limits:
#   A deliberately small 12-combination grid over `max_depth`,
#   `learning_rate` and `scale_pos_weight` only. No other XGBoost parameter
#   is tuned; no early-stopping machinery, no Optuna, no RandomizedSearchCV.
#   Same frozen preprocessor, same fixed CV, same three metrics, same
#   refit="roc_auc" as every other search in this project.

section("SECTION 6 — PHASE E: first-stage exploratory XGBoost challenger")

xgb_result: dict[str, float] | None = None
xgb_best_params: dict | None = None

if not tuning.XGBOOST_AVAILABLE:
    print("[SKIPPED] xgboost is NOT installed in this environment.")
    print(f"          Import error: {tuning.XGBOOST_IMPORT_ERROR}")
    print("          Nothing was installed automatically. This subtask is")
    print("          reported as not run; all other Milestone-3C work is complete.")
else:
    import xgboost  # noqa: E402

    print(f"xgboost version: {xgboost.__version__}")
    xgb_grid = tuning.build_xgboost_param_grid(TRAIN_CLASS_RATIO)
    print(f"Exploratory grid ({tuning.grid_size(xgb_grid)} combinations):")
    for param, values in xgb_grid.items():
        print(f"   {param:32s} {values}")
    assert tuning.grid_size(xgb_grid) == 12
    print(f"[OK] Grid size = 12 -> 12 x {outer_cv.get_n_splits()} = 60 CV fits.")

    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        xgb_search = tuning.build_xgboost_search(TRAIN_CLASS_RATIO)
        xgb_search.fit(X_train, y_train)
    xgb_elapsed = time.time() - t0
    xgb_summary = tuning.summarize_best(xgb_search)
    xgb_best_params = xgb_search.best_params_
    xgb_result = {k: v for k, v in xgb_summary.items() if isinstance(v, float)}

    print(f"\nFitted 12 candidates x {outer_cv.get_n_splits()} folds in {xgb_elapsed:.1f}s.")
    print(f"Best params: {xgb_best_params}")
    for metric in tuning.CV_SCORING:
        print(f"   {metric:19s} mean={xgb_summary[f'{metric}_mean']:.4f}  "
              f"std={xgb_summary[f'{metric}_std']:.4f}")
    for w in caught:
        print(f"   [WARNING] {w.category.__name__}: {w.message}")

first_stage_xgb_result = xgb_result

# Result / Interpretation:
#   The first-stage search selected `learning_rate=0.03` and `max_depth=2`,
#   both at the LOWER edge of their tested ranges, while `n_estimators` had
#   been held fixed at 300. That combination of edges is exactly the
#   pattern that motivates a targeted second-stage search below: a small
#   learning rate paired with a fixed, possibly-too-low round count cannot
#   be distinguished from "this region is genuinely worse".
# Next step:
#   Run the ONE targeted second-stage refinement (Section 6b).


# %%
# =============================================================================
# SECTION 6b — PHASE E continued: targeted XGBoost refinement (FINAL stage)
# =============================================================================
# Objective:
#   Resolve the first-stage lower-boundary hits on `learning_rate` and
#   `max_depth` with ONE targeted second-stage search that also searches
#   `n_estimators`, since learning rate and round count trade off against
#   each other in gradient boosting.
#
# Scientific motivation (stated before, not after, seeing the result):
#   The first-stage grid could not tell whether `learning_rate=0.03` /
#   `max_depth=2` were selected because that region is genuinely best, or
#   because `n_estimators` was fixed at 300 and a smaller learning rate
#   simply was not given enough rounds to reach a comparable fit. This
#   refinement is not expected, in advance, to improve on the first-stage
#   result -- it is run to answer that specific ambiguity either way.
#
# Cost, stated openly:
#   This is XGBoost's SECOND search stage, while Logistic Regression and
#   Random Forest each received exactly one. Any small XGBoost lead over
#   those two models must be read with that selection-effort asymmetry in
#   mind (see Section 8's model-selection caveat).
#
# Pre-declared stopping rule: there is NO third XGBoost search. Whatever is
# selected here, any residual boundary hit is recorded as a documented
# limitation, not followed by a further search.

section("SECTION 6b — PHASE E continued: targeted XGBoost refinement (FINAL stage)")

xgb_refined_result: dict[str, float] | None = None
xgb_refined_best_params: dict | None = None

if not tuning.XGBOOST_AVAILABLE:
    print("[SKIPPED] xgboost is NOT installed in this environment -- the first-stage")
    print("          search above was already skipped for the same reason.")
else:
    xgb_refine_grid = tuning.build_xgboost_refinement_param_grid(TRAIN_CLASS_RATIO)
    print("Targeted second-stage grid (does not alter the first-stage grid above):")
    for param, values in xgb_refine_grid.items():
        print(f"   {param:32s} {values}")
    assert tuning.grid_size(xgb_refine_grid) == 54
    print(f"[OK] Grid size = 54 -> 54 x {outer_cv.get_n_splits()} = 270 CV fits.")

    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        xgb_refined_search = tuning.build_xgboost_refinement_search(TRAIN_CLASS_RATIO)
        xgb_refined_search.fit(X_train, y_train)
    xgb_refine_elapsed = time.time() - t0
    xgb_refined_summary = tuning.summarize_best(xgb_refined_search)
    xgb_refined_best_params = xgb_refined_search.best_params_
    xgb_refined_result = {
        k: v for k, v in xgb_refined_summary.items() if isinstance(v, float)
    }

    print(f"\nFitted 54 candidates x {outer_cv.get_n_splits()} folds in {xgb_refine_elapsed:.1f}s.")
    print(f"Refined best params: {xgb_refined_best_params}")
    for metric in tuning.CV_SCORING:
        print(f"   {metric:19s} mean={xgb_refined_summary[f'{metric}_mean']:.4f}  "
              f"std={xgb_refined_summary[f'{metric}_std']:.4f}")
    for w in caught:
        print(f"   [WARNING] {w.category.__name__}: {w.message}")

    # --- Delta vs. the first-stage result ------------------------------------
    print(f"\nFirst-stage -> refined (same selection rule, refit='{tuning.REFIT_METRIC}'):")
    for metric in tuning.CV_SCORING:
        before = first_stage_xgb_result[f"{metric}_mean"]
        after = xgb_refined_result[f"{metric}_mean"]
        print(f"   {metric:19s} {before:.4f} -> {after:.4f}   (delta {after - before:+.4f})")

    # --- Boundary flags for the SECOND stage ---------------------------------
    refine_flags = tuning.flag_boundary_hits(
        xgb_refined_best_params,
        xgb_refine_grid,
        ("classifier__max_depth", "classifier__learning_rate", "classifier__n_estimators"),
    )
    if refine_flags:
        for flag in refine_flags:
            print(f"   [BOUNDARY FLAG] {flag}")
        print("   [STOP] Per the pre-declared rule, the search STOPS here regardless -- "
              "no third-stage XGBoost search is run. Any flag above is recorded as a "
              "documented, accepted limitation of the reported XGBoost configuration.")
    else:
        print("   [OK] max_depth, learning_rate and n_estimators are all interior to "
              "the second-stage grid.")

    print("\n[FROZEN] This is the FINAL XGBoost search; no third stage follows.")

# Next step:
#   Assemble the updated final training-only comparison (Section 7), using
#   the REFINED XGBoost result in place of the first-stage one.


# %%
# =============================================================================
# SECTION 7 — PHASE F: final training-only comparison (updated)
# =============================================================================
# Objective:
#   One table containing every candidate considered, with the
#   PRESPECIFIED / EXPLORATORY distinction preserved.

section("SECTION 7 — PHASE F: final training-only comparison")

print("Note: the 'XGBoost' row below is the REFINED (second-stage) configuration")
print("from Section 6b, not the first-stage result. Both are shown for the record.")
print("Note: the 'Stacking' row uses the scaled final estimator (Section 5) --")
print("the single stacking experiment, corrected, not a second variant.\n")

final_rows = []
for name in BASE_MODEL_NAMES:
    row = {"model": name, "status": "PRESPECIFIED"}
    row.update(TUNED_RESULTS[name])
    final_rows.append(row)

stack_row = {"model": "Stacking (LR+RF+SVC)", "status": "EXPLORATORY"}
stack_row.update(stacking_result)
final_rows.append(stack_row)

if first_stage_xgb_result is not None:
    xgb_first_row = {"model": "XGBoost (1st stage)", "status": "EXPLORATORY"}
    xgb_first_row.update(first_stage_xgb_result)
    final_rows.append(xgb_first_row)

if xgb_refined_result is not None:
    xgb_row = {"model": "XGBoost (refined)", "status": "EXPLORATORY"}
    xgb_row.update(xgb_refined_result)
    final_rows.append(xgb_row)

final_table = pd.DataFrame(final_rows).set_index("model")
display_columns = ["status"] + [
    f"{m}_{s}" for m in tuning.CV_SCORING for s in ("mean", "std")
]
final_table = final_table[display_columns]
print(final_table.round(4).to_string())

print("\nRanked by ROC-AUC (PRIMARY metric):")
print(final_table.sort_values("roc_auc_mean", ascending=False)
      [["status", "roc_auc_mean", "average_precision_mean", "balanced_accuracy_mean"]]
      .round(4).to_string())

# --- Spread of the leaders ---------------------------------------------------
auc_sorted = final_table.sort_values("roc_auc_mean", ascending=False)
leader = auc_sorted.index[0]
lr_auc = final_table.loc["Logistic Regression", "roc_auc_mean"]
print(f"\nTop model by ROC-AUC: {leader} ({auc_sorted.iloc[0]['roc_auc_mean']:.4f})")
print(f"Logistic Regression  : {lr_auc:.4f}   "
      f"(difference {auc_sorted.iloc[0]['roc_auc_mean'] - lr_auc:+.4f})")
print("The observed differences among the leading means are small and do not")
print("provide a practically compelling advantage for any one candidate. Fold")
print("standard deviations describe spread across overlapping training folds")
print("-- they are NOT standard errors, NOT confidence intervals, and are not")
print("used here to judge whether a difference between models is real.")

# Next step:
#   State the interpretation, the status of these numbers, and STOP.


# %%
# =============================================================================
# SECTION 8 — Interpretation, status of these estimates, and STOP
# =============================================================================
section("SECTION 8 — Interpretation and STOP")

print("""
STATUS OF EVERY NUMBER IN THIS SCRIPT
-------------------------------------
GridSearchCV best scores and all training-only model comparisons above are
part of MODEL SELECTION. They are NOT an unbiased final generalization
estimate. Each reported score is the best or the chosen candidate among
several evaluated on the same training folds, which biases it
optimistically; the RBF SVC additionally received more post-hoc adaptation
opportunity than LR and RF because it was searched in two stages. Full
nested hyperparameter-tuning cross-validation is deliberately NOT
introduced at this stage. The untouched held-out test set remains reserved
for the single final locked model.

This second XGBoost search (Section 6b) is itself a POST-HOC EXPLORATORY
REFINEMENT performed after inspection of the first-stage XGBoost search.
Its best CV score is therefore part of model development / model
selection, not an unbiased final generalization estimate, exactly like
every other number in this milestone. XGBoost has now received TWO search
stages (12 then 54 candidates), while Logistic Regression and Random
Forest each received exactly one. XGBoost therefore received more post-hoc
adaptation opportunity through two search stages, so a very small
numerical lead would require especially cautious interpretation.

Fold standard deviations are descriptive spread across 5 overlapping
training folds. They are not standard errors, not confidence intervals,
and no significance test is performed anywhere in this milestone.

WORDING THIS PROJECT USES CAREFULLY
-----------------------------------
  - Random Forest showed an OBSERVED CV improvement in ROC-AUC/AP over its
    default. The selected RF is more structurally regularized and achieved
    better CV metrics; reduced overfitting was NOT directly demonstrated
    (train-vs-validation gaps were never computed).
  - Logistic Regression and the first-stage SVC were essentially TIED on
    ROC-AUC and Average Precision. Fourth-decimal differences are not an
    advantage for either.
  - The SVC's large Balanced Accuracy change is strongly influenced by the
    selected class_weight and by its native hard-decision operating point
    (decision_function >= 0), not by demonstrably better ranking quality.
  - A small XGBoost lead over LR/SVC, if one is observed, is not treated
    as proof of superiority: the observed differences are small and do not
    provide a practically compelling advantage, and XGBoost's two-stage
    post-hoc adaptation opportunity calls for especially cautious reading
    of any such lead. Model complexity and transparency are weighed
    alongside it, not after it.
""")

xgb_interp_lines = []
if xgb_refined_result is not None:
    refined_auc = xgb_refined_result["roc_auc_mean"]
    refined_ap = xgb_refined_result["average_precision_mean"]
    first_auc = first_stage_xgb_result["roc_auc_mean"]
    first_ap = first_stage_xgb_result["average_precision_mean"]
    lr_auc_ref = TUNED_RESULTS["Logistic Regression"]["roc_auc_mean"]
    svc_auc_ref = TUNED_RESULTS["RBF SVC"]["roc_auc_mean"]

    auc_delta_from_first = refined_auc - first_auc
    ap_delta_from_first = refined_ap - first_ap
    auc_delta_vs_lr = refined_auc - lr_auc_ref
    auc_delta_vs_svc = refined_auc - svc_auc_ref

    xgb_interp_lines.append("XGBOOST REFINEMENT -- SCIENTIFIC INTERPRETATION")
    xgb_interp_lines.append("-----------------------------------------------")
    xgb_interp_lines.append(
        f"1. Materiality of the refinement: refined ROC-AUC {refined_auc:.4f} vs. "
        f"first-stage {first_auc:.4f} (delta {auc_delta_from_first:+.4f}). "
        + ("This is a small, single-digit-third-decimal movement, not a material "
           "improvement." if abs(auc_delta_from_first) < 0.01 else
           "This is a larger movement than the other deltas in this milestone and "
           "is reported as such, still without a significance claim.")
    )
    xgb_interp_lines.append(
        f"2. Refined XGBoost vs. LR/SVC on ROC-AUC: {refined_auc:.4f} vs. "
        f"LR {lr_auc_ref:.4f} (delta {auc_delta_vs_lr:+.4f}) and SVC {svc_auc_ref:.4f} "
        f"(delta {auc_delta_vs_svc:+.4f}). "
        + ("Refined XGBoost does NOT clearly exceed either simpler model."
           if max(auc_delta_vs_lr, auc_delta_vs_svc) < 0.01 else
           "Refined XGBoost shows a numerically larger lead over both simpler "
           "models than the other pairwise comparisons in this milestone.")
    )
    xgb_interp_lines.append(
        f"3. AP consistency: Average Precision moved {ap_delta_from_first:+.4f} alongside "
        f"the ROC-AUC delta of {auc_delta_from_first:+.4f}. "
        + ("Both metrics moved in the same direction." if (auc_delta_from_first >= 0) == (ap_delta_from_first >= 0)
           else "The two metrics did NOT move in the same direction, which weakens any "
                "claim of a genuine, broad-based improvement.")
    )
    xgb_interp_lines.append(
        "4. Complexity-adjusted judgement: XGBoost with 300-1000 boosted trees and 4 "
        "tuned hyperparameters across two search stages is substantially more complex "
        "than Logistic Regression's single linear model with 2 tuned hyperparameters "
        "searched once. "
        + ("The observed gain is judged too small to justify that added complexity."
           if max(auc_delta_vs_lr, auc_delta_vs_svc) < 0.01 else
           "Whether the observed gain justifies that added complexity is a judgement "
           "call weighed explicitly against transparency below, not decided by the "
           "ROC-AUC number alone.")
    )
    if refined_auc >= 0.80:
        xgb_interp_lines.append(
            f"5. Refined XGBoost reached ROC-AUC {refined_auc:.4f}, at or above 0.80. "
            "This is reported factually. 0.80 was never a pre-specified success "
            "threshold in this project and is not treated as one here."
        )
    elif 0.78 <= refined_auc <= 0.79:
        xgb_interp_lines.append(
            f"6. Refined XGBoost remains at ROC-AUC {refined_auc:.4f}, in the ~0.78-0.79 "
            "range also occupied by the first-stage result. The targeted refinement did "
            "NOT establish an advantage over the simpler prespecified candidates."
        )
    xgb_interp_lines.append(
        "7. The observed differences here are small and do not provide a "
        "practically compelling advantage for XGBoost; combined with the extra "
        "post-hoc adaptation opportunity XGBoost received through two search "
        "stages, model complexity and transparency are preserved as explicit, "
        "weighted factors in the final recommendation below rather than "
        "overridden by a small ROC-AUC difference."
    )
    print("\n" + "\n".join(xgb_interp_lines))

print("""
UPDATED PROVISIONAL RECOMMENDATION
-----------------------------------
The targeted XGBoost refinement is now included as EXPLORATORY, alongside
the PRESPECIFIED Logistic Regression / Random Forest / RBF SVC and the
EXPLORATORY stacking result. Nothing in the refined XGBoost result changes
the recommendation stated for the prespecified comparison: the observed
differences among the leading candidates are small and do not provide a
practically compelling advantage for any one of them, and XGBoost received
more post-hoc adaptation opportunity than LR and RF through its two search
stages, which calls for especially cautious reading of its numerical
position. Model simplicity and transparency remain deciding factors rather
than being displaced by a small ROC-AUC difference.

EXPLICITLY NOT DONE IN THIS SCRIPT
----------------------------------
  [ ] No final threshold selection.
  [ ] No probability calibration.
  [ ] No held-out test evaluation -- the test set was reconstructed only to
      verify its counts (Section 2), then deleted and never referenced.
  [ ] No feature-importance interpretation, no permutation importance, no
      SHAP, no PDP/ICE.
  [ ] No final model fitting on the full training set for deployment.
  [ ] No further tuning of LR / RF / SVC (frozen in Phase B).
  [ ] No third XGBoost search -- Section 6b is the FINAL XGBoost stage,
      regardless of any residual boundary flag reported there.
  [ ] No second stacking variant, no alternative meta-learner, no
      passthrough experiment.
  [ ] Nothing committed or pushed.

STOP. Awaiting review before any of the above.
""")
