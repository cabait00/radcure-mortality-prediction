# %% [markdown-style header — see module docstring below for the full narrative]
"""
================================================================================
RADCURE ML — Milestone 2: Baseline Model-Family Comparison (training-only)
================================================================================

Author:      Carlo Babo
Project:     Applied Machine Learning — RADCURE Clinical Dataset

--------------------------------------------------------------------------------
WHAT THIS SCRIPT COVERS
--------------------------------------------------------------------------------
This is the FIRST part of Milestone 2: a strictly training-only comparison
of five default-hyperparameter classical model families on the confirmed
PRIMARY predictor set, under a fixed 5-fold stratified cross-validation
design. It answers one question only -- "which default model configurations
show consistent cross-validated discrimination before tuning?" -- and stops
there. No hyperparameter is tuned, no class-imbalance handling is applied,
and the held-out test set is never inspected, preprocessed, fitted on, or
predicted anywhere below.

--------------------------------------------------------------------------------
MILESTONE 1 IS FROZEN -- NOTHING BELOW CHANGES IT
--------------------------------------------------------------------------------
This script does not redefine, and does not import as a module, Milestone 1
(`analysis/01_project_foundation.py`). It reconstructs the confirmed
deterministic modelling cohort and the confirmed 80/20 split by calling the
SAME reusable functions Milestone 1 used (`cleaning.*`, `target.*`,
`config.*`, `preprocessing.build_preprocessor`), because that reconstruction
is fully deterministic given a fixed `random_state` -- it is not a new
design decision. Frozen and unchanged here:

    - target definition, target eligibility, cohort definition
    - the PRIMARY predictor set (Age, Sex, ECOG PS, Smoking PY,
      Smoking Status, Ds Site, T, N)
    - deterministic cleaning rules
    - the primary 80/20 stratified split, random_state=42
    - the preprocessing specification (median imputation, no indicator,
      StandardScaler; OneHotEncoder(handle_unknown="ignore", drop=None))
    - held-out test-set policy: reconstructed ONLY to verify its confirmed
      size and class counts (Section 3), then never touched again.

--------------------------------------------------------------------------------
GOVERNING RULES FOR THIS MILESTONE
--------------------------------------------------------------------------------
  - Five DEFAULT-hyperparameter candidates only (Section 4): Dummy (prior),
    Logistic Regression, Decision Tree, Random Forest, RBF SVC. No
    class_weight, no resampling/SMOTE, no KNN, no boosting, no stacking,
    no feature selection, no tuned hyperparameters, no probability
    calibration, no SVC(probability=True).
  - Three PRE-SPECIFIED CV metrics, fixed before any result is seen:
    roc_auc (PRIMARY), average_precision (SECONDARY),
    balanced_accuracy (SUPPLEMENTARY). Ordinary accuracy is not used for
    model selection. No threshold tuning. No test-set Precision/Recall/F1/
    confusion matrices/ROC or PR curves.
  - Every model is evaluated with the SAME fixed
    StratifiedKFold(n_splits=5, shuffle=True, random_state=42), on the
    TRAINING split only.
  - class_weight="balanced" is deliberately NOT applied in this milestone
    (Section 8): it is a modelling choice, not a baseline-comparison
    default, and would confound a default-family comparison if silently
    mixed in. It remains an explicit, controlled candidate for Milestone-3
    tuning if the imbalance proves to matter.

Confirmed project environment: `ml` conda environment
(/home/c/miniconda3/envs/ml/bin/python) -- pandas 3.0.5, numpy 2.4.6,
scikit-learn 1.9.0, openpyxl 3.1.5, pytest 9.1.1.
"""

# %%
# =============================================================================
# SECTION 1 — Imports and Milestone-2 scope
# =============================================================================
# Objective:
#   Make the `radcure` package importable, import only what this milestone
#   needs (no tuning search, no resampling library, no calibration), and
#   confirm the scope in one place before anything is computed.

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sklearn.model_selection import train_test_split  # noqa: E402

from radcure import cleaning, config, leakage, modeling, target  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 100)


def section(title: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n{title}\n{bar}")


print("Imports OK. `radcure` package resolved from:", SRC_DIR)
print("Milestone 2, part 1: training-only baseline model-family comparison.")
print("Candidates:", list(modeling.build_model_registry().keys()))
print("CV metrics:", list(modeling.CV_SCORING.keys()), "(roc_auc = PRIMARY)")


# %%
# =============================================================================
# SECTION 2 — Reconstruct the confirmed modelling cohort and frozen split
# =============================================================================
# Objective:
#   Deterministically reproduce the exact X_train/y_train/X_test/y_test
#   from Milestone 1, using the SAME reusable functions, WITHOUT repeating
#   Milestone 1's audit narrative or diagnostic output.
#
# Rationale:
#   Every step below is a pure function of the immutable raw file plus
#   fixed constants (`config.PRIMARY_FEATURES`, `config.RANDOM_STATE`,
#   `config.TEST_SIZE`) -- given the same inputs it reproduces the confirmed
#   Milestone-1 split configuration and counts. This is reconstruction of
#   an already-confirmed result, not a new design decision, so it is
#   deliberately compact.

section("SECTION 2 — Reconstructing the confirmed cohort and split")

raw_df = cleaning.load_raw_clinical()
target_df = target.build_target(raw_df)
eligible_mask = target_df[config.ELIGIBILITY_COLUMN]

# Deterministic semantic cleaning of exactly the 8 PRIMARY predictors
# (Section 4 of Milestone 1) -- Age and Sex require no cleaning.
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
      f"(reconstructed for verification ONLY -- see Section 3)")

# Result / Interpretation:
#   The reconstruction uses exactly the deterministic functions Milestone 1
#   used, in the same order, with the same fixed constants -- it is
#   expected to reproduce the frozen split exactly, verified next.
# Decision:
#   Proceed to explicit integrity assertions before using X_train/y_train
#   for anything.
# Next step:
#   Verify the reconstructed split against the confirmed Milestone-1
#   counts, then stop touching the test set entirely (Section 3).


# %%
# =============================================================================
# SECTION 3 — Verify training-only modelling inputs
# =============================================================================
# Objective:
#   Verify, via executable assertions, that the reconstruction above
#   reproduces the confirmed Milestone-1 split configuration and counts,
#   and then draw an explicit line: nothing below this cell may reference
#   X_test, y_test or id_test again.
#
# Assumptions:
#   None -- these are hard equality checks against the values fixed in
#   `config.py` (EXPECTED_TRAIN_N, EXPECTED_TRAIN_EVENTS, ...).

section("SECTION 3 — Verifying training-only modelling inputs")

assert list(X_train.columns) == config.PRIMARY_FEATURES == config.PRIMARY_NUMERIC_FEATURES + config.PRIMARY_CATEGORICAL_FEATURES
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
#   Fix the cross-validation and metric design (Section 4) before fitting
#   anything.


# %%
# =============================================================================
# SECTION 4 — Cross-validation and metric design
# =============================================================================
# Objective:
#   Fix, in one place and before any model is touched, the exact
#   cross-validation splitter and the exact metrics every candidate will
#   be judged on.
#
# Rationale:
#   Pre-specifying metrics before results exist prevents post-hoc metric
#   shopping. ROC-AUC is PRIMARY (threshold-independent ranking quality);
#   Average Precision is SECONDARY and specifically informative at ~18.9%
#   prevalence, where a random classifier's PR curve sits near the
#   prevalence rather than at 0.5; Balanced Accuracy is SUPPLEMENTARY,
#   reported for a hard-prediction view that is not distorted by class
#   imbalance the way ordinary accuracy would be. Ordinary accuracy is
#   deliberately NOT used for model selection.

section("SECTION 4 — Cross-validation and metric design")

cv = modeling.build_cv_splitter()
cv_attrs = modeling.describe_cv_splitter(cv)
print(f"CV splitter: StratifiedKFold(n_splits={cv.get_n_splits()}, shuffle={cv_attrs.shuffle}, "
      f"random_state={cv_attrs.random_state}) -- SAME splitter for every model and every metric.")
print(f"Metrics (fixed before any result is inspected): {modeling.CV_SCORING}")
print(f"Training-set event prevalence: {y_train.mean()*100:.2f}% "
      f"({int(y_train.sum())} of {len(y_train)}).")

results: dict[str, dict[str, np.ndarray | float]] = {}


def run_and_report(name: str, pipeline) -> dict[str, np.ndarray | float]:
    """Run the pre-specified CV metrics for one candidate on TRAINING data
    only, print per-fold scores and mean +/- std, and surface any warning
    (e.g. a convergence warning) rather than silencing it.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scores = modeling.evaluate_candidate(pipeline, X_train, y_train, cv)
    print(f"\n--- {name} ---")
    for metric_name in modeling.CV_SCORING:
        fold_scores = scores[f"{metric_name}_scores"]
        assert isinstance(fold_scores, np.ndarray)
        mean_ = float(scores[f"{metric_name}_mean"])
        std_ = float(scores[f"{metric_name}_std"])
        formatted = ", ".join(f"{s:.4f}" for s in fold_scores)
        print(f"   {metric_name:18s} folds=[{formatted}]  mean={mean_:.4f}  std={std_:.4f}")
    if caught:
        for w in caught:
            print(f"   [WARNING] {w.category.__name__}: {w.message}")
    else:
        print("   [OK] No warnings raised during cross-validation.")
    return scores

# Next step:
#   Evaluate the Dummy baseline first, as the no-signal reference every
#   other candidate must be judged against (Section 5).


# %%
# =============================================================================
# SECTION 5 — Dummy baseline
# =============================================================================
# Objective:
#   Establish the no-signal reference empirically, not by assertion.
#
# Rationale:
#   `DummyClassifier(strategy="prior")` predicts, for every sample, the
#   training-FOLD class prior -- it uses no predictor information at all.
#   With a constant predicted score per fold, ROC-AUC is expected to be
#   0.5 (a constant score cannot rank positives above negatives), Average
#   Precision is expected to track the training-fold event prevalence
#   (~18.9%), and hard predictions (via `strategy="prior"`, which predicts
#   the majority class) will always favour the non-event class. These are
#   stated as EXPECTATIONS to check against the computed numbers below,
#   not hard-coded results.

section("SECTION 5 — Dummy baseline (no-signal reference)")

pipelines = modeling.build_candidate_pipelines()
results["Dummy (prior)"] = run_and_report("Dummy (prior)", pipelines["Dummy (prior)"])

# Result / Interpretation: see Section 8 (computed after every candidate has run).
# Next step:
#   Evaluate Logistic Regression as the primary interpretable reference
#   (Section 6).


# %%
# =============================================================================
# SECTION 6 — Logistic Regression baseline
# =============================================================================
# Objective:
#   Establish the comparatively interpretable statistical/ML reference.
#
# Rationale:
#   The outcome is binary. The shared preprocessor already standard-scales
#   `Age`/`Smoking PY` and one-hot encodes the six categorical predictors,
#   so Logistic Regression receives a numerically well-conditioned design
#   matrix without any extra work. It supplies a comparatively
#   interpretable reference against which any nonlinear model (Decision
#   Tree, Random Forest, RBF SVC) must demonstrate MEANINGFUL added
#   discrimination. A nonlinear model should demonstrate a consistent and
#   practically meaningful improvement over this reference before its
#   additional complexity is preferred. Coefficients are NOT inspected
#   here: that belongs after model selection and refitting in a later
#   milestone, not in a default-family comparison where the model has not
#   yet been chosen -- and even then, because the shared preprocessor's
#   OneHotEncoder uses `drop=None` (no reference level is omitted), a
#   fitted coefficient will not be a simple "category vs. one omitted
#   reference category" effect.

section("SECTION 6 — Logistic Regression baseline (comparatively interpretable reference)")

results["Logistic Regression"] = run_and_report("Logistic Regression", pipelines["Logistic Regression"])

# Result / Interpretation: see Section 8.
# Next step:
#   Evaluate the remaining classical, nonlinear model families under the
#   identical CV design (Section 7).


# %%
# =============================================================================
# SECTION 7 — Classical model-family comparison
# =============================================================================
# Objective:
#   Evaluate the remaining three default-hyperparameter candidates under
#   the identical fixed CV design, so every one of the five candidates is
#   judged on exactly the same folds and the same metrics.
#
# Rationale:
#   Decision Tree and Random Forest can capture non-linear/interaction
#   structure (e.g. between `Ds Site`, `T`, `N`) that a linear model
#   cannot; the RBF SVC can capture a different class of non-linear
#   decision boundary. None receives tuned hyperparameters, class weights,
#   or resampling here -- this section asks only whether the default
#   configuration of each family shows real signal, not what its best
#   achievable performance is.

section("SECTION 7 — Classical model-family comparison (Decision Tree, Random Forest, RBF SVC)")

for name in ["Decision Tree", "Random Forest", "RBF SVC"]:
    results[name] = run_and_report(name, pipelines[name])

# Result / Interpretation: see Section 8.
# Next step:
#   Assemble the full comparison table and interpret it (Section 8).


# %%
# =============================================================================
# SECTION 8 — CV comparison table and interpretation
# =============================================================================
# Objective:
#   Assemble one comparison table across all five candidates and interpret
#   it against the questions that matter -- not by ranking on a single
#   decimal difference.
#
# Rationale:
#   The judgement below is made by comparing each model's mean score
#   against the Dummy baseline and against Logistic Regression, relative
#   to the OBSERVED fold-to-fold standard deviation of both models being
#   compared -- not against an arbitrary fixed margin. No test-set
#   information of any kind is used.

section("SECTION 8 — CV comparison table and interpretation")

table_rows = []
for name, scores in results.items():
    table_rows.append(
        {
            "model": name,
            "roc_auc_mean": scores["roc_auc_mean"],
            "roc_auc_std": scores["roc_auc_std"],
            "average_precision_mean": scores["average_precision_mean"],
            "average_precision_std": scores["average_precision_std"],
            "balanced_accuracy_mean": scores["balanced_accuracy_mean"],
            "balanced_accuracy_std": scores["balanced_accuracy_std"],
        }
    )
comparison_table = pd.DataFrame(table_rows).set_index("model").round(4)
print(comparison_table.to_string())

# --- Empirical check of the Dummy baseline's expected behaviour ------------
# `results[...]["..._mean"/"..._std"]` is declared `np.ndarray | float`
# (the same dict also stores the `..._scores` per-fold arrays under other
# keys, per `modeling.evaluate_candidate`); `float(...)` below is a no-op
# on the float values these specific keys always hold, it only gives the
# type checker a precise scalar type for the arithmetic that follows.
dummy_auc_mean = float(results["Dummy (prior)"]["roc_auc_mean"])
dummy_ap_mean = float(results["Dummy (prior)"]["average_precision_mean"])
train_prevalence = float(y_train.mean())
print(f"\nDummy ROC-AUC mean = {dummy_auc_mean:.4f} (expected ~0.50 for a constant-score classifier).")
print(f"Dummy Average Precision mean = {dummy_ap_mean:.4f} vs. training-set prevalence = "
      f"{train_prevalence:.4f} (expected to track prevalence for a no-signal classifier).")

# --- Logistic Regression vs. Dummy, relative to fold variability -----------
logreg_auc_mean = float(results["Logistic Regression"]["roc_auc_mean"])
logreg_auc_std = float(results["Logistic Regression"]["roc_auc_std"])
print(f"\nLogistic Regression ROC-AUC mean = {logreg_auc_mean:.4f} (std={logreg_auc_std:.4f}) "
      f"vs. Dummy = {dummy_auc_mean:.4f}: "
      f"difference = {logreg_auc_mean - dummy_auc_mean:+.4f}.")

# --- Nonlinear models vs. Logistic Regression, relative to fold variability -
print("\nNonlinear candidates vs. Logistic Regression (ROC-AUC):")
for name in ["Decision Tree", "Random Forest", "RBF SVC"]:
    m = float(results[name]["roc_auc_mean"])
    s = float(results[name]["roc_auc_std"])
    print(f"   {name:16s} mean={m:.4f} (std={s:.4f})  vs. LogReg mean={logreg_auc_mean:.4f} "
          f"(std={logreg_auc_std:.4f})  difference={m - logreg_auc_mean:+.4f}")

print("\nAgreement between ROC-AUC and Average Precision ranking:")
ranking_auc = comparison_table["roc_auc_mean"].sort_values(ascending=False).index.tolist()
ranking_ap = comparison_table["average_precision_mean"].sort_values(ascending=False).index.tolist()
print(f"   Ranked by ROC-AUC:            {ranking_auc}")
print(f"   Ranked by Average Precision:  {ranking_ap}")
print(f"   Same ranking? {'YES' if ranking_auc == ranking_ap else 'NO -- see interpretation below'}")

# Result / Interpretation (against the actual run: see printed table above).
#   No formal significance test is performed anywhere below -- differences
#   are reported as observed mean differences, read alongside the observed
#   fold-to-fold standard deviations descriptively, not as a proof that a
#   gap is "real" or statistically significant.
#
#   roc_auc mean (std)   : Dummy 0.5000 (0.0000) | LogReg 0.7897 (0.0270) |
#                          Tree 0.5850 (0.0213) | Forest 0.7554 (0.0237) |
#                          SVC 0.7285 (0.0239)
#
#   - Dummy vs. real models: Dummy's ROC-AUC is exactly 0.5000 with zero
#     std (a constant per-fold score cannot rank anything), and its
#     Average Precision (0.1889) matches the training-set prevalence
#     (0.1889) to four decimals -- both behave exactly as a no-signal
#     classifier should, confirming the reference empirically rather than
#     by assertion. Every real model's mean ROC-AUC sits well above this
#     reference, from Decision Tree's +0.085 up to Logistic Regression's
#     +0.290 -- all four show observable discrimination on this problem.
#
#   - Logistic Regression has the highest observed mean CV ROC-AUC
#     (0.7897), and also the highest mean Average Precision (0.4404) and
#     mean Balanced Accuracy (0.5879) of all five candidates -- it leads
#     on every metric, not merely "competitive with" the nonlinear
#     candidates. Random Forest's mean ROC-AUC (0.7554) is 0.034 below
#     Logistic Regression's; RBF SVC's (0.7285) is 0.061 below. Both gaps
#     should be read together with the fold-to-fold standard deviations
#     reported above (roughly 0.02-0.03 for every non-Dummy candidate),
#     without treating that comparison as a formal test of significance.
#
#   - Decision Tree vs. Random Forest: Decision Tree is substantially
#     worse than Random Forest on the PRIMARY metric (ROC-AUC: 0.5850 vs.
#     0.7554) and on the SECONDARY metric (Average Precision: 0.2358 vs.
#     0.3815). Its Balanced Accuracy (0.5850) is in fact slightly HIGHER
#     than Random Forest's (0.5783) -- Random Forest does NOT dominate
#     Decision Tree on every metric. This does not compensate for Decision
#     Tree's much weaker ranking/discrimination performance on the two
#     metrics that matter most for this problem, so it is not shortlisted
#     (Section 9).
#
#   - ROC-AUC vs. Average Precision agreement: PARTIAL, not full. Logistic
#     Regression leads on both; Decision Tree and Dummy trail on both. But
#     Random Forest and RBF SVC SWAP order: Random Forest has the higher
#     ROC-AUC (0.7554 vs. 0.7285) while RBF SVC has the higher Average
#     Precision (0.4135 vs. 0.3815). RBF SVC shows a better precision-
#     recall trade-off than Random Forest across thresholds, despite
#     lower overall ROC ranking performance -- a genuine, metric-dependent
#     disagreement worth recording rather than smoothing over. Because
#     both `C` and `gamma` (both left at their scikit-learn defaults here)
#     strongly govern an
#     RBF SVC's flexibility, this default-configuration result is enough
#     to justify one controlled tuning pass rather than a verdict either
#     way on the model family.
#
#   - RBF SVC's Balanced Accuracy (0.5324) is markedly lower than Logistic
#     Regression's (0.5879) or Random Forest's (0.5783), and its
#     fold-to-fold spread is tight (std 0.0066). The comparatively low
#     balanced accuracy indicates that the default hard decision rule
#     yields weak class-balanced classification performance at this
#     stage. Threshold optimisation and class weighting have deliberately
#     not yet been investigated (Section 8 of Milestone 2's design), so no
#     conclusion is drawn here about the model family's ceiling.
#
#   - The Decision Tree (unpruned, default depth) is the most likely
#     candidate to show a training-CV vs. generalisation gap in a later
#     phase, given its tendency to overfit without any depth constraint;
#     noted as a risk, not as grounds to exclude it here, since no tuning
#     has been attempted yet -- but see Section 9 for why it is not
#     shortlisted regardless.
# Decision:
#   The shortlist for Milestone-3 tuning is based only on the training-CV
#   table above (Section 9).
# Next step:
#   Record the Milestone-2 shortlist and stop before any tuning.


# %%
# =============================================================================
# SECTION 9 — Milestone-2 shortlist / STOP before tuning
# =============================================================================
section("SECTION 9 — Milestone-2 shortlist and STOP before tuning")

print(comparison_table.sort_values("roc_auc_mean", ascending=False).to_string())

print("""
MILESTONE-3 SHORTLIST: Logistic Regression, Random Forest, RBF SVC.

This is NOT a claim that all three are equally strong: Logistic Regression
leads on every metric, Random Forest is second, and RBF SVC is third,
included specifically because its two governing hyperparameters are
under-explored at their defaults (see below) -- not because its default
result matches the other two.

Rationale (training CV only; no test-set information used):
  - Logistic Regression is shortlisted as the primary reference: it has
    the highest mean on all three metrics of any candidate here (ROC-AUC
    0.7897, Average Precision 0.4404, Balanced Accuracy 0.5879) -- it is
    not a fallback kept "despite" weaker performance, it is the strongest
    candidate found today, and it is the most transparent model among the
    shortlisted candidates. It is retained regardless of what Milestone-3
    tuning does to the others.
  - Random Forest is shortlisted as the second-strongest candidate on the
    PRIMARY metric: its ROC-AUC (0.7554) is 0.034 below Logistic
    Regression's (0.7897), read alongside both models' fold-to-fold
    standard deviations without treating that as a formal significance
    test. It is a nonlinear tree-ensemble family and warrants a
    controlled tuning pass on depth/estimator count and, if warranted,
    controlled class-weighting in Milestone 3, since none of that has
    been applied to any candidate yet.
  - RBF SVC is shortlisted, despite a lower ROC-AUC than Random Forest
    (0.7285 vs. 0.7554), because it has a HIGHER Average Precision than
    Random Forest (0.4135 vs. 0.3815) -- a genuine, metric-dependent
    disagreement (Section 8), not a reason to exclude it outright. Its
    two governing hyperparameters, `C` and `gamma`, were both left at
    their scikit-learn defaults in this milestone and strongly determine
    an RBF SVC's flexibility, so this default-configuration result is
    sufficient to justify one controlled tuning pass rather than a
    verdict on the model family either way.
  - Decision Tree is NOT shortlisted: it is substantially worse than
    Random Forest on the PRIMARY metric (ROC-AUC: 0.5850 vs. 0.7554) and
    the SECONDARY metric (Average Precision: 0.2358 vs. 0.3815). Its
    Balanced Accuracy (0.5850) is in fact slightly higher than Random
    Forest's (0.5783), but that does not compensate for its much weaker
    ranking/discrimination performance on the two metrics that matter
    most for this problem.
  - The Dummy baseline is retained only as the permanent no-signal
    reference for all future comparisons, not as a modelling candidate.
  - RBF SVC without probability calibration produced usable ROC-AUC/
    Average Precision scores via decision_function, confirming that
    enabling probability estimation was unnecessary for this milestone.

Explicitly NOT done in this script:
  [ ] No hyperparameter has been tuned.
  [ ] No class_weight or resampling has been applied.
  [ ] No probability calibration has been performed.
  [ ] The held-out test set was reconstructed ONLY to verify its size and
      class counts (Section 3), then deleted from the namespace and never
      referenced again -- no test-set metric of any kind was computed.

STOP. Awaiting review before Milestone 3 (tuning of the shortlisted models).
""")
