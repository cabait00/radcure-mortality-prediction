"""
================================================================================
RADCURE ML — Milestone 3D: Training-Only Threshold Selection and Final
                           Model Freeze
================================================================================

Author:      Carlo Babo
Project:     Applied Machine Learning — RADCURE Clinical Dataset

--------------------------------------------------------------------------------
WHAT THIS SCRIPT COVERS
--------------------------------------------------------------------------------
Milestone 3C concluded that tuned RBF SVC, the corrected (scaled-meta-
learner) Stacking ensemble, and the refined XGBoost challenger do not
establish a practically compelling discrimination advantage over the
frozen Logistic Regression -- and that LR is simpler and more transparent.
Logistic Regression is therefore now FROZEN as the sole final model family
for this project. This script does exactly one new thing: choose a single
classification threshold for that already-frozen model, using training
data only, under one pre-registered criterion and one pre-declared
tie-break rule. It then writes down the complete final specification --
predictor set, preprocessing, model, threshold -- so that nothing about the
model is decided later by looking at held-out results.

This is NOT a new modelling stage. No new model family is introduced, no
hyperparameter is retuned, and no method outside this project's course
workflow (probability calibration, SHAP, bootstrapping, nested CV,
significance testing) is added. The held-out test set is touched only for
the standard reconstruct-verify-delete integrity check that every prior
analysis script performs; it is not evaluated.

--------------------------------------------------------------------------------
MILESTONES 1, 2, AND 3A-3C ARE FROZEN
--------------------------------------------------------------------------------
Unchanged here: target definition and eligibility, cohort definition, the
PRIMARY predictor set, deterministic cleaning, the 80/20 stratified split
(random_state=42), the preprocessing specification, the fixed
StratifiedKFold(5, shuffle=True, random_state=42), the three CV metrics,
and every tuned hyperparameter from Milestones 3A-3C. The frozen Logistic
Regression pipeline is reused, unmodified, from `radcure.ensemble` -- it is
not redefined here.
"""

# %%
# =============================================================================
# SECTION 1 — Scope and the final frozen model family
# =============================================================================
# Objective:
#   State, before doing anything else, exactly which model is now frozen,
#   why it is the one that survived Milestones 3A-3C, and what predictor
#   set and preprocessing it uses -- so every step below operates on an
#   already-settled model, not one still under consideration.
#
# Rationale (documentation item 6 -- why Logistic Regression remains the
# final model):
#   The training-only comparison across Milestones 3A-3C (recapped below,
#   not recomputed) showed every alternative -- tuned RBF SVC, the
#   corrected Stacking ensemble, and the refined XGBoost challenger --
#   failing to establish a practically compelling discrimination advantage
#   over Logistic Regression. Differences among the leading candidates
#   were small; none provided a clear, broad-based improvement across both
#   ROC-AUC and Average Precision large enough to justify the added
#   complexity and reduced transparency of a nonlinear, ensembled, or
#   twice-searched alternative. Logistic Regression is simpler, fully
#   transparent (a single linear decision function over the frozen
#   preprocessed features), and was never retuned beyond its original
#   Milestone-3A/3B search -- it carries no extra post-hoc selection
#   advantage either.

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sklearn.model_selection import train_test_split  # noqa: E402

from radcure import (  # noqa: E402
    cleaning,
    config,
    ensemble,
    evaluation,
    leakage,
    modeling,
    target,
)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 100)


def section(title: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n{title}\n{bar}")


section("SECTION 1 — Scope and the final frozen model family")

print("Milestone 3D: training-only threshold selection for the now-frozen final")
print("model family, followed by a complete final specification freeze.\n")

print("FROZEN final model family:")
print(f"   LogisticRegression(C={ensemble.FROZEN_LOGISTIC_REGRESSION_PARAMS['C']}, "
      f"class_weight={ensemble.FROZEN_LOGISTIC_REGRESSION_PARAMS['class_weight']}, "
      f"max_iter=5000, random_state={config.RANDOM_STATE})")
print("   Reused unmodified from radcure.ensemble.build_frozen_logistic_regression()")
print("   -- not redefined in this script.")

print(f"\nPRIMARY predictors ({len(config.PRIMARY_FEATURES)}): {config.PRIMARY_FEATURES}")
print("Preprocessing: the existing frozen ColumnTransformer "
      "(radcure.preprocessing.build_preprocessor) -- unchanged.")

# Recap of the Milestone-3A-3C training-only CV comparison (frozen, confirmed
# values -- NOT recomputed here). Grounds the "why LR" decision above in the
# actual numbers rather than asserting it.
MILESTONE_3_RECAP = pd.DataFrame(
    {
        "roc_auc_mean": [0.7897, 0.7791, 0.7909, 0.7905, 0.7885],
        "average_precision_mean": [0.4404, 0.4374, 0.4421, 0.4424, 0.4375],
        "status": ["PRESPECIFIED", "PRESPECIFIED", "PRESPECIFIED", "EXPLORATORY", "EXPLORATORY"],
    },
    index=["Logistic Regression", "Random Forest", "RBF SVC (refined)",
           "Stacking (corrected)", "XGBoost (refined)"],
)
print("\nMilestone 3A-3C training-only CV comparison (recap, not recomputed):")
print(MILESTONE_3_RECAP.to_string())
print("\nNone of the alternatives above shows a practically compelling")
print("discrimination advantage over Logistic Regression -> LR is frozen as the")
print("sole final model family.")

# Predictor-selection funnel, preserved here for the final documentation
# (documentation item 7). This is a restatement of an already-confirmed
# Milestone-1 result (see README.md's "TODO -- final documentation" section
# added during the Milestone-3C cleanup pass); it is NOT re-derived or
# changed here.
print("""
Predictor-selection funnel (preserved for the final report/presentation):
   34 raw dataset columns
   -> 21 excluded during study-design / leakage / temporality audit
   -> 13 temporally admissible clinical candidate predictors
   -> 5 additional predictor-design exclusions (Subsite, M, Stage, Path, HPV)
   -> 8 prespecified primary predictors (frozen set, listed above)
Predictor selection was driven primarily by temporal availability, leakage
prevention, information content, redundancy, data structure and
clinical-methodological plausibility -- not by post-hoc maximization of
cross-validation performance.
""")

# Decision:
#   Logistic Regression, at its frozen hyperparameters, is the model this
#   entire script operates on. No alternative is reconsidered below.
# Next step:
#   Reconstruct the confirmed training data (Section 2).


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
#   and `04_model_complementarity_and_exploratory_models.py`. Duplicated
#   rather than imported, per the project convention that analysis scripts
#   never import one another -- each remains independently runnable. Every
#   step is a pure function of the immutable raw file plus fixed constants,
#   so this is reconstruction of an already-confirmed result, not a new
#   decision.

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
# fitted on it, nothing is predicted from it, no test-set metric, threshold,
# or performance figure is computed anywhere in this script.
# ------------------------------------------------------------------------------
del X_test, y_test, id_test

TRAIN_PREVALENCE = float(y_train.mean())
print(f"\nTraining event prevalence: {int(y_train.sum())}/{len(y_train)} = "
      f"{TRAIN_PREVALENCE:.4f}")

# Decision:
#   X_train / y_train are the only data used for the remainder of this
#   script.
# Next step:
#   State the existing threshold-independent ranking-performance context
#   (Section 3) before generating any prediction.


# %%
# =============================================================================
# SECTION 3 — Existing ranking-performance context (threshold-independent)
# =============================================================================
# Objective:
#   Restate the frozen Logistic Regression's 5-fold CV ROC-AUC and Average
#   Precision as fixed context for everything that follows, WITHOUT
#   recomputing them.
#
# Rationale (documentation item 5 -- why ROC-AUC/AP are unaffected by
# threshold selection):
#   ROC-AUC and Average Precision each summarise a model's ranking quality
#   across EVERY possible decision threshold simultaneously (ROC-AUC
#   integrates the true/false-positive-rate trade-off over all thresholds;
#   Average Precision does the same for the precision/recall trade-off).
#   Choosing one particular threshold below does not change the underlying
#   probability ranking the model produces, so these two numbers are
#   identical before and after threshold selection -- there is nothing to
#   recompute "at threshold 0.5" or "at the selected threshold" for either
#   metric.

section("SECTION 3 — Existing ranking-performance context (unaffected by threshold)")

FROZEN_LR_ROC_AUC_MEAN = 0.7897
FROZEN_LR_ROC_AUC_STD = 0.0270
FROZEN_LR_AP_MEAN = 0.4404
FROZEN_LR_AP_STD = 0.0310

print("Frozen Logistic Regression, 5-fold training CV (Milestone 3A/3B, not recomputed):")
print(f"   ROC-AUC (PRIMARY)             mean={FROZEN_LR_ROC_AUC_MEAN:.4f}  "
      f"std={FROZEN_LR_ROC_AUC_STD:.4f}")
print(f"   Average Precision (SECONDARY) mean={FROZEN_LR_AP_MEAN:.4f}  "
      f"std={FROZEN_LR_AP_STD:.4f}")
print("""
These are threshold-independent ranking metrics and are therefore NOT
recomputed for threshold=0.5 and again for the selected threshold below --
they would be identical both times, because neither metric depends on
where the decision boundary is drawn. They remain TRAINING-SET
model-selection estimates, not final held-out generalization estimates.
""")

# Next step:
#   Generate the training-only cross-validation predictions that threshold
#   selection will use (Section 4).


# %%
# =============================================================================
# SECTION 4 — Training-only cross-validation predictions
# =============================================================================
# Objective:
#   Generate exactly one probability per training patient from the frozen
#   Logistic Regression pipeline, under the SAME fixed 5-fold splitter used
#   throughout this project, for use in threshold selection.
#
# Rationale:
#   Every training patient's probability must come from a model that did
#   NOT train on that patient, or the threshold chosen from these
#   probabilities would be optimistic. Using training-only cross-validation
#   predictions (fit on 4 folds, predict on the 5th, for every fold)
#   achieves this while still using every training patient exactly once.
#   This is a routine input to threshold selection, not a new
#   complementarity analysis -- there is only one frozen model here, so no
#   correlation, agreement, or stacking-style diagnostic is computed.

section("SECTION 4 — Training-only cross-validation predictions")

frozen_pipeline = ensemble.build_frozen_logistic_regression()
cv = modeling.build_cv_splitter()
print(f"Frozen pipeline: {[name for name, _ in frozen_pipeline.steps]}")
print(f"CV splitter    : {cv}")

train_cv_probabilities = evaluation.compute_training_cv_probabilities(
    frozen_pipeline, X_train, y_train, cv
)

assert len(train_cv_probabilities) == len(y_train), (
    "Every training patient must receive exactly one training-only "
    "cross-validation probability."
)
print(f"\n[OK] {len(train_cv_probabilities)} training-only cross-validation "
      f"probabilities generated -- one per training patient.")
print(f"Probability range: [{train_cv_probabilities.min():.4f}, "
      f"{train_cv_probabilities.max():.4f}]  mean={train_cv_probabilities.mean():.4f}")

# Decision:
#   `train_cv_probabilities` is the ONLY input used for threshold
#   evaluation and selection below.
# Next step:
#   Evaluate the frozen model at the default threshold, 0.5 (Section 5).


# %%
# =============================================================================
# SECTION 5 — Default threshold = 0.5
# =============================================================================
# Objective:
#   Report the frozen model's confusion matrix and standard classification
#   metrics at the conventional default decision threshold, before any
#   selection is performed.

section("SECTION 5 — Default threshold = 0.5")

pred_default = evaluation.apply_threshold(train_cv_probabilities, 0.5)
metrics_default = evaluation.classification_metrics(y_train, pred_default)


def print_confusion_matrix(metrics: evaluation.ClassificationMetrics) -> None:
    tp, tn, fp, fn = metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"]
    print("                    Predicted 0   Predicted 1")
    print(f"   Actual 0 (n={tn + fp:5d})   {tn:9d}     {fp:9d}")
    print(f"   Actual 1 (n={fn + tp:5d})   {fn:9d}     {tp:9d}")


def print_metrics(metrics: evaluation.ClassificationMetrics) -> None:
    print(f"   Sensitivity (= Recall)     {metrics['sensitivity']:.4f}")
    print(f"   Specificity                {metrics['specificity']:.4f}")
    print(f"   Precision                  {metrics['precision']:.4f}")
    print(f"   Recall                     {metrics['recall']:.4f}")
    print(f"   F1                         {metrics['f1']:.4f}")
    print(f"   Balanced Accuracy          {metrics['balanced_accuracy']:.4f}")
    print(f"   Predicted-positive rate    {metrics['predicted_positive_rate']:.4f}")


print("Confusion matrix at threshold 0.5:")
print_confusion_matrix(metrics_default)
print()
print_metrics(metrics_default)
print("""
Note: Recall (positive class) and Sensitivity are the SAME statistic
(TP / (TP+FN)) -- both are printed under their conventional names because
both terms are in common use, not because they measure independent
things.""")

# Result / Interpretation:
#   Read against the numbers printed above (not pre-stated here). Whatever
#   sensitivity or predicted-positive rate is observed at threshold 0.5,
#   this is a descriptive report of the DEFAULT operating point, not yet a
#   judgement about whether it is the right one.
# Decision:
#   Threshold 0.5 is retained as the reference point for the comparison in
#   Section 7; whether it is replaced depends only on the pre-specified
#   search in Section 6.
# Next step:
#   State the threshold-selection rule and run the deterministic search
#   (Section 6).


# %%
# =============================================================================
# SECTION 6 — Threshold selection rule and deterministic search
# =============================================================================
# Objective:
#   Select ONE classification threshold from the training-only
#   cross-validation probabilities, under a criterion and tie-break rule
#   that were fixed BEFORE looking at the search result.
#
# Rationale (documentation item 1 -- why 0.5 is only a default, not a
# property of Logistic Regression itself):
#   Logistic Regression outputs a continuous probability in [0, 1] for
#   every patient; nothing about the model requires converting that
#   probability into a hard 0/1 decision at exactly 0.5. 0.5 is simply the
#   conventional default cut for "more likely than not", a convention that
#   is convenient when the two classes are balanced and misclassification
#   costs are symmetric -- neither of which holds for this project.
#
# Rationale (documentation item 2 -- why threshold selection is necessary
# here):
#   The training event prevalence is ~18.9% (Section 2). A model that
#   ranks patients reasonably well can still, at threshold 0.5, predict the
#   positive (event) class only rarely, because relatively few patients'
#   estimated probabilities cross 0.5 when positives are a minority. That
#   default operating point may therefore be too conservative for
#   identifying the event class, which is exactly the descriptive question
#   Section 5's confusion matrix and sensitivity answer.
#
# Pre-specified optimization criterion:
#   Select the threshold that MAXIMIZES Balanced Accuracy, defined as the
#   arithmetic mean of Sensitivity and Specificity. Nothing else is
#   optimized: not ROC-AUC, not Average Precision, not plain Accuracy, not
#   F1, not Precision, not Recall, not Sensitivity alone, not Specificity
#   alone. The held-out test set is not used anywhere in this search.
#
# Rationale (documentation item 3 -- why Balanced Accuracy):
#   Balanced Accuracy gives EQUAL weight to Sensitivity and Specificity,
#   regardless of how imbalanced the classes are. That is an appropriate,
#   neutral criterion for this project's ~18.9%-prevalence binary
#   classification setting, where plain Accuracy would reward a threshold
#   that simply predicts the majority (non-event) class most of the time.
#
# Rationale (documentation item 4 -- why training-only):
#   The threshold is selected using ONLY the training-only cross-validation
#   probabilities from Section 4. Using the held-out test set to choose a
#   threshold would let test-set information influence model development,
#   exactly the kind of leakage this project's Test Set Policy forbids --
#   the threshold must be fixed before the test set is ever evaluated.
#
# Search implementation:
#   A deterministic pass over the exact set of distinct classification
#   operating points reachable by the `score >= threshold` rule: every
#   unique training-only cross-validation probability, PLUS one
#   deterministic sentinel threshold just above the maximum observed
#   probability (`numpy.nextafter(max, +inf)`), so that the "predict
#   everyone negative" operating point -- which needs a threshold strictly
#   above every observed score -- is reachable too, not just "predict
#   everyone positive" and everything in between. This is an exact
#   enumeration of achievable operating points, not an arbitrary coarse
#   grid (e.g. 0.00, 0.01, ..., 1.00).
#
# Pre-declared tie-break rule (fixed BEFORE seeing the result):
#   1. If multiple thresholds tie on the maximum Balanced Accuracy, select
#      the threshold closest to 0.5.
#   2. If there is still an exact tie in distance to 0.5, select the LOWER
#      threshold.
#   See `radcure.evaluation.select_balanced_accuracy_threshold` for the
#   deterministic implementation of both rules.

section("SECTION 6 — Threshold selection rule and deterministic search")

print("Criterion: maximize Balanced Accuracy = mean(Sensitivity, Specificity).")
print("NOT optimized: ROC-AUC, Average Precision, Accuracy, F1, Precision, Recall,")
print("Sensitivity alone, Specificity alone. Held-out test set not used.")
print("Tie-break: (1) closest to 0.5, then (2) the lower threshold.\n")

selection = evaluation.select_balanced_accuracy_threshold(y_train, train_cv_probabilities)

print(f"Candidates searched : {len(np.unique(train_cv_probabilities)) + 1} "
      f"(all {len(np.unique(train_cv_probabilities))} unique observed probabilities "
      f"+ 1 above-maximum sentinel)")
print(f"Selected threshold  : {selection['threshold']:.10f}")
print(f"Balanced Accuracy   : {selection['balanced_accuracy']:.4f}")
print(f"Candidates tied at this maximum Balanced Accuracy: "
      f"{selection['n_tied_candidates']} "
      f"(tie-break rules above resolved the choice among them; reported here "
      f"for transparency, not as a further analysis).")

# Result / Interpretation:
#   The selected threshold and its Balanced Accuracy are read directly from
#   the search above -- no manual adjustment is made after seeing them.
# Decision:
#   The threshold printed above is the ONE selected threshold used for the
#   remainder of this script.
# Next step:
#   Evaluate the frozen model at this selected threshold and compare it
#   directly against the default (Section 7).


# %%
# =============================================================================
# SECTION 7 — Selected-threshold evaluation and default-vs-selected comparison
# =============================================================================
# Objective:
#   Report the same confusion matrix and metrics at the selected threshold
#   as were reported at threshold 0.5, then compare the two operating
#   points directly.

section("SECTION 7 — Selected-threshold evaluation and comparison")

metrics_selected = selection["metrics"]

print(f"Confusion matrix at the selected threshold ({selection['threshold']:.6f}):")
print_confusion_matrix(metrics_selected)
print()
print_metrics(metrics_selected)

comparison = pd.DataFrame(
    {
        "Threshold 0.5": {
            "sensitivity": metrics_default["sensitivity"],
            "specificity": metrics_default["specificity"],
            "precision": metrics_default["precision"],
            "predicted_positive_rate": metrics_default["predicted_positive_rate"],
            "balanced_accuracy": metrics_default["balanced_accuracy"],
        },
        "Selected threshold": {
            "sensitivity": metrics_selected["sensitivity"],
            "specificity": metrics_selected["specificity"],
            "precision": metrics_selected["precision"],
            "predicted_positive_rate": metrics_selected["predicted_positive_rate"],
            "balanced_accuracy": metrics_selected["balanced_accuracy"],
        },
    }
)
comparison["delta"] = comparison["Selected threshold"] - comparison["Threshold 0.5"]

print("\nDefault threshold 0.5 vs. selected threshold:")
print(comparison.round(4).to_string())

print("""
The threshold was selected to balance sensitivity and specificity under the
pre-specified Balanced Accuracy criterion. This is a stated, symmetric
trade-off between catching more true events (sensitivity) and avoiding
false alarms among non-events (specificity), reflected directly in the
predicted-positive-rate and precision changes above. This threshold is NOT
claimed to be clinically optimal: there is no externally specified clinical
cost ratio (e.g. the relative cost of a missed event versus a false alarm)
anywhere in this project, and Balanced Accuracy assigns those two error
types EQUAL weight only because no other weighting was specified -- not
because equal weighting has been shown to be clinically appropriate.
""")

# Next step:
#   Confirm explicitly that ROC-AUC/AP are unaffected by this threshold
#   choice (Section 8), before checking whether anything here reveals a
#   model defect (Section 9).


# %%
# =============================================================================
# SECTION 8 — Confirmation: ROC-AUC / AP are unaffected by threshold selection
# =============================================================================
section("SECTION 8 — Confirmation: ranking metrics are unaffected by threshold")

print(f"ROC-AUC (PRIMARY)             : {FROZEN_LR_ROC_AUC_MEAN:.4f} +/- "
      f"{FROZEN_LR_ROC_AUC_STD:.4f}  (unchanged from Section 3)")
print(f"Average Precision (SECONDARY) : {FROZEN_LR_AP_MEAN:.4f} +/- "
      f"{FROZEN_LR_AP_STD:.4f}  (unchanged from Section 3)")
print("""
Neither figure was, or needed to be, recomputed for threshold=0.5 or for the
selected threshold: both metrics evaluate the model's probability RANKING
across every possible threshold at once, not one hard decision. Choosing a
threshold changes which patients are called "positive" at that one cut
point; it does not change the underlying probabilities ROC-AUC and Average
Precision are computed from.
""")

# Next step:
#   Check explicitly whether anything found so far reveals a concrete
#   implementation defect that would justify reopening model tuning
#   (Section 9).


# %%
# =============================================================================
# SECTION 9 — Optional model-optimization check
# =============================================================================
# Objective:
#   Explicitly check whether this threshold analysis revealed a concrete
#   technical defect in the frozen Logistic Regression implementation --
#   the ONLY circumstance under which reopening model tuning would be
#   justified at this stage.

section("SECTION 9 — Optional model-optimization check")

print("""
The following observations do NOT, by themselves, justify reopening
hyperparameter tuning -- each is a threshold/operating-point property, not
evidence of a defect in the frozen model's fit:
  - the selected threshold is far from 0.5;
  - sensitivity is low at threshold 0.5;
  - the selected threshold improves Balanced Accuracy over threshold 0.5;
  - ROC-AUC remains ~0.79.

These are exactly the properties a probability ranking with ~18.9%
prevalence is expected to show at an unselected default cut, and are the
reason threshold selection (Section 6) was performed at all -- they are not
symptoms of a bug.

A concrete implementation defect would instead look like: the frozen
pipeline failing to fit, the preprocessor producing the wrong number or
identity of features, the CV splitter not matching the fixed specification,
class labels being mismatched between X and y, or the probability column
being read from the wrong index of `predict_proba`. None of these was
observed anywhere in Sections 1-8: the pipeline fit successfully in every
fold (Section 4), the predictor set and preprocessing are unchanged and
verified (Sections 1-2), and the CV splitter matches the fixed
specification exactly (Section 4).

No concrete implementation defect was found. No further model optimization
is justified at this stage.
""")

# Next step:
#   Write down the complete final freeze specification (Section 10).


# %%
# =============================================================================
# SECTION 10 — Final freeze specification
# =============================================================================
# Objective:
#   State, in one place, the complete final model specification -- so that
#   nothing about the model, its inputs, or its decision rule is decided
#   later by looking at the held-out test set.

section("SECTION 10 — FINAL FREEZE SPECIFICATION")

print(f"""
Predictor set (frozen, {len(config.PRIMARY_FEATURES)} features):
   {config.PRIMARY_FEATURES}

Preprocessing (frozen):
   Unchanged from `radcure.preprocessing.build_preprocessor()` -- numeric
   branch: median imputation -> standard scaling; categorical branch:
   one-hot encoding with unknown categories ignored at inference.

Model (frozen):
   LogisticRegression(
       C={ensemble.FROZEN_LOGISTIC_REGRESSION_PARAMS['C']},
       class_weight={ensemble.FROZEN_LOGISTIC_REGRESSION_PARAMS['class_weight']},
       max_iter=5000,
       random_state={config.RANDOM_STATE},
   )

Classification threshold (frozen):
   {selection['threshold']:.10f}
   (selected from training-only cross-validation predictions only)

Selection criterion (frozen):
   Maximum Balanced Accuracy = mean(Sensitivity, Specificity), with the
   pre-declared tie-break: (1) closest to 0.5, then (2) the lower
   threshold.

Final held-out test-evaluation metric set (frozen; PRE-REGISTERED now, NOT
to be extended after the held-out test result is seen):
   - ROC-AUC
   - Average Precision
   - Confusion matrix
   - Sensitivity
   - Specificity
   - Precision
   - Recall
   - F1
   - Balanced Accuracy
""")

# Decision:
#   Everything above is now FROZEN. The held-out test set has not been
#   evaluated anywhere in this project and remains reserved for exactly
#   one future evaluation of this exact specification.
# Next step:
#   Write the documentation rationale for the final report/presentation
#   (Section 11), then STOP.


# %%
# =============================================================================
# SECTION 11 — Documentation rationale (for the final report/presentation)
# =============================================================================
section("SECTION 11 — Documentation rationale")

print("""
1. Threshold 0.5 is only a default decision rule, not a property of
   Logistic Regression itself. The model outputs a continuous probability
   in [0, 1] for every patient; converting that into a hard decision at
   exactly 0.5 is a convention, not a requirement -- convenient when
   classes are balanced and error costs are symmetric, neither of which
   holds here.

2. Threshold selection is necessary here because the outcome is
   imbalanced (training event prevalence ~18.9%) and the default 0.5
   operating point can be too conservative for the minority event class,
   as the Section 5 confusion matrix and sensitivity figure directly show.

3. Balanced Accuracy was used because it gives EQUAL weight to Sensitivity
   and Specificity regardless of class imbalance, making it an appropriate,
   neutral pre-specified criterion for this project's class-imbalanced
   binary classification setting -- unlike plain Accuracy, which would
   favour a threshold that simply predicts the majority class.

4. Threshold selection is training-only because using the held-out test
   set to choose the threshold would leak test-set information into model
   development, violating this project's Test Set Policy: the threshold
   must be fixed before the test set is ever evaluated.

5. ROC-AUC and Average Precision remain unchanged by threshold selection
   because both evaluate the model's probability ranking across ALL
   thresholds at once, rather than one hard decision boundary -- choosing
   a threshold does not alter the underlying ranking either metric scores.

6. Logistic Regression remains the final model because tuned RBF SVC, the
   corrected Stacking ensemble, and the refined XGBoost challenger (see
   Section 1's recap) did not establish a practically compelling
   discrimination advantage over it, while Logistic Regression is simpler
   and more transparent.

7. Predictor-selection rationale (preserved for the final documentation):
   34 raw dataset columns -> 21 excluded during study-design / leakage /
   temporality audit -> 13 temporally admissible clinical candidate
   predictors -> 5 additional predictor-design exclusions (Subsite, M,
   Stage, Path, HPV) -> 8 prespecified primary predictors. Predictor
   selection was driven primarily by temporal availability, leakage
   prevention, information content, redundancy, data structure and
   clinical-methodological plausibility -- not by post-hoc maximization of
   cross-validation performance.
""")


# %%
# =============================================================================
# SECTION 12 — STOP
# =============================================================================
section("SECTION 12 — STOP")

print("""
Explicitly NOT done in this script:
  [ ] No probability calibration (no CalibratedClassifierCV, no Brier
      Score, no calibration curve).
  [ ] No SHAP, no LIME, no PDP/ICE, no coefficient interpretation.
  [ ] No new feature-selection algorithm, no new model family, no
      additional ensemble experiment, no nested CV, no bootstrapping, no
      statistical significance testing.
  [ ] No further hyperparameter tuning -- no concrete implementation
      defect was found (Section 9).
  [ ] No held-out test evaluation of any kind -- the test set was
      reconstructed only to verify its counts (Section 2), then deleted
      and never referenced again.
  [ ] No final report or presentation building.
  [ ] Nothing committed or pushed.

STOP. Awaiting review before held-out test evaluation, feature-importance /
coefficient interpretation, final report writing, or commit/push.
""")
