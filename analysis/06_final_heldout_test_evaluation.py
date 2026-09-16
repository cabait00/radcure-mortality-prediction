"""
================================================================================
RADCURE ML — Milestone 4: Final Held-Out Test Evaluation
================================================================================

Author:      Carlo Babo
Project:     Applied Machine Learning — RADCURE Clinical Dataset

--------------------------------------------------------------------------------
WHAT THIS SCRIPT COVERS
--------------------------------------------------------------------------------
This is the FIRST and ONLY performance evaluation of this project on the
held-out test set. Every prior script (Milestones 1 through 3D) touched the
test set exactly once each, only to reconstruct it and verify its size and
class counts, then deleted it immediately without computing a single
performance number from it. This script is the single, pre-registered
exception: it fits the fully frozen pipeline on the training set, generates
ONE set of test-set probabilities, applies the ALREADY-SELECTED threshold,
and reports EXACTLY the metric set that was pre-registered in Milestone 3D
-- nothing more, nothing chosen after seeing the result.

--------------------------------------------------------------------------------
WHY THE TEST SET WAS UNTOUCHED UNTIL NOW
--------------------------------------------------------------------------------
Every decision that could have been influenced by test-set performance --
which model family to use, which hyperparameters to tune it to, how to
preprocess the predictors, which predictors to include, and which
classification threshold to apply -- was made using ONLY the training split
(Milestones 1 through 3D). This is not a formality: if the test set had
informed any of those choices, the resulting test performance would no
longer estimate how the model behaves on data it did not help shape, and
the whole point of holding out a test set would be lost. This script is
the one place in the project where that discipline pays off: what follows
is an honest, un-tuned read of the frozen model's performance.

--------------------------------------------------------------------------------
WHAT WAS FROZEN BEFORE THIS SCRIPT WAS EVEN WRITTEN
--------------------------------------------------------------------------------
  - Predictor set:      Age, Sex, ECOG PS, Smoking PY, Smoking Status,
                         Ds Site, T, N (Milestone 1, Section 3.12).
  - Preprocessing:       the existing frozen ColumnTransformer (Milestone 1,
                         Section 4).
  - Model family:        Logistic Regression (Milestone 2 default
                         comparison; Milestone 3A/3B tuning; Milestone 3C
                         confirmed no alternative -- tuned RBF SVC,
                         corrected Stacking, refined XGBoost -- showed a
                         practically compelling advantage over it).
  - Hyperparameters:     C=1.0, class_weight=None, max_iter=5000,
                         random_state=42 (Milestone 3A/3B; re-selected
                         exactly the Milestone-2 default cell).
  - Threshold rule:      maximize Balanced Accuracy on training-only 5-fold
                         cross-validation probabilities, tie-break closest
                         to 0.5 then lower (Milestone 3D).
  - Final metric set:    ROC-AUC, Average Precision, confusion matrix,
                         Sensitivity, Specificity, Precision, Recall, F1,
                         Balanced Accuracy (Milestone 3D, Section 10).

Every one of these was fixed BEFORE this script ran, precisely so that
none of them could be adjusted in response to the test result this script
produces. The threshold in particular is not re-selected here: Section 3
below re-derives it from training data only, as a REPRODUCIBILITY check,
not a new selection step, and the test set plays no part in that
re-derivation.

--------------------------------------------------------------------------------
PROJECT NARRATIVE (preserved for the final report/presentation)
--------------------------------------------------------------------------------
Predictor-selection funnel:
    34 raw dataset columns
    -> 21 excluded during study-design / leakage / temporality audit
    -> 13 temporally admissible clinical candidate predictors
    -> 5 additional predictor-design exclusions (Subsite, M, Stage, Path,
       HPV)
    -> 8 prespecified primary predictors (the frozen set used throughout)

Modelling narrative:
    Dummy / Logistic Regression / Decision Tree / Random Forest / RBF SVC
    (Milestone 2, default-hyperparameter comparison)
    -> controlled hyperparameter tuning of the three shortlisted families
       (Milestone 3A/3B)
    -> exploratory Stacking and XGBoost challengers, added post hoc
       (Milestone 3C)
    -> no practically compelling discrimination advantage over Logistic
       Regression found anywhere in that comparison
    -> Logistic Regression frozen as the sole final model family
    -> training-only threshold selection (Milestone 3D)
    -> THIS SCRIPT: final held-out test evaluation (Milestone 4)

--------------------------------------------------------------------------------
THIS IS INTERNAL HELD-OUT EVALUATION, NOT EXTERNAL VALIDATION
--------------------------------------------------------------------------------
The test set is a stratified random 20% split of the SAME RADCURE cohort
the training set was drawn from (Milestone 1, Section 3.10) -- the same
data source, the same recruitment period and sites, the same measurement
process. It answers "how does this exact frozen model perform on RADCURE
patients it did not train on", not "how does it perform on an independent
population, a different institution, or a later time period". That
broader question is out of scope here and would require a genuinely
external dataset -- future work, not this script.

--------------------------------------------------------------------------------
GOVERNING RULE FOR THIS SCRIPT
--------------------------------------------------------------------------------
Once the metrics below are printed, this project's model-selection cycle
is OVER. No feature, hyperparameter, threshold, or model-family decision
made anywhere else in this project may be revisited in light of what this
script reports -- regardless of whether the result looks better, the same,
or worse than the training-stage expectations it is compared against.
"""

# %%
# =============================================================================
# SECTION 1 — Preflight: frozen specification and scope
# =============================================================================
# Objective:
#   State the complete frozen specification this script evaluates, before
#   touching any data, so every subsequent step is checked against a
#   target fixed in advance rather than described after the fact.

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

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


# The exact full-precision threshold Milestone 3D reported. This script
# re-derives its own copy from training data only (Section 3) and asserts
# it matches this recorded value to 10 decimal places -- it is never taken
# on faith, and never re-selected here.
MILESTONE_3D_REPORTED_THRESHOLD = 0.1692981878

section("SECTION 1 — Preflight: frozen specification and scope")

print("Milestone 4: the FIRST and ONLY held-out test evaluation in this project.")
print(f"\nFROZEN predictors ({len(config.PRIMARY_FEATURES)}): {config.PRIMARY_FEATURES}")
print("FROZEN preprocessing: radcure.preprocessing.build_preprocessor() -- unchanged.")
print("FROZEN model: LogisticRegression(C=1.0, class_weight=None, max_iter=5000, "
      f"random_state={config.RANDOM_STATE})")
print(f"Milestone-3D reported threshold (to be reproduced, not re-selected): "
      f"{MILESTONE_3D_REPORTED_THRESHOLD}")

print("""
Pre-registered final held-out metric set (Milestone 3D, Section 10) -- this
script reports EXACTLY this set, nothing added after seeing the result:
   1. ROC-AUC                 6. Precision
   2. Average Precision       7. Recall
   3. Confusion Matrix        8. F1
   4. Sensitivity             9. Balanced Accuracy
   5. Specificity
   (Recall and Sensitivity are the SAME positive-class statistic,
   TP / (TP+FN); both are reported because both were pre-registered by
   name, not because they measure independent things.)
""")

# Next step:
#   Reconstruct the confirmed split and assert its exact counts before
#   fitting or evaluating anything (Section 2).


# %%
# =============================================================================
# SECTION 2 — Reconstruct the confirmed split
# =============================================================================
# Objective:
#   Deterministically reproduce the exact X_train/y_train/X_test/y_test
#   from Milestone 1, using the SAME reusable functions and the SAME split
#   parameters, and assert every confirmed count BEFORE any fitting or
#   evaluation happens.
#
# Rationale:
#   Identical in substance to Section 2 of every prior Milestone-3 analysis
#   script. Duplicated rather than imported, per the project convention
#   that analysis scripts never import one another. This is reconstruction
#   of an already-confirmed, deterministic result (fixed random_state=42,
#   fixed test_size=0.20, stratified on y) -- not a new split decision. The
#   ONLY difference from every prior script: X_test/y_test are NOT deleted
#   below, because this is the one script in the project that is permitted
#   to use them.

section("SECTION 2 — Reconstructing the confirmed split")

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

# Assert BEFORE any fitting or evaluation -- exactly the confirmed counts,
# no alternative split.
assert list(X_train.columns) == config.PRIMARY_FEATURES
assert len(X_train) == config.EXPECTED_TRAIN_N == 2351
assert int(y_train.sum()) == config.EXPECTED_TRAIN_EVENTS == 444
assert int((y_train == 0).sum()) == config.EXPECTED_TRAIN_NONEVENTS == 1907
assert len(X_test) == config.EXPECTED_TEST_N == 588
assert int(y_test.sum()) == config.EXPECTED_TEST_EVENTS == 111
assert int((y_test == 0).sum()) == config.EXPECTED_TEST_NONEVENTS == 477
assert set(id_train) & set(id_test) == set()

print(f"[OK] Training set: n={len(X_train)}, events={int(y_train.sum())}, "
      f"non-events={int((y_train == 0).sum())}.")
print(f"[OK] Held-out test set: n={len(X_test)}, events={int(y_test.sum())}, "
      f"non-events={int((y_test == 0).sum())}.")
print("[OK] Train/test patient_id sets are disjoint.")
print("[OK] Split parameters: test_size=0.20, stratify=y, random_state=42 -- "
      "no alternative split considered.")

# Decision:
#   X_test / y_test are NOT deleted here -- this is the one script in the
#   project permitted to use them, and Section 7 below uses them exactly
#   once.
# Next step:
#   Reproduce the frozen threshold from TRAINING data only, as a
#   reproducibility check (Section 3), before X_test is touched again.


# %%
# =============================================================================
# SECTION 3 — Reconstruct / verify the frozen threshold (training data only)
# =============================================================================
# Objective:
#   Re-derive the Milestone-3D threshold from scratch, using ONLY
#   X_train/y_train, and assert it matches the recorded value to 10
#   decimal places, BEFORE making a single test-set prediction.
#
# Rationale:
#   This is a REPRODUCIBILITY ASSERTION, not a new model-selection step.
#   The threshold was already selected in Milestone 3D; nothing here
#   searches for a threshold again or considers any alternative. Re-running
#   the exact same deterministic procedure (same frozen pipeline, same
#   fixed 5-fold splitter, same training-only cross-validation predictions,
#   same Balanced-Accuracy selection rule, same tie-break) on the same
#   training data must reproduce the exact same floating-point threshold --
#   confirming that this script's copy of the frozen model and procedure
#   is bit-for-bit consistent with Milestone 3D's, before it is trusted
#   with the one test-set prediction that follows.
#
# If this assertion were to fail, this script is required to STOP before
# any test prediction -- there would be no basis for trusting a threshold
# this script computed independently, and no held-out result should be
# computed against a possibly-inconsistent operating point.

section("SECTION 3 — Reconstructing the frozen threshold (training data only)")

reproduction_pipeline = ensemble.build_frozen_logistic_regression()
cv = modeling.build_cv_splitter()
print(f"Frozen pipeline: {[name for name, _ in reproduction_pipeline.steps]}")
print(f"CV splitter    : {cv}")

reproduction_probabilities = evaluation.compute_training_cv_probabilities(
    reproduction_pipeline, X_train, y_train, cv
)
assert len(reproduction_probabilities) == len(y_train)

reproduction_selection = evaluation.select_balanced_accuracy_threshold(
    y_train, reproduction_probabilities
)
reconstructed_threshold = reproduction_selection["threshold"]

print(f"\nReconstructed threshold (full precision): {reconstructed_threshold!r}")
print(f"Milestone-3D reported threshold          : {MILESTONE_3D_REPORTED_THRESHOLD!r}")

# The strict numeric assertion this script must pass before any test
# prediction is generated.
assert round(reconstructed_threshold, 10) == round(MILESTONE_3D_REPORTED_THRESHOLD, 10), (
    "Reconstructed threshold does not match the Milestone-3D reported "
    "value to 10 decimal places -- STOPPING before any test-set "
    "prediction. See the module docstring: this must never be resolved "
    "by re-selecting a threshold with the test set in view."
)
print("[OK] Reconstructed threshold matches the Milestone-3D reported value "
      "to 10 decimal places -- reproducibility confirmed.")

# Freeze this exact full-precision float locally. This is the ONLY
# threshold used below; no alternative threshold (including 0.5) is
# evaluated against the test set anywhere in this script.
FROZEN_THRESHOLD = reconstructed_threshold
print(f"\n[FROZEN] Threshold locked for the one test evaluation below: "
      f"{FROZEN_THRESHOLD!r}")

# Decision:
#   The threshold is now fixed for the remainder of this script. It is not
#   revisited, re-derived, or replaced regardless of anything observed
#   below.
# Next step:
#   Fit the final pipeline, once, on the full training set (Section 4).


# %%
# =============================================================================
# SECTION 4 — Final fit
# =============================================================================
# Objective:
#   Fit ONE fresh copy of the complete frozen pipeline on ALL of
#   X_train/y_train -- the only fit in this project whose purpose is
#   deployment-shaped (train on the full training split), rather than
#   cross-validated model comparison or threshold selection.
#
# Rationale:
#   The preprocessing (imputation statistics, scaling parameters, one-hot
#   category vocabulary) must be learned EXCLUSIVELY from X_train, inside
#   this single `Pipeline.fit` call. `X_test` has not been passed to any
#   `fit`-like call anywhere above, and is not passed to one here either --
#   this is what makes the test-set prediction in Section 7 an honest
#   generalization check rather than one that has already leaked test
#   information into the fitted preprocessing.

section("SECTION 4 — Final fit on the full training set")

final_pipeline = ensemble.build_frozen_logistic_regression()
final_lr = final_pipeline.named_steps["classifier"]
assert isinstance(final_lr, LogisticRegression)
lr_config = modeling.describe_logistic_regression(final_lr)
assert lr_config["C"] == 1.0
assert lr_config["class_weight"] is None
assert lr_config["max_iter"] == 5000
assert lr_config["random_state"] == config.RANDOM_STATE == 42
print(f"Final pipeline built: {[name for name, _ in final_pipeline.steps]}")
print(f"Classifier hyperparameters confirmed: C={lr_config['C']}, "
      f"class_weight={lr_config['class_weight']}, max_iter={lr_config['max_iter']}, "
      f"random_state={lr_config['random_state']}")

final_pipeline.fit(X_train, y_train)
print("[OK] Final pipeline fit exactly once, on X_train/y_train only.")

# Decision:
#   `final_pipeline` is now fitted and frozen for the remainder of this
#   script -- it is not refit, and X_test has not been used in this fit.
# Next step:
#   Generate the one held-out test prediction (Section 5).


# %%
# =============================================================================
# SECTION 5 — One held-out test prediction
# =============================================================================
# Objective:
#   Generate exactly one set of test-set probabilities and, from it,
#   exactly one set of hard predictions at the frozen threshold -- the
#   single held-out prediction this entire project has been building
#   toward.
#
# Rationale:
#   Threshold 0.5 was a training-stage comparison point only (Milestone
#   3D, Section 5); it is deliberately NOT evaluated here. No alternative
#   model (SVC, Random Forest, Stacking, XGBoost) is evaluated on the test
#   set anywhere in this script -- Milestone 3C already established, using
#   training data only, that none of them showed a practically compelling
#   advantage over Logistic Regression, and re-litigating that comparison
#   with test-set information in view is exactly what a held-out test set
#   exists to prevent.

section("SECTION 5 — One held-out test prediction")

test_probabilities = final_pipeline.predict_proba(X_test)[:, 1]
print(f"[OK] {len(test_probabilities)} held-out test probabilities generated -- "
      f"one per test patient, from the single fit in Section 4.")

test_predictions = (test_probabilities >= FROZEN_THRESHOLD).astype(int)
print(f"[OK] Hard predictions derived using ONLY the frozen threshold "
      f"({FROZEN_THRESHOLD!r}). Threshold=0.5 is NOT evaluated on the test set.")

# Decision:
#   `test_probabilities` and `test_predictions` are the ONLY held-out
#   predictions computed in this project. Neither is recomputed below.
# Next step:
#   Compute the pre-registered final metric set (Section 6).


# %%
# =============================================================================
# SECTION 6 — Final held-out metrics (pre-registered set, nothing added)
# =============================================================================
# Objective:
#   Compute exactly the metric set pre-registered in Milestone 3D --
#   ROC-AUC, Average Precision, confusion matrix, Sensitivity, Specificity,
#   Precision, Recall, F1, Balanced Accuracy -- and nothing else.

section("SECTION 6 — Final held-out metrics")

test_roc_auc = roc_auc_score(y_test, test_probabilities)
test_average_precision = average_precision_score(y_test, test_probabilities)

test_metrics = evaluation.classification_metrics(y_test, test_predictions)

tn, fp, fn, tp = test_metrics["tn"], test_metrics["fp"], test_metrics["fn"], test_metrics["tp"]
assert tn + fp + fn + tp == 588
assert tp + fn == 111
assert tn + fp == 477
print("[OK] Confusion-matrix counts sum to the confirmed test-set counts "
      "(n=588, events=111, non-events=477).")

print(f"\nROC-AUC (PRIMARY, threshold-independent)             : {test_roc_auc:.4f}")
print(f"Average Precision (SECONDARY, threshold-independent)  : {test_average_precision:.4f}")

print(f"\nConfusion matrix at the frozen threshold ({FROZEN_THRESHOLD:.6f}):")
print("                    Predicted 0   Predicted 1")
print(f"   Actual 0 (n={tn + fp:5d})   {tn:9d}     {fp:9d}")
print(f"   Actual 1 (n={fn + tp:5d})   {fn:9d}     {tp:9d}")

print(f"\nSensitivity (= Recall)     {test_metrics['sensitivity']:.4f}")
print(f"Specificity                {test_metrics['specificity']:.4f}")
print(f"Precision                  {test_metrics['precision']:.4f}")
print(f"Recall                     {test_metrics['recall']:.4f}")
print(f"F1                         {test_metrics['f1']:.4f}")
print(f"Balanced Accuracy          {test_metrics['balanced_accuracy']:.4f}")
print("""
Note: Recall (positive class) and Sensitivity are the SAME statistic
(TP / (TP+FN)). Both are reported because both were pre-registered by
name in Milestone 3D, not because they are independent measurements.

No metric beyond this pre-registered set is computed: no plain Accuracy,
no Brier Score, no calibration metric, no Matthews Correlation
Coefficient, no log loss, no confidence interval, and no metric added
after seeing this result.
""")

# Next step:
#   Compare descriptively against the training-stage expectations and
#   interpret (Section 7), then STOP.


# %%
# =============================================================================
# SECTION 7 — Descriptive comparison with training-CV expectations
# =============================================================================
# Objective:
#   Compare the held-out metrics against the training-stage context,
#   descriptively only -- no significance claim, no re-optimization, no
#   change to any frozen decision based on this comparison.

section("SECTION 7 — Descriptive comparison with training-CV expectations")

TRAINING_CONTEXT = {
    "roc_auc": 0.7897,
    "average_precision": 0.4404,
    "sensitivity": 0.7590,
    "specificity": 0.6901,
    "precision": 0.3631,
    "f1": 0.4913,
    "balanced_accuracy": 0.7245,
}
held_out_values = {
    "roc_auc": test_roc_auc,
    "average_precision": test_average_precision,
    "sensitivity": test_metrics["sensitivity"],
    "specificity": test_metrics["specificity"],
    "precision": test_metrics["precision"],
    "f1": test_metrics["f1"],
    "balanced_accuracy": test_metrics["balanced_accuracy"],
}

comparison = pd.DataFrame(
    {
        "Training-stage context": TRAINING_CONTEXT,
        "Held-out test": held_out_values,
    }
)
comparison["delta"] = comparison["Held-out test"] - comparison["Training-stage context"]
print(comparison.round(4).to_string())

print("""
This is a DESCRIPTIVE comparison only:
  - the training-stage ROC-AUC/AP are 5-fold CROSS-VALIDATION means with a
    reported fold standard deviation; that standard deviation describes
    spread across overlapping TRAINING folds and is not a confidence
    interval for the single held-out figure above, and no significance
    claim is made from either number.
  - the training-stage Sensitivity/Specificity/Precision/F1/Balanced
    Accuracy are themselves ONE set of training-only cross-validation
    figures at the selected threshold, not a distribution with its own
    reported spread -- so only a single-point descriptive delta is shown.
  - a wider or narrower gap in either direction is reported as an
    observation, not as evidence that any prior decision was wrong.
""")

# Result / Interpretation:
#   Interpretation is written against the actual numbers computed above
#   (Section 6), not pre-supposed here. In general terms this project
#   commits to stating: whether the held-out ROC-AUC/AP land close to the
#   training-CV means (broadly supporting the training-stage ranking
#   expectation) or further from them (weakening it) is reported plainly;
#   whether the held-out Sensitivity/Specificity/Balanced Accuracy land
#   close to their training-only counterparts (broadly supporting the
#   threshold's expected operating point) or further from them is
#   likewise reported plainly. The held-out values are numerically
#   different from the training-CV means. With a single internal held-out
#   test set and no inferential uncertainty analysis, these differences
#   are interpreted descriptively only -- whichever way the comparison
#   falls, it is not grounds to reopen any frozen decision.
# Decision:
#   NONE of the following is done, regardless of the comparison above:
#     - no feature is changed
#     - no threshold is changed
#     - no hyperparameter is changed
#     - no model family is switched
#     - no additional model (including XGBoost) is run or evaluated here
#     - no metric is added beyond the pre-registered set
#   This held-out result is FINAL. It is an INTERNAL held-out evaluation
#   from the same RADCURE cohort the training data came from -- it
#   estimates how this exact frozen model performs on RADCURE patients it
#   did not train on, not how it would perform on an independent
#   population, institution, or time period. That broader question would
#   require a genuinely external dataset and is future work, not part of
#   this project's model-selection cycle.
# Next step:
#   STOP (Section 8).


# %%
# =============================================================================
# SECTION 8 — STOP
# =============================================================================
section("SECTION 8 — STOP")

print("""
This project's model-selection cycle is now COMPLETE. Regardless of
whether the held-out ROC-AUC printed above is above 0.80, around 0.79, or
below 0.79:

  [ ] No feature was changed based on this result.
  [ ] No threshold was changed based on this result.
  [ ] No hyperparameter was changed based on this result.
  [ ] No model family was switched based on this result.
  [ ] No additional model (XGBoost or otherwise) was run or evaluated on
      the test set.
  [ ] No metric was added beyond the pre-registered set.
  [ ] No confidence interval or bootstrap estimate was computed.
  [ ] Nothing committed or pushed.

Any further model development would require a new, independent validation
dataset and belongs to future work -- not to this project's
model-selection cycle, which is now closed.

STOP for review.
""")
