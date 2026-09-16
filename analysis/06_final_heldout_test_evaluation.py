# %% [markdown-style header — see module docstring below for the full narrative]
"""
================================================================================
RADCURE ML — 06 Final Held-Out Test Evaluation
================================================================================

Author:      Carlo Babo
Project:     Applied Machine Learning — RADCURE Clinical Dataset

--------------------------------------------------------------------------------
WHAT THIS SCRIPT COVERS
--------------------------------------------------------------------------------
The dedicated held-out evaluation stage of this project. Everything it needs
was decided and frozen before it ran:

  - which predictors to use          (analysis/01)
  - how to preprocess them           (analysis/01)
  - which model family               (analysis/04, on training CV only)
  - its hyperparameters              (analysis/03, on training CV only)
  - which decision threshold         (analysis/05, on training-only
                                      out-of-fold predictions)
  - which metrics to report          (analysis/05, pre-specified)

This script loads the fitted model and the threshold as artefacts -- it does
not refit the model, does not re-derive the threshold, and does not reconstruct
the split. It generates one set of test probabilities, applies the frozen
threshold, and reports exactly the pre-specified metric set.

That discipline is what makes the result meaningful: had the test partition
informed any of the choices above, its performance would no longer estimate how
the model behaves on data it did not help shape.

--------------------------------------------------------------------------------
INPUT AND OUTPUT
--------------------------------------------------------------------------------
Consumes  models/final_logistic_regression.joblib  fitted in analysis/05
          artifacts/05_threshold.json              the frozen threshold
          data/processed/modeling_cohort.csv       held-out partition
Produces  artifacts/06_heldout_predictions.csv     per-patient probabilities
          artifacts/06_heldout_metrics.json        the final metric set
          figures/06_heldout_confusion_matrix.png
          figures/06_heldout_roc_pr_curves.png

--------------------------------------------------------------------------------
THIS IS INTERNAL HELD-OUT EVALUATION, NOT EXTERNAL VALIDATION
--------------------------------------------------------------------------------
The test partition is a stratified random 20% split of the SAME RADCURE cohort
the training data came from -- same data source, same recruitment period and
sites, same measurement process. It answers "how does this exact frozen model
perform on RADCURE patients it did not train on", not "how does it perform on
an independent population, institution or time period". The broader question
needs a genuinely external dataset and is future work.

--------------------------------------------------------------------------------
GOVERNING RULE
--------------------------------------------------------------------------------
Once the metrics below are printed, the model-selection cycle is over. No
feature, hyperparameter, threshold or model-family decision may be revisited in
light of what this script reports -- whether the result looks better, the same,
or worse than the training-stage expectations it is compared against.
"""

# %%
# =============================================================================
# SECTION 1 — Load the frozen model, threshold and held-out partition
# =============================================================================
# Objective:
#   Load the three frozen inputs and verify each against the specification
#   recorded when it was created, before any prediction is generated.

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from radcure import artifacts, config, evaluation, modeling, plots  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 100)
plots.apply_project_style()


def section(title: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n{title}\n{bar}")


section("SECTION 1 — Loading the frozen model, threshold and held-out partition")

# --- The fitted model, loaded not refitted -----------------------------------
final_pipeline = artifacts.load_final_model()
final_lr = final_pipeline.named_steps["classifier"]
assert isinstance(final_lr, LogisticRegression)
lr_config = modeling.describe_logistic_regression(final_lr)
assert lr_config["C"] == 1.0
assert lr_config["class_weight"] is None
assert lr_config["max_iter"] == 5000
assert lr_config["random_state"] == config.RANDOM_STATE
print(f"Fitted model loaded from models/{config.FINAL_MODEL_FILENAME}")
print(f"   LogisticRegression(C={lr_config['C']}, class_weight={lr_config['class_weight']}, "
      f"max_iter={lr_config['max_iter']}, random_state={lr_config['random_state']})")
print("   This script does not refit it.")

# The preprocessing vocabulary was learned in analysis/05, on training data
# only. Confirming it here shows that nothing about this fitted object could
# have seen the held-out partition being loaded below.
encoder = final_pipeline.named_steps["preprocessor"].named_transformers_["categorical"].named_steps["encoder"]
n_encoded = sum(len(cats) for cats in encoder.categories_)
print(f"   Fitted preprocessing: {len(config.PRIMARY_NUMERIC_FEATURES)} numeric + "
      f"{n_encoded} one-hot columns, all learned on the training partition in analysis/05.")

# --- The frozen threshold -----------------------------------------------------
threshold_spec = artifacts.load_json(artifacts.THRESHOLD)
FROZEN_THRESHOLD = threshold_spec["threshold"]
assert threshold_spec["selected_after_model_choice"] is True
assert round(FROZEN_THRESHOLD, 10) == round(config.EXPECTED_FINAL_THRESHOLD, 10), (
    f"Loaded threshold {FROZEN_THRESHOLD!r} does not match the frozen value "
    f"{config.EXPECTED_FINAL_THRESHOLD!r}. STOPPING before any test prediction -- this must "
    f"never be resolved by re-selecting a threshold with the test set in view."
)
print(f"\nFrozen threshold: {FROZEN_THRESHOLD:.17g}")
print(f"   criterion : {threshold_spec['criterion']}")
print(f"   equivalent: {threshold_spec['equivalent_criterion']}")
print("   Threshold 0.5 is NOT evaluated against the test partition anywhere below.")

# --- The held-out partition, loaded here and nowhere else in the project -----
X_test, y_test, id_test = artifacts.load_heldout_partition()
assert list(X_test.columns) == config.PRIMARY_FEATURES
assert len(X_test) == config.EXPECTED_TEST_N
assert int(y_test.sum()) == config.EXPECTED_TEST_EVENTS
assert int((y_test == 0).sum()) == config.EXPECTED_TEST_NONEVENTS
print(f"\nHeld-out partition: n={len(X_test)}  events={int(y_test.sum())}  "
      f"non-events={int((y_test == 0).sum())}  prevalence={y_test.mean() * 100:.2f}%")
print("[OK] Matches the frozen study design.")

print("""
Pre-specified metric set (analysis/05) -- exactly this, nothing added after
seeing the result:
   ROC-AUC · Average Precision · confusion matrix · Sensitivity · Specificity ·
   Precision · Recall · F1 · Balanced Accuracy
(Recall and Sensitivity are the SAME positive-class statistic, TP / (TP+FN);
both are reported because both were pre-specified by name.)
""")


# %%
# =============================================================================
# SECTION 2 — One held-out prediction
# =============================================================================
# Objective:
#   Generate exactly one set of test probabilities and, from them, one set of
#   hard predictions at the frozen threshold.

section("SECTION 2 — One held-out prediction")

test_probabilities = final_pipeline.predict_proba(X_test)[:, 1]
test_predictions = evaluation.apply_threshold(test_probabilities, FROZEN_THRESHOLD)

print(f"[OK] {len(test_probabilities)} held-out probabilities generated from the loaded "
      f"fitted model -- one per test patient.")
print(f"[OK] Hard predictions derived using only the frozen threshold "
      f"({FROZEN_THRESHOLD:.6f}).")
print(f"Probability range: [{test_probabilities.min():.4f}, {test_probabilities.max():.4f}]  "
      f"predicted-positive rate: {test_predictions.mean():.4f}")

predictions_frame = pd.DataFrame({
    config.PATIENT_ID_COLUMN: id_test.to_numpy(),
    "y_true": y_test.to_numpy(),
    "probability": test_probabilities,
    "prediction": test_predictions,
})
predictions_path = artifacts.save_table(predictions_frame, artifacts.HELDOUT_PREDICTIONS)
print(f"\n[artifact] {predictions_path.relative_to(PROJECT_ROOT)}")


# %%
# =============================================================================
# SECTION 3 — The pre-specified metric set
# =============================================================================

section("SECTION 3 — Final held-out metrics")

test_roc_auc = float(roc_auc_score(y_test, test_probabilities))
test_average_precision = float(average_precision_score(y_test, test_probabilities))
test_metrics = evaluation.classification_metrics(y_test, test_predictions)

tn, fp, fn, tp = test_metrics["tn"], test_metrics["fp"], test_metrics["fn"], test_metrics["tp"]
assert tn + fp + fn + tp == config.EXPECTED_TEST_N
assert tp + fn == config.EXPECTED_TEST_EVENTS
assert tn + fp == config.EXPECTED_TEST_NONEVENTS

print(f"ROC-AUC (PRIMARY, threshold-independent)            : {test_roc_auc:.4f}")
print(f"Average Precision (SECONDARY, threshold-independent): {test_average_precision:.4f}")
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

print("\nNo metric beyond the pre-specified set is computed: no plain Accuracy, no Brier")
print("score, no calibration metric, no confidence interval, and nothing added after")
print("seeing this result.")

# --- Frozen expectations ------------------------------------------------------
observed_confusion = {k: test_metrics[k] for k in ("tn", "fp", "fn", "tp")}
assert observed_confusion == config.EXPECTED_HELDOUT_CONFUSION, (
    f"Held-out confusion matrix is {observed_confusion}, expected "
    f"{config.EXPECTED_HELDOUT_CONFUSION}. Investigate the pipeline rather than accepting "
    f"a different result."
)
observed_metrics = {
    "roc_auc": test_roc_auc,
    "average_precision": test_average_precision,
    **{k: test_metrics[k] for k in
       ["sensitivity", "specificity", "precision", "f1", "balanced_accuracy"]},
}
for name, expected in config.EXPECTED_HELDOUT_METRICS.items():
    assert abs(observed_metrics[name] - expected) < config.METRIC_TOLERANCE, (
        f"Held-out {name} is {observed_metrics[name]:.6f}, expected ~{expected}."
    )
print(f"\n[OK] Confusion matrix reproduces the frozen result "
      f"(TN={tn}, FP={fp}, FN={fn}, TP={tp}) and every metric matches its recorded "
      f"value within {config.METRIC_TOLERANCE}.")

# %%
# --- FIGURE: final held-out confusion matrix ---------------------------------
fig, ax = plt.subplots(figsize=(5.6, 4.8))
plots.confusion_matrix_panel(
    ax, tn, fp, fn, tp,
    f"Held-out test partition (n = {len(y_test)}, {int(y_test.sum())} events)\n"
    f"frozen threshold {FROZEN_THRESHOLD:.4f}",
)
ax.set_xlabel(
    f"Sensitivity {test_metrics['sensitivity']:.4f} · Specificity {test_metrics['specificity']:.4f}\n"
    f"Precision {test_metrics['precision']:.4f} · Balanced Accuracy "
    f"{test_metrics['balanced_accuracy']:.4f}",
    fontsize=9,
)
fig.tight_layout()
plots.save_figure(fig, "06_heldout_confusion_matrix.png")

# %%
# --- FIGURE: ROC and Precision-Recall curves ---------------------------------
# Both curves are drawn from the SAME frozen probability vector as every number
# above. They are descriptive final-test plots; nothing is chosen from them.
fpr, tpr, roc_thresholds = roc_curve(y_test, test_probabilities)
precision, recall, _ = precision_recall_curve(y_test, test_probabilities)
test_prevalence = float(y_test.mean())

# Where the frozen threshold sits on the ROC curve.
operating_fpr = 1 - test_metrics["specificity"]
operating_tpr = test_metrics["sensitivity"]

fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5.2))

ax_roc.plot(fpr, tpr, color=plots.NON_EVENT_COLOR, linewidth=2.2,
            label=f"Logistic Regression (AUC = {test_roc_auc:.4f})")
ax_roc.plot([0, 1], [0, 1], color=plots.NEUTRAL_COLOR, linestyle="--", linewidth=1.4,
            label="random classifier (AUC = 0.50)")
ax_roc.plot([operating_fpr], [operating_tpr], marker="o", markersize=10,
            color=plots.ACCENT_COLOR, zorder=5,
            label=f"frozen operating point ({FROZEN_THRESHOLD:.4f})")
ax_roc.set_xlabel("False positive rate (1 - Specificity)")
ax_roc.set_ylabel("True positive rate (Sensitivity)")
ax_roc.set_title("ROC curve — held-out test partition")
ax_roc.set_xlim(-0.01, 1.01)
ax_roc.set_ylim(-0.01, 1.01)
ax_roc.legend(loc="lower right", fontsize=9)

ax_pr.plot(recall, precision, color=plots.EVENT_COLOR, linewidth=2.2,
           label=f"Logistic Regression (AP = {test_average_precision:.4f})")
ax_pr.axhline(test_prevalence, color=plots.NEUTRAL_COLOR, linestyle="--", linewidth=1.4,
              label=f"no-skill = prevalence ({test_prevalence:.4f})")
ax_pr.plot([test_metrics["recall"]], [test_metrics["precision"]], marker="o", markersize=10,
           color=plots.ACCENT_COLOR, zorder=5,
           label=f"frozen operating point ({FROZEN_THRESHOLD:.4f})")
ax_pr.set_xlabel("Recall (Sensitivity)")
ax_pr.set_ylabel("Precision")
ax_pr.set_title("Precision-Recall curve — held-out test partition")
ax_pr.set_xlim(-0.01, 1.01)
ax_pr.set_ylim(0, 1.01)
ax_pr.legend(loc="upper right", fontsize=9)

fig.suptitle(
    "Final held-out discrimination — both curves from the same frozen probability vector\n"
    "(descriptive final evaluation; nothing is selected from these plots)",
    fontsize=12.5, fontweight="bold",
)
fig.tight_layout()
plots.save_figure(fig, "06_heldout_roc_pr_curves.png")


# %%
# =============================================================================
# SECTION 4 — Descriptive comparison with the training-stage context
# =============================================================================
# Objective:
#   Compare the held-out metrics against the training-stage figures,
#   descriptively only.

section("SECTION 4 — Descriptive comparison with the training-stage context")

specification = artifacts.load_json(artifacts.FINAL_MODEL_SPECIFICATION)
training_oof = threshold_spec["training_oof_metrics"]["at_selected_threshold"]

training_context = {
    "roc_auc": specification["cv_roc_auc_mean"],
    "average_precision": specification["cv_average_precision_mean"],
    **{k: training_oof[k] for k in
       ["sensitivity", "specificity", "precision", "f1", "balanced_accuracy"]},
}
comparison = pd.DataFrame({
    "Training-stage context": training_context,
    "Held-out test": observed_metrics,
})
comparison["delta"] = comparison["Held-out test"] - comparison["Training-stage context"]
print(comparison.round(4).to_string())

print("""
This is a DESCRIPTIVE comparison only:
  - the training-stage ROC-AUC and Average Precision are 5-fold
    cross-validation means. Their fold standard deviation describes spread
    across overlapping training folds and is not a confidence interval for the
    single held-out figure above;
  - the training-stage Sensitivity/Specificity/Precision/F1/Balanced Accuracy
    are one set of training-only out-of-fold figures at the selected threshold,
    so only a single-point delta can be shown;
  - a wider or narrower gap in either direction is an observation, not evidence
    that any prior decision was wrong.

The held-out values land close to the training-stage figures on the
operating-point metrics and somewhat above them on the two ranking metrics.
With one internal held-out partition and no inferential uncertainty analysis,
that is reported descriptively and nothing more is claimed from it.
""")


# %%
# =============================================================================
# SECTION 5 — Result recorded, and the evaluation guardrail
# =============================================================================

section("SECTION 5 — Final result recorded")

heldout_payload = {
    "evaluation": "held-out evaluation of the frozen model at the frozen threshold",
    "scope": (
        "INTERNAL held-out evaluation -- a stratified random 20% partition of the same "
        "RADCURE cohort as the training data. Not external validation."
    ),
    "n_test": len(y_test),
    "n_events": int(y_test.sum()),
    "n_non_events": int((y_test == 0).sum()),
    "threshold": FROZEN_THRESHOLD,
    "confusion_matrix": observed_confusion,
    "metrics": observed_metrics,
    "training_stage_context": training_context,
}
metrics_path = artifacts.save_json(heldout_payload, artifacts.HELDOUT_METRICS)
print(f"[artifact] {metrics_path.relative_to(PROJECT_ROOT)}")

print(f"""
FINAL HELD-OUT RESULT
   ROC-AUC              {test_roc_auc:.4f}
   Average Precision    {test_average_precision:.4f}
   Sensitivity / Recall {test_metrics['sensitivity']:.4f}
   Specificity          {test_metrics['specificity']:.4f}
   Precision            {test_metrics['precision']:.4f}
   F1                   {test_metrics['f1']:.4f}
   Balanced Accuracy    {test_metrics['balanced_accuracy']:.4f}
   Confusion matrix     TN={tn}  FP={fp}  FN={fn}  TP={tp}

These results are the project's final descriptive evaluation. They are not
used to modify the predictor set, the model family, any hyperparameter, the
preprocessing, or the threshold -- all of which were frozen before this script
ran. Re-running it on the same frozen inputs reproduces the same numbers, which
is a reproducibility check, not a new experiment.

Next: analysis/07_final_model_interpretation.py — describe what the frozen
model learned, from the same fitted pipeline loaded here.
""")
