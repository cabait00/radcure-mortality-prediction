# %% [markdown-style header — see module docstring below for the full narrative]
"""
================================================================================
RADCURE ML — 05 Threshold Selection and Final Model Freeze
================================================================================

Author:      Carlo Babo
Project:     Applied Machine Learning — RADCURE Clinical Dataset

--------------------------------------------------------------------------------
WHAT THIS SCRIPT COVERS
--------------------------------------------------------------------------------
`analysis/04` selected Logistic Regression as the final model family, on
threshold-independent discrimination and on simplicity, and BEFORE any
threshold was considered. This script does exactly two new things:

  1. chooses a single classification threshold for that already-selected
     model, from training-only out-of-fold predictions, under one
     pre-specified criterion and one pre-declared tie-break rule;
  2. fits the complete frozen pipeline on the full training partition and
     persists it, so `analysis/06` and `analysis/07` consume one fitted model
     rather than each refitting their own.

The model family is not reconsidered here, and no alternative model's
threshold is optimised: this is the operating point of a model that has
already been chosen, not a competition among finalists.

--------------------------------------------------------------------------------
INPUT AND OUTPUT
--------------------------------------------------------------------------------
Consumes  data/processed/modeling_cohort.csv          training partition only
          artifacts/04_final_model_specification.json the selected model
          artifacts/04_oof_scores.csv                 the LR out-of-fold
                                                      probabilities, already
                                                      computed in analysis/04
Produces  artifacts/05_threshold_sweep.csv            metrics at every
                                                      achievable operating point
          artifacts/05_threshold.json                 the selected threshold
          models/final_logistic_regression.joblib     the fitted final pipeline
          figures/05_threshold_tradeoff.png
          figures/05_threshold_confusion_matrices.png

The held-out test partition is not loaded by this script.
"""

# %%
# =============================================================================
# SECTION 1 — Inputs: the selected model and its out-of-fold predictions
# =============================================================================
# Objective:
#   Load the model specification `analysis/04` settled on, the training
#   partition, and the training-only out-of-fold probabilities that threshold
#   selection will use.
#
# Rationale:
#   Every training patient's probability must come from a model that did NOT
#   train on that patient, or a threshold chosen from those probabilities would
#   be optimistic. `analysis/04` already produced exactly that -- one
#   out-of-fold probability per training patient, from the same frozen pipeline
#   under the same fixed 5-fold splitter -- so it is loaded here rather than
#   recomputed. The provenance recorded alongside it is verified below, and the
#   threshold the loaded scores yield is checked against the frozen expected
#   value, so a stale or mismatched artefact cannot pass silently.

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from radcure import (  # noqa: E402
    artifacts,
    config,
    ensemble,
    evaluation,
    modeling,
    plots,
)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 100)
plots.apply_project_style()


def section(title: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n{title}\n{bar}")


section("SECTION 1 — Inputs: the selected model and its out-of-fold predictions")

X_train, y_train, id_train = artifacts.load_training_partition()
assert len(X_train) == config.EXPECTED_TRAIN_N
assert int(y_train.sum()) == config.EXPECTED_TRAIN_EVENTS
TRAIN_PREVALENCE = float(y_train.mean())
print(f"Training partition: n={len(X_train)}  events={int(y_train.sum())}  "
      f"prevalence={TRAIN_PREVALENCE:.4f}")
print("The held-out partition is not loaded by this script.")

specification = artifacts.load_json(artifacts.FINAL_MODEL_SPECIFICATION)
assert specification["model_family"] == "Logistic Regression"
assert specification["selected_before_threshold_selection"] is True
print(f"\nFinal model family (selected in analysis/04): {specification['model_family']}")
print(f"   hyperparameters : {specification['hyperparameters']}")
print(f"   CV ROC-AUC      : {specification['cv_roc_auc_mean']:.4f} "
      f"+/- {specification['cv_roc_auc_std']:.4f}")
print(f"   CV Avg Precision: {specification['cv_average_precision_mean']:.4f} "
      f"+/- {specification['cv_average_precision_std']:.4f}")
print(f"\n   {specification['plateau_note']}")

# --- Load the out-of-fold probabilities and verify their provenance ----------
oof_metadata = artifacts.load_json(artifacts.OOF_METADATA)
assert oof_metadata["n_rows"] == config.EXPECTED_TRAIN_N
assert oof_metadata["cv"]["n_splits"] == config.N_CV_FOLDS
assert oof_metadata["cv"]["shuffle"] is True
assert oof_metadata["cv"]["random_state"] == config.RANDOM_STATE
assert oof_metadata["score_methods"]["Logistic Regression"] == "predict_proba"
assert oof_metadata["model_params"]["Logistic Regression"] == dict(
    ensemble.FROZEN_LOGISTIC_REGRESSION_PARAMS
)

oof_frame = artifacts.load_table(artifacts.OOF_SCORES, dtype={config.PATIENT_ID_COLUMN: "string"})
assert (oof_frame[config.PATIENT_ID_COLUMN].to_numpy() == id_train.to_numpy()).all(), (
    "The out-of-fold artefact's patient order does not match the frozen training partition."
)
assert (oof_frame["y_true"].to_numpy() == y_train.to_numpy()).all()

train_cv_probabilities = oof_frame["Logistic Regression"].to_numpy()
print(f"\n[OK] Loaded {len(train_cv_probabilities)} training-only out-of-fold probabilities "
      f"from analysis/04 -- one per training patient, same CV design, same frozen model.")
print(f"Probability range: [{train_cv_probabilities.min():.4f}, "
      f"{train_cv_probabilities.max():.4f}]  mean={train_cv_probabilities.mean():.4f}")


# %%
# =============================================================================
# SECTION 2 — Why a threshold has to be chosen at all
# =============================================================================
# Objective:
#   Establish what threshold 0.5 actually is, and report the frozen model's
#   operating point there, before any selection is made.
#
# Rationale:
#   Logistic Regression outputs a continuous probability in [0, 1]; nothing
#   about the model requires converting that into a hard 0/1 decision at
#   exactly 0.5. 0.5 is the conventional default cut for "more likely than
#   not" -- convenient when the classes are balanced and misclassification
#   costs are symmetric, neither of which holds here. At ~18.9% prevalence, a
#   model that ranks patients well can still predict the event class only
#   rarely at 0.5, simply because few patients' estimated probabilities cross
#   it.
#
#   ROC-AUC and Average Precision are unaffected by any of this: both summarise
#   the model's ranking across EVERY threshold at once, so choosing one
#   threshold does not change them. There is nothing to recompute "at 0.5" and
#   again "at the selected threshold" for either metric.

section("SECTION 2 — The default operating point (threshold = 0.5)")

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
    print(f"   F1                         {metrics['f1']:.4f}")
    print(f"   Balanced Accuracy          {metrics['balanced_accuracy']:.4f}")
    print(f"   Predicted-positive rate    {metrics['predicted_positive_rate']:.4f}")


print("Confusion matrix at threshold 0.5 (training-only out-of-fold predictions):")
print_confusion_matrix(metrics_default)
print()
print_metrics(metrics_default)
print("\nRecall and Sensitivity are the SAME statistic, TP / (TP+FN); both names appear")
print("because both are in common use, not because they measure different things.")

# Result:
#   Sensitivity is low and the predicted-positive rate far below prevalence:
#   the default cut is conservative about calling the event class, which is
#   exactly the property that motivates choosing a threshold deliberately.


# %%
# =============================================================================
# SECTION 3 — The selection rule, and why it is Youden's J in disguise
# =============================================================================
# Objective:
#   State the criterion and the tie-break rule, fixed BEFORE the search result
#   is seen, and show the algebraic identity that connects them to the course
#   material.
#
# Pre-specified criterion:
#   Maximise Balanced Accuracy, the arithmetic mean of Sensitivity and
#   Specificity. Nothing else is optimised: not ROC-AUC, not Average Precision,
#   not plain Accuracy, not F1, not Precision or Recall alone.
#
#   Balanced Accuracy weights the two error types EQUALLY regardless of class
#   imbalance, which makes it a neutral criterion at ~18.9% prevalence -- unlike
#   plain Accuracy, which would reward a threshold that simply predicts the
#   majority class.
#
# Why training-only:
#   Using the held-out test set to choose a threshold would leak test-set
#   information into model development. The threshold must be fixed before the
#   test set is ever evaluated.
#
# Search implementation:
#   A deterministic pass over the exact set of distinct operating points
#   reachable by the `score >= threshold` rule -- every unique out-of-fold
#   probability, plus one sentinel just above the maximum so the "predict
#   everyone negative" extreme is reachable too. An exact enumeration, not an
#   arbitrary coarse grid.
#
# Pre-declared tie-break: (1) the threshold closest to 0.5; (2) on an exact
# distance tie, the lower threshold.

section("SECTION 3 — Selection rule and the Youden's J identity")

print("""
Balanced Accuracy and Youden's J are the same optimisation problem:

    BA = (Sensitivity + Specificity) / 2

    Specificity = 1 - FPR,  Sensitivity = TPR

    BA = (TPR + 1 - FPR) / 2
       = (1 + (TPR - FPR)) / 2

The right-hand side is a strictly increasing function of (TPR - FPR), so the
threshold that maximises Balanced Accuracy is exactly the threshold that
maximises

    J = TPR - FPR

which is Youden's J statistic -- the point of greatest vertical distance from
the ROC diagonal. Choosing the criterion as "maximum Balanced Accuracy" and
choosing it as "maximum Youden's J" therefore select the same operating point;
they are two names for one rule, not two competing criteria.
""")

selection = evaluation.select_balanced_accuracy_threshold(y_train, train_cv_probabilities)
metrics_selected = selection["metrics"]
SELECTED_THRESHOLD = selection["threshold"]

print(f"Candidates searched : {len(np.unique(train_cv_probabilities)) + 1} "
      f"(every unique observed probability + 1 above-maximum sentinel)")
print(f"Selected threshold  : {SELECTED_THRESHOLD:.17g}")
print(f"Balanced Accuracy   : {selection['balanced_accuracy']:.4f}")
print(f"Candidates tied at this maximum: {selection['n_tied_candidates']} "
      f"(resolved by the pre-declared tie-break, reported for transparency)")

# The frozen expectation: a change beyond floating-point noise means something
# upstream moved and must be investigated, not accepted as a new value.
assert round(SELECTED_THRESHOLD, 10) == round(config.EXPECTED_FINAL_THRESHOLD, 10), (
    f"Selected threshold {SELECTED_THRESHOLD!r} does not reproduce the frozen value "
    f"{config.EXPECTED_FINAL_THRESHOLD!r}. Investigate the upstream artefacts before "
    f"accepting any different threshold."
)
print(f"\n[OK] Reproduces the frozen threshold {config.EXPECTED_FINAL_THRESHOLD!r} "
      f"to 10 decimal places.")

# --- Wording discipline -------------------------------------------------------
print("""
How this threshold may be described:
  - "a training-derived operating point maximising Balanced Accuracy /
     Youden's J", or
  - "a training-only threshold under an equal sensitivity/specificity
     weighting criterion".

How it may NOT be described: as a clinically optimal threshold. No downstream
clinical action was defined for this project and no cost ratio between a missed
event and a false alarm was ever specified, so the equal weighting Balanced
Accuracy applies is a stated modelling convention -- not a demonstration that
equal weighting is clinically appropriate here.
""")


# %%
# =============================================================================
# SECTION 4 — The selected operating point, and the full trade-off curve
# =============================================================================
# Objective:
#   Report the selected operating point against the default, and sweep every
#   achievable threshold so the trade-off is visible rather than asserted.

section("SECTION 4 — Selected operating point and the trade-off curve")

print(f"Confusion matrix at the selected threshold ({SELECTED_THRESHOLD:.6f}):")
print_confusion_matrix(metrics_selected)
print()
print_metrics(metrics_selected)

comparison = pd.DataFrame(
    {
        "Threshold 0.5": {k: metrics_default[k] for k in
                          ["sensitivity", "specificity", "precision", "f1",
                           "balanced_accuracy", "predicted_positive_rate"]},
        "Selected threshold": {k: metrics_selected[k] for k in
                               ["sensitivity", "specificity", "precision", "f1",
                                "balanced_accuracy", "predicted_positive_rate"]},
    }
)
comparison["delta"] = comparison["Selected threshold"] - comparison["Threshold 0.5"]
print("\nDefault threshold 0.5 vs. selected threshold:")
print(comparison.round(4).to_string())

# Frozen expectations for both operating points.
for label, metrics, expected in [
    ("default", metrics_default, config.EXPECTED_TRAIN_OOF_DEFAULT_CONFUSION),
    ("selected", metrics_selected, config.EXPECTED_TRAIN_OOF_SELECTED_CONFUSION),
]:
    actual = {k: metrics[k] for k in ("tn", "fp", "fn", "tp")}
    assert actual == expected, (
        f"Training out-of-fold confusion matrix at the {label} threshold is {actual}, "
        f"expected {expected}."
    )
print("\n[OK] Both training out-of-fold confusion matrices reproduce their frozen values.")

# --- Sweep every achievable operating point ---------------------------------
candidate_thresholds = np.append(
    np.unique(train_cv_probabilities),
    np.nextafter(np.max(train_cv_probabilities), np.inf),
)
sweep_rows = []
for threshold in candidate_thresholds:
    m = evaluation.classification_metrics(
        y_train, evaluation.apply_threshold(train_cv_probabilities, threshold)
    )
    sweep_rows.append({
        "threshold": float(threshold),
        "sensitivity": m["sensitivity"],
        "specificity": m["specificity"],
        "balanced_accuracy": m["balanced_accuracy"],
        "precision": m["precision"],
        "f1": m["f1"],
        "predicted_positive_rate": m["predicted_positive_rate"],
        "youden_j": m["sensitivity"] - (1 - m["specificity"]),
    })
threshold_sweep = pd.DataFrame(sweep_rows)

# The identity from Section 3, checked numerically rather than only argued.
ba_values = threshold_sweep["balanced_accuracy"].to_numpy(dtype=float)
youden_values = threshold_sweep["youden_j"].to_numpy(dtype=float)
assert np.allclose(ba_values, (1.0 + youden_values) / 2.0), (
    "Balanced Accuracy and (1 + Youden's J)/2 disagree -- the identity in Section 3 "
    "does not hold numerically, which should be impossible."
)
# Tie-safe agreement check: the Youden's J value AT the Balanced-Accuracy-best
# index must equal the global Youden's J maximum. This holds whenever the two
# criteria are maximised at the same operating point, including when several
# thresholds tie for that maximum -- it does not depend on which tied index
# `argmax` happens to return first.
best_ba_index = int(np.argmax(ba_values))
assert np.isclose(youden_values[best_ba_index], youden_values.max()), (
    "The Balanced-Accuracy maximum and the Youden's J maximum are not the same "
    "operating point."
)
print("[OK] Balanced Accuracy == (1 + Youden's J) / 2 holds at every one of the "
      f"{len(threshold_sweep)} achievable operating points, and both are maximised "
      f"at the same threshold.")

sweep_path = artifacts.save_table(threshold_sweep, artifacts.THRESHOLD_SWEEP)
print(f"[artifact] {sweep_path.relative_to(PROJECT_ROOT)}  ({len(threshold_sweep)} operating points)")

# %%
# --- FIGURE: the sensitivity / specificity trade-off -------------------------
plot_data = threshold_sweep.loc[threshold_sweep["threshold"] <= 1.0]

fig, ax = plt.subplots(figsize=(10.5, 5.4))
ax.plot(plot_data["threshold"], plot_data["sensitivity"],
        color=plots.EVENT_COLOR, linewidth=2, label="Sensitivity (recall of events)")
ax.plot(plot_data["threshold"], plot_data["specificity"],
        color=plots.NON_EVENT_COLOR, linewidth=2, label="Specificity (recall of non-events)")
ax.plot(plot_data["threshold"], plot_data["balanced_accuracy"],
        color="black", linewidth=2.4, label="Balanced Accuracy = (Sens + Spec) / 2")

ax.axvline(SELECTED_THRESHOLD, color=plots.ACCENT_COLOR, linestyle="-", linewidth=2)
ax.axvline(0.5, color=plots.NEUTRAL_COLOR, linestyle="--", linewidth=1.6)
ax.plot([SELECTED_THRESHOLD], [metrics_selected["balanced_accuracy"]],
        marker="o", markersize=9, color=plots.ACCENT_COLOR, zorder=5)
ax.plot([0.5], [metrics_default["balanced_accuracy"]],
        marker="o", markersize=8, color=plots.NEUTRAL_COLOR, zorder=5)

ax.annotate(
    f"selected threshold = {SELECTED_THRESHOLD:.4f}\n"
    f"BA {metrics_selected['balanced_accuracy']:.4f} · "
    f"Sens {metrics_selected['sensitivity']:.3f} · Spec {metrics_selected['specificity']:.3f}",
    xy=(SELECTED_THRESHOLD, metrics_selected["balanced_accuracy"]),
    xytext=(SELECTED_THRESHOLD + 0.12, 0.90),
    arrowprops={"arrowstyle": "->", "color": plots.ACCENT_COLOR},
    fontsize=9, color=plots.ACCENT_COLOR, fontweight="bold",
)
ax.annotate(
    f"default 0.5\nBA {metrics_default['balanced_accuracy']:.4f} · "
    f"Sens {metrics_default['sensitivity']:.3f} · Spec {metrics_default['specificity']:.3f}",
    xy=(0.5, metrics_default["balanced_accuracy"]),
    xytext=(0.55, 0.30),
    arrowprops={"arrowstyle": "->", "color": plots.NEUTRAL_COLOR},
    fontsize=9, color="black",
)

ax.set_xlabel("Decision threshold applied to the predicted probability")
ax.set_ylabel("Metric value")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.set_title(
    "Operating-point trade-off on training-only out-of-fold predictions\n"
    "(the threshold maximising Balanced Accuracy is equivalently the threshold "
    "maximising Youden's J)"
)
# Legend below the axes: every in-axes region is occupied by a curve or an
# annotation at some x, so an inside placement always collides with something.
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=9)
fig.tight_layout()
plots.save_figure(fig, "05_threshold_tradeoff.png")

# %%
# --- FIGURE: the two operating points side by side ---------------------------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
plots.confusion_matrix_panel(
    axes[0], metrics_default["tn"], metrics_default["fp"],
    metrics_default["fn"], metrics_default["tp"],
    f"Default threshold 0.5\nSens {metrics_default['sensitivity']:.3f} · "
    f"Spec {metrics_default['specificity']:.3f} · BA {metrics_default['balanced_accuracy']:.3f}",
)
plots.confusion_matrix_panel(
    axes[1], metrics_selected["tn"], metrics_selected["fp"],
    metrics_selected["fn"], metrics_selected["tp"],
    f"Selected threshold {SELECTED_THRESHOLD:.4f}\nSens {metrics_selected['sensitivity']:.3f} · "
    f"Spec {metrics_selected['specificity']:.3f} · BA {metrics_selected['balanced_accuracy']:.3f}",
)
fig.suptitle(
    f"Training-only out-of-fold confusion matrices (n = {len(y_train)}, "
    f"{int(y_train.sum())} events)\nshading shows each cell's share of its ACTUAL-class row",
    fontsize=12, fontweight="bold",
)
fig.tight_layout()
plots.save_figure(fig, "05_threshold_confusion_matrices.png")

print("""
Interpretation: the selected threshold trades specificity for sensitivity under
the stated equal-weighting criterion -- many more events are caught, at the cost
of substantially more false alarms among non-events, visible directly in the
predicted-positive rate and precision. Whether that trade is the right one for
a given clinical use would depend on a cost ratio this project never specified.
""")


# %%
# =============================================================================
# SECTION 5 — Fit and persist the final model
# =============================================================================
# Objective:
#   Fit ONE copy of the frozen pipeline on the COMPLETE training partition and
#   persist it, so downstream scripts consume a single fitted model instead of
#   each refitting their own.
#
# Rationale:
#   All learned preprocessing -- the imputation median, the scaler's
#   mean/variance, the encoder's category vocabulary -- is estimated inside
#   this single `Pipeline.fit` call, on training data only. The held-out
#   partition has not been loaded anywhere in this script, so nothing about
#   this fitted object can have seen it.

section("SECTION 5 — Fitting and persisting the final model")

final_pipeline = ensemble.build_frozen_logistic_regression()
final_lr = final_pipeline.named_steps["classifier"]
lr_config = modeling.describe_logistic_regression(final_lr)
assert lr_config["C"] == specification["hyperparameters"]["C"] == 1.0
assert lr_config["class_weight"] is None
assert lr_config["max_iter"] == 5000
assert lr_config["random_state"] == config.RANDOM_STATE
print(f"Pipeline steps: {[name for name, _ in final_pipeline.steps]}")
print(f"Classifier: LogisticRegression(C={lr_config['C']}, "
      f"class_weight={lr_config['class_weight']}, max_iter={lr_config['max_iter']}, "
      f"random_state={lr_config['random_state']})")

final_pipeline.fit(X_train, y_train)
print(f"[OK] Fitted once, on the {len(X_train)} training patients only.")

model_path = artifacts.save_final_model(final_pipeline)
print(f"[artifact] {model_path.relative_to(PROJECT_ROOT)}")

# Round-trip check: the persisted object must score identically to the one
# just fitted, or downstream results would not be the ones verified here.
reloaded = artifacts.load_final_model()
assert np.allclose(
    reloaded.predict_proba(X_train)[:, 1], final_pipeline.predict_proba(X_train)[:, 1]
), "The persisted model does not reproduce the in-memory model's predictions."
print("[OK] The persisted model reproduces the in-memory model's predictions exactly.")


# %%
# =============================================================================
# SECTION 6 — Final freeze specification
# =============================================================================
# Objective:
#   Write down the complete specification -- predictors, preprocessing, model,
#   threshold, and the metric set the final evaluation will report -- so that
#   nothing about the model or its decision rule can be decided later by
#   looking at held-out results.

section("SECTION 6 — Final freeze specification")

threshold_payload = {
    "threshold": SELECTED_THRESHOLD,
    "criterion": "maximum Balanced Accuracy on training-only out-of-fold predictions",
    "equivalent_criterion": "maximum Youden's J (TPR - FPR); BA = (1 + J) / 2",
    "tie_break": "closest to 0.5, then the lower threshold",
    "selected_on": "training-only out-of-fold predictions (analysis/04); no test-set data used",
    "selected_after_model_choice": True,
    "n_candidates_searched": len(candidate_thresholds),
    "n_tied_candidates": int(selection["n_tied_candidates"]),
    "description": (
        "A training-derived operating point under equal sensitivity/specificity "
        "weighting. NOT a clinically optimal threshold: no downstream clinical action "
        "and no cost ratio between a missed event and a false alarm was specified in "
        "this project."
    ),
    "training_oof_metrics": {
        "at_default_0.5": {k: metrics_default[k] for k in
                           ["tn", "fp", "fn", "tp", "sensitivity", "specificity",
                            "precision", "f1", "balanced_accuracy", "predicted_positive_rate"]},
        "at_selected_threshold": {k: metrics_selected[k] for k in
                                  ["tn", "fp", "fn", "tp", "sensitivity", "specificity",
                                   "precision", "f1", "balanced_accuracy",
                                   "predicted_positive_rate"]},
    },
    "final_model_artifact": f"models/{config.FINAL_MODEL_FILENAME}",
    "pre_specified_heldout_metrics": [
        "ROC-AUC", "Average Precision", "confusion matrix", "Sensitivity",
        "Specificity", "Precision", "Recall", "F1", "Balanced Accuracy",
    ],
}
threshold_path = artifacts.save_json(threshold_payload, artifacts.THRESHOLD)

print(f"""
Predictor set (frozen, {len(config.PRIMARY_FEATURES)}):
   {config.PRIMARY_FEATURES}

Preprocessing (frozen):
   radcure.preprocessing.build_preprocessor() -- median imputation ->
   standard scaling for the numeric branch; one-hot encoding with unknown
   categories ignored at inference for the categorical branch. Fitted
   training-only, inside the pipeline.

Model (frozen, fitted and persisted):
   LogisticRegression(C={lr_config['C']}, class_weight={lr_config['class_weight']},
                      max_iter={lr_config['max_iter']}, random_state={lr_config['random_state']})
   models/{config.FINAL_MODEL_FILENAME}

Decision threshold (frozen):
   {SELECTED_THRESHOLD:.17g}
   selected from training-only out-of-fold predictions by maximising Balanced
   Accuracy, equivalently Youden's J, AFTER the model family had already been
   chosen in analysis/04.

Held-out metric set (PRE-SPECIFIED now, not to be extended once the held-out
result is seen):
   ROC-AUC · Average Precision · confusion matrix · Sensitivity · Specificity ·
   Precision · Recall · F1 · Balanced Accuracy

[artifact] {threshold_path.relative_to(PROJECT_ROOT)}
""")

print("""
The model and its threshold are now frozen, before the held-out partition has
been touched for evaluation. Everything analysis/06 reports is therefore a read
of a specification fixed in advance.

Next: analysis/06_final_heldout_test_evaluation.py — load this fitted model and
this threshold, and evaluate them in the dedicated held-out evaluation stage on
the frozen held-out partition.
""")
