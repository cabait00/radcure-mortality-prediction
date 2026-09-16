# %% [markdown-style header — see module docstring below for the full narrative]
"""
================================================================================
RADCURE ML — 04 Complementarity, Stacking, and an Exploratory Challenger
================================================================================

Author:      Carlo Babo
Project:     Applied Machine Learning — RADCURE Clinical Dataset

--------------------------------------------------------------------------------
WHAT THIS SCRIPT COVERS
--------------------------------------------------------------------------------
`analysis/03` tuned three model families and refined the RBF SVC once. Those
configurations are now frozen. This script asks three follow-up questions, all
on the training partition only, and ends by fixing the final model
specification:

  1. Do the three frozen models actually make DIFFERENT errors, or are they
     near-redundant? (training-only out-of-fold complementarity analysis)
  2. Does ONE controlled stacking ensemble of them add discrimination?
  3. Does ONE exploratory gradient-boosting challenger (XGBoost) beat the
     simpler prespecified models?

--------------------------------------------------------------------------------
PRESPECIFIED vs. EXPLORATORY — READ THIS BEFORE READING ANY TABLE
--------------------------------------------------------------------------------
  PRESPECIFIED / CORE  Logistic Regression, Random Forest, RBF SVC
                       -- shortlisted in `analysis/02` from a
                          default-hyperparameter comparison, tuned in
                          `analysis/03`.

  EXPLORATORY          Stacking, XGBoost
                       -- both added POST HOC, after the prespecified
                          comparison had been run and inspected.

The distinction is preserved in every table below and is not quietly dropped
later. A post-hoc model that wins by a small margin has had a weaker test
applied to it than a prespecified one, because the decision to try it at all
was informed by having already seen the prespecified results.

--------------------------------------------------------------------------------
INPUT AND OUTPUT
--------------------------------------------------------------------------------
Consumes  data/processed/modeling_cohort.csv  training partition only
          artifacts/03_tuned_results.json     the frozen tuned configurations
Produces  artifacts/04_oof_scores.csv         one out-of-fold score per
                                              training patient per model
          artifacts/04_oof_metadata.json      provenance of those scores
          artifacts/04_model_comparison.csv   every candidate considered
          artifacts/04_final_model_specification.json
          figures/04_oof_score_correlations.png

No model family is tuned further here; the only search performed anywhere
below is the small exploratory XGBoost grid, which is labelled as such. The
held-out partition is not loaded by this script.
"""

# %%
# =============================================================================
# SECTION 1 — Inputs: frozen training partition and tuned configurations
# =============================================================================

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import Protocol, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sklearn.model_selection import StratifiedKFold, cross_validate  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from radcure import artifacts, config, ensemble, plots, tuning  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 100)
plots.apply_project_style()


def section(title: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n{title}\n{bar}")


BASE_MODEL_NAMES = ["Logistic Regression", "Random Forest", "RBF SVC"]

section("SECTION 1 — Inputs: training partition and tuned configurations")

X_train, y_train, id_train = artifacts.load_training_partition()
assert list(X_train.columns) == config.PRIMARY_FEATURES
assert len(X_train) == config.EXPECTED_TRAIN_N
assert int(y_train.sum()) == config.EXPECTED_TRAIN_EVENTS
print(f"Training partition: n={len(X_train)}  events={int(y_train.sum())}  "
      f"non-events={int((y_train == 0).sum())}")
print("The held-out partition is not loaded by this script.")

tuned = artifacts.load_json(artifacts.TUNED_RESULTS)
TUNED_RESULTS = dict(tuned["models"])

print("\nTuned configurations carried forward from analysis/03:")
for name in BASE_MODEL_NAMES:
    print(f"   {name:22s} {TUNED_RESULTS[name]['best_params']}")

# The frozen pipelines are rebuilt from constants in `radcure.ensemble`, and
# those constants are checked against the persisted search results -- so a
# silent divergence between "what the search selected" and "what gets used"
# would fail here rather than go unnoticed.
FROZEN_PARAMS = {
    "Logistic Regression": ensemble.FROZEN_LOGISTIC_REGRESSION_PARAMS,
    "Random Forest": ensemble.FROZEN_RANDOM_FOREST_PARAMS,
    "RBF SVC": ensemble.FROZEN_RBF_SVC_PARAMS,
}
for name, frozen in FROZEN_PARAMS.items():
    searched = TUNED_RESULTS[name]["best_params"]
    for key, value in searched.items():
        assert frozen[key] == value, (
            f"{name}: analysis/03 selected {key}={value!r}, but radcure.ensemble has "
            f"{key}={frozen[key]!r}. The frozen specification and the search result have "
            f"diverged -- investigate before proceeding."
        )
print("\n[OK] The frozen pipelines in radcure.ensemble match the configurations analysis/03 selected.")

# Class ratio for the exploratory XGBoost `scale_pos_weight`, computed from the
# training partition only -- never hardcoded as a rounded literal.
TRAIN_POSITIVE_COUNT = int(y_train.sum())
TRAIN_NEGATIVE_COUNT = int((y_train == 0).sum())
TRAIN_CLASS_RATIO = TRAIN_NEGATIVE_COUNT / TRAIN_POSITIVE_COUNT
print(f"Training class ratio (negatives/positives) = {TRAIN_NEGATIVE_COUNT}/"
      f"{TRAIN_POSITIVE_COUNT} = {TRAIN_CLASS_RATIO:.6f}")


# %%
# =============================================================================
# SECTION 2 — Freeze the three tuned base models
# =============================================================================
# Objective:
#   Build each frozen model as a full preprocessing -> classifier pipeline with
#   its OWN fresh preprocessor, so fitted preprocessing state is never shared
#   between models.

section("SECTION 2 — Frozen tuned base models")

base_pipelines = ensemble.build_frozen_base_pipelines()
for name, pipe in base_pipelines.items():
    print(f"{name:22s} -> {type(pipe.named_steps['classifier']).__name__}")

preprocessor_ids = {id(p.named_steps["preprocessor"]) for p in base_pipelines.values()}
assert len(preprocessor_ids) == len(base_pipelines), "Base pipelines share a preprocessor object."
assert not hasattr(base_pipelines["RBF SVC"].named_steps["classifier"], "predict_proba")
print("\n[OK] Each frozen pipeline has its own fresh, unfitted preprocessor.")
print("[OK] The frozen RBF SVC exposes decision_function only (probability never enabled).")


# %%
# =============================================================================
# SECTION 3 — Training-only out-of-fold complementarity
# =============================================================================
# Objective:
#   Determine whether the three frozen models are near-redundant or make
#   genuinely different errors -- the factual basis for judging whether an
#   ensemble has any plausible rationale.
#
# Method:
#   Out-of-fold predictions under the SAME fixed 5-fold splitter give each
#   training patient a score from a model that never saw them.
#
#   Spearman (rank) correlation is the PRIMARY pairwise measure, for a concrete
#   reason: ROC-AUC and Average Precision are both rank-based, and the three
#   models do not share a score scale (LR/RF emit probabilities in [0,1]; the
#   SVC emits unbounded decision_function margins, because probability=True is
#   deliberately never enabled). Pearson correlation on such mixed scales
#   conflates scale differences with genuine disagreement, so it is reported as
#   SUPPLEMENTARY only.
#
#   Hard-prediction disagreement uses each estimator's NATIVE decision rule
#   (probability >= 0.5, decision_function >= 0). No threshold is selected or
#   tuned here -- these are the built-in defaults, used only to describe where
#   the models differ.

section("SECTION 3 — Training-only out-of-fold complementarity")

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

print("\nNote the differing score ranges above: these are NOT a common scale, which is")
print("why the rank-based Spearman comparison is the primary measure.")

# --- Pairwise correlations ---------------------------------------------------
pairs = [(a, b) for i, a in enumerate(BASE_MODEL_NAMES) for b in BASE_MODEL_NAMES[i + 1:]]


class _ScalarCorrelationResult(Protocol):
    """Structural description of the one attribute this project reads off a
    `scipy.stats.spearmanr`/`pearsonr` result. Neither function carries a
    return annotation and each delegates to a different code path for some
    argument combinations, so a type checker infers a broad union -- even
    though scipy's own result classes declare `statistic: float`. This
    Protocol lets a checker verify `.statistic` access without depending on
    scipy's private internals.
    """

    statistic: float


def _correlation_statistic(result: object) -> float:
    return cast(_ScalarCorrelationResult, result).statistic


corr_rows = []
for a, b in pairs:
    corr_rows.append({
        "pair": f"{a} vs {b}",
        "spearman_PRIMARY": _correlation_statistic(spearmanr(oof_scores[a], oof_scores[b])),
        "pearson_supplementary": _correlation_statistic(pearsonr(oof_scores[a], oof_scores[b])),
    })
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
        "FN_both": stats["fn_both"], "FN_only_A": stats["fn_only_a"], "FN_only_B": stats["fn_only_b"],
        "FP_both": stats["fp_both"], "FP_only_A": stats["fp_only_a"], "FP_only_B": stats["fp_only_b"],
    })
overlap_table = pd.DataFrame(overlap_rows).set_index("pair")
print("\nOOF hard-prediction disagreement and error overlap (A = first model in the pair):")
print(overlap_table.round(4).to_string())
print(f"\nFor reference, training prevalence: {TRAIN_POSITIVE_COUNT}/{len(y_train)} "
      f"= {y_train.mean():.4f} positive.")

# --- Persist the OOF scores: analysis/05 reuses the LR column ----------------
oof_frame = pd.DataFrame({config.PATIENT_ID_COLUMN: id_train.to_numpy(), "y_true": y_train.to_numpy()})
for name in BASE_MODEL_NAMES:
    oof_frame[name] = oof_scores[name]
oof_path = artifacts.save_table(oof_frame, artifacts.OOF_SCORES)

oof_metadata = {
    "description": (
        "Training-only out-of-fold scores. Each patient's score comes from a copy of the "
        "frozen pipeline fitted on the four folds that did not contain that patient. Row "
        "order matches the frozen training partition exactly."
    ),
    "n_rows": len(oof_frame),
    "cv": tuned["cv"],
    "score_methods": dict(ensemble.OOF_SCORE_METHODS),
    "model_params": {
        name: {k: v for k, v in FROZEN_PARAMS[name].items()} for name in BASE_MODEL_NAMES
    },
}
oof_metadata_path = artifacts.save_json(oof_metadata, artifacts.OOF_METADATA)
print(f"\n[artifact] {oof_path.relative_to(PROJECT_ROOT)}")
print(f"[artifact] {oof_metadata_path.relative_to(PROJECT_ROOT)}")

# %%
# --- FIGURE: OOF score correlations ------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
for ax, (kind, column) in zip(
    axes, [("Spearman (rank) — PRIMARY", "spearman_PRIMARY"), ("Pearson — supplementary", "pearson_supplementary")]
):
    matrix = np.eye(len(BASE_MODEL_NAMES))
    for i, a in enumerate(BASE_MODEL_NAMES):
        for j, b in enumerate(BASE_MODEL_NAMES):
            if i != j:
                key = f"{a} vs {b}" if f"{a} vs {b}" in corr_table.index else f"{b} vs {a}"
                matrix[i, j] = float(cast(float, corr_table.at[key, column]))
    short_labels = ["LR", "RF", "SVC"]
    plots.annotated_heatmap(
        ax, matrix, row_labels=short_labels, col_labels=short_labels,
        cmap="YlGnBu", colorbar_label="correlation",
    )
    ax.set_title(kind)

fig.suptitle(
    "Training-only out-of-fold score correlations between the three tuned models\n"
    "(high values mean the models order patients almost identically, so an ensemble "
    "can only re-weight near-identical information)",
    fontsize=12, fontweight="bold",
)
fig.tight_layout()
plots.save_figure(fig, "04_oof_score_correlations.png")

# Result / Interpretation:
#   Spearman correlations are high throughout and highest for Logistic
#   Regression vs. RBF SVC (~0.99): the two models order patients almost
#   identically, which is consistent with their near-identical ROC-AUC and
#   Average Precision. Random Forest is the most distinct of the three
#   (~0.92 against each of the others) while also having the lowest standalone
#   ROC-AUC, so what it adds is different but not obviously better
#   information. The `only_A_wrong` / `only_B_wrong` columns show the
#   observations one model gets right and the other does not -- what an
#   ensemble could in principle exploit. Note that the high hard-prediction
#   disagreement rate is driven largely by the SVC's class_weight="balanced"
#   operating point rather than by different ranking information; the Spearman
#   column is the check against over-reading it.


# %%
# =============================================================================
# SECTION 4 — One controlled stacking experiment
# =============================================================================
# Objective:
#   Evaluate ONE prespecified stacking configuration of the three frozen base
#   models, under a proper outer cross-validation.
#
# Leakage control (the critical design point) -- TWO levels of CV:
#   - INNER (inside StackingClassifier): builds the meta-features. Every
#     meta-feature for an observation comes from a base learner fitted WITHOUT
#     that observation. Base learners are never fitted on the full training set
#     and then asked for in-sample predictions to train the meta-learner, which
#     would let the meta-learner see optimistic, partly-memorised inputs.
#   - OUTER (cross_validate): scores the ENTIRE stacking procedure, including
#     its internal meta-feature generation, on folds held out from the whole
#     thing.
#
# Meta-feature scale correction (part of this SAME single experiment, not a
# second variant):
#   stack_method="auto" gives the meta-learner two probability columns (LR, RF)
#   and one unbounded decision_function column (SVC). Because the final
#   estimator is an L2-regularised LogisticRegression and L2 is scale-
#   dependent, the final estimator standardises its inputs first. That scaler
#   is fit only on the inner-CV meta-features.
#
# Scope, fixed in advance: no tuning of the stack, no alternative meta-learner,
# no passthrough variant.

section("SECTION 4 — One controlled stacking experiment")

stack = ensemble.build_stacking_classifier()
stack_params = stack.get_params(deep=False)
print(f"Base estimators : {[name for name, _ in stack_params['estimators']]}")

final_estimator = stack_params["final_estimator"]
assert isinstance(final_estimator, Pipeline)
print(f"Final estimator : {' -> '.join(type(step).__name__ for _, step in final_estimator.steps)}")
print(f"stack_method    : {stack_params['stack_method']!r}    passthrough: {stack_params['passthrough']}")

inner_cv = stack_params["cv"]
assert isinstance(inner_cv, StratifiedKFold)
outer_cv = ensemble.build_cv_splitter()
print(f"INNER cv        : {inner_cv}")
print(f"OUTER cv        : {outer_cv}")

t0 = time.time()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    stack_cv = cross_validate(
        stack, X_train, y_train, cv=outer_cv, scoring=tuning.CV_SCORING,
        n_jobs=1, return_train_score=False,
    )
print(f"\nOuter-CV evaluation of the stack completed in {time.time() - t0:.1f}s.")
for w in caught:
    print(f"   [WARNING] {w.category.__name__}: {w.message}")

stacking_result = {}
for metric in tuning.CV_SCORING:
    fold_scores = stack_cv[f"test_{metric}"]
    stacking_result[f"{metric}_mean"] = float(np.mean(fold_scores))
    stacking_result[f"{metric}_std"] = float(np.std(fold_scores))
    print(f"   {metric:19s} mean={np.mean(fold_scores):.4f}  std={np.std(fold_scores):.4f}")

print("\nThese are OUTER-fold means across the training partition -- still")
print("model-selection estimates, not a final unbiased generalization estimate.")


# %%
# =============================================================================
# SECTION 5 — Exploratory XGBoost challenger (first stage)
# =============================================================================
# Objective:
#   Check whether a gradient-boosted tree ensemble offers a materially better
#   training-CV result than the simpler prespecified models.
#
# Scientific status, stated plainly and not softened later:
#   XGBoost is a POST-HOC EXPLORATORY CHALLENGER, added AFTER inspection of the
#   prespecified LR/RF/SVC comparison. It was not in the `analysis/02`
#   shortlist. It is labelled EXPLORATORY in every table below.
#
# Scope: a deliberately small 12-combination grid over `max_depth`,
# `learning_rate` and `scale_pos_weight` only. Same frozen preprocessor, same
# fixed CV, same three metrics, same refit="roc_auc" as every other search.

section("SECTION 5 — Exploratory XGBoost challenger (first stage)")

xgb_result: dict[str, float] | None = None
xgb_best_params: dict | None = None

if not tuning.XGBOOST_AVAILABLE:
    print("[SKIPPED] xgboost is NOT installed in this environment.")
    print(f"          Import error: {tuning.XGBOOST_IMPORT_ERROR}")
    print("          Nothing was installed automatically; this subtask is reported as not run.")
else:
    import xgboost  # noqa: E402

    print(f"xgboost version: {xgboost.__version__}")
    xgb_grid = tuning.build_xgboost_param_grid(TRAIN_CLASS_RATIO)
    print(f"Exploratory grid ({tuning.grid_size(xgb_grid)} combinations):")
    for param, values in xgb_grid.items():
        print(f"   {param:32s} {values}")
    assert tuning.grid_size(xgb_grid) == 12

    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        xgb_search = tuning.build_xgboost_search(TRAIN_CLASS_RATIO)
        xgb_search.fit(X_train, y_train)
    xgb_summary = tuning.summarize_best(xgb_search)
    xgb_best_params = xgb_search.best_params_
    xgb_result = {k: v for k, v in xgb_summary.items() if isinstance(v, float)}

    print(f"\nFitted 12 candidates x {outer_cv.get_n_splits()} folds in {time.time() - t0:.1f}s.")
    print(f"Best params: {xgb_best_params}")
    xgb_view = tuning.as_float_view(xgb_summary)
    for metric in tuning.CV_SCORING:
        print(f"   {metric:19s} mean={xgb_view[f'{metric}_mean']:.4f}  std={xgb_view[f'{metric}_std']:.4f}")
    for w in caught:
        print(f"   [WARNING] {w.category.__name__}: {w.message}")

first_stage_xgb_result = xgb_result

# Result:
#   The first-stage search selected `learning_rate=0.03` and `max_depth=2`,
#   both at the LOWER edge of their tested ranges, while `n_estimators` was
#   held fixed at 300. A small learning rate paired with a fixed, possibly
#   too-low round count cannot be distinguished from "this region is genuinely
#   worse" -- which is exactly what the targeted refinement below resolves.


# %%
# =============================================================================
# SECTION 5b — Targeted XGBoost refinement (FINAL stage)
# =============================================================================
# Objective:
#   Resolve the first-stage lower-boundary hits on `learning_rate` and
#   `max_depth` with ONE targeted search that also searches `n_estimators`,
#   since learning rate and round count trade off against each other in
#   gradient boosting.
#
# Stated before, not after, seeing the result: this refinement is not expected
# to improve on the first stage -- it is run to answer that specific ambiguity
# either way.
#
# Cost, stated openly: this is XGBoost's SECOND search stage, while Logistic
# Regression and Random Forest each received one. Any small XGBoost lead must
# be read with that selection-effort asymmetry in mind.
#
# Pre-declared stopping rule: there is NO third XGBoost search.

section("SECTION 5b — Targeted XGBoost refinement (FINAL stage)")

xgb_refined_result: dict[str, float] | None = None
xgb_refined_best_params: dict | None = None

if not tuning.XGBOOST_AVAILABLE:
    print("[SKIPPED] xgboost is NOT installed -- the first-stage search was already skipped.")
else:
    xgb_refine_grid = tuning.build_xgboost_refinement_param_grid(TRAIN_CLASS_RATIO)
    print("Targeted second-stage grid (does not alter the first-stage grid above):")
    for param, values in xgb_refine_grid.items():
        print(f"   {param:32s} {values}")
    assert tuning.grid_size(xgb_refine_grid) == 54

    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        xgb_refined_search = tuning.build_xgboost_refinement_search(TRAIN_CLASS_RATIO)
        xgb_refined_search.fit(X_train, y_train)
    xgb_refined_summary = tuning.summarize_best(xgb_refined_search)
    xgb_refined_best_params = xgb_refined_search.best_params_
    xgb_refined_result = {k: v for k, v in xgb_refined_summary.items() if isinstance(v, float)}

    print(f"\nFitted 54 candidates x {outer_cv.get_n_splits()} folds in {time.time() - t0:.1f}s.")
    print(f"Refined best params: {xgb_refined_best_params}")
    refined_view = tuning.as_float_view(xgb_refined_summary)
    for metric in tuning.CV_SCORING:
        print(f"   {metric:19s} mean={refined_view[f'{metric}_mean']:.4f}  "
              f"std={refined_view[f'{metric}_std']:.4f}")
    for w in caught:
        print(f"   [WARNING] {w.category.__name__}: {w.message}")

    # `first_stage_xgb_result` is None only when xgboost is unavailable, and
    # this branch is reached only when it IS available -- asserted explicitly
    # because that guarantee spans two separate availability checks.
    assert first_stage_xgb_result is not None
    print(f"\nFirst-stage -> refined (same selection rule, refit='{tuning.REFIT_METRIC}'):")
    for metric in tuning.CV_SCORING:
        before, after = first_stage_xgb_result[f"{metric}_mean"], xgb_refined_result[f"{metric}_mean"]
        print(f"   {metric:19s} {before:.4f} -> {after:.4f}   (delta {after - before:+.4f})")

    refine_flags = tuning.flag_boundary_hits(
        xgb_refined_best_params, xgb_refine_grid,
        ("classifier__max_depth", "classifier__learning_rate", "classifier__n_estimators"),
    )
    if refine_flags:
        for flag in refine_flags:
            print(f"   [BOUNDARY FLAG] {flag}")
        print("   Per the pre-declared rule the search stops here regardless; any flag above "
              "is recorded as an accepted limitation of the reported configuration.")
    else:
        print("   [OK] max_depth, learning_rate and n_estimators are all interior to the grid.")

    print("\n[FROZEN] This is the FINAL XGBoost search; no third stage follows.")


# %%
# =============================================================================
# SECTION 6 — Final training-only comparison across every candidate
# =============================================================================
# Objective:
#   One table containing every candidate considered, with the PRESPECIFIED /
#   EXPLORATORY distinction preserved.

section("SECTION 6 — Final training-only comparison")

final_rows = []
for name in BASE_MODEL_NAMES:
    row: dict[str, object] = {"model": name, "status": "PRESPECIFIED"}
    row.update({
        f"{m}_{s}": TUNED_RESULTS[name][f"{m}_{s}"]
        for m in tuning.CV_SCORING for s in ("mean", "std")
    })
    final_rows.append(row)

stack_row: dict[str, object] = {"model": "Stacking (LR+RF+SVC)", "status": "EXPLORATORY"}
stack_row.update(stacking_result)
final_rows.append(stack_row)

if first_stage_xgb_result is not None:
    row = {"model": "XGBoost (1st stage)", "status": "EXPLORATORY"}
    row.update(first_stage_xgb_result)
    final_rows.append(row)
if xgb_refined_result is not None:
    row = {"model": "XGBoost (refined)", "status": "EXPLORATORY"}
    row.update(xgb_refined_result)
    final_rows.append(row)

display_columns = ["model", "status"] + [
    f"{m}_{s}" for m in tuning.CV_SCORING for s in ("mean", "std")
]
final_table = pd.DataFrame(final_rows)[display_columns]
print(final_table.round(4).to_string(index=False))

print("\nRanked by ROC-AUC (PRIMARY metric):")
print(final_table.sort_values("roc_auc_mean", ascending=False)
      [["model", "status", "roc_auc_mean", "average_precision_mean", "balanced_accuracy_mean"]]
      .round(4).to_string(index=False))

comparison_path = artifacts.save_table(final_table, artifacts.MODEL_COMPARISON)
print(f"\n[artifact] {comparison_path.relative_to(PROJECT_ROOT)}")


# %%
# =============================================================================
# SECTION 7 — The performance plateau, and the final model decision
# =============================================================================

section("SECTION 7 — Performance plateau and final model decision")

indexed = final_table.set_index("model")
leader = str(indexed["roc_auc_mean"].idxmax())
lr_auc = float(cast(float, indexed.at["Logistic Regression", "roc_auc_mean"]))
lr_auc_std = float(cast(float, indexed.at["Logistic Regression", "roc_auc_std"]))
lr_ap_mean = float(
    cast(float, indexed.at["Logistic Regression", "average_precision_mean"])
)
lr_ap_std = float(
    cast(float, indexed.at["Logistic Regression", "average_precision_std"])
)
leader_auc = float(cast(float, indexed.at[leader, "roc_auc_mean"]))

roc_auc_values = indexed["roc_auc_mean"].to_numpy(dtype=float)
auc_spread = float(np.max(roc_auc_values) - np.min(roc_auc_values))

# The candidates that represent a distinct modelling APPROACH: the two
# prespecified core families other than Random Forest, the one stacking
# experiment, and the FINAL (not first-stage) exploratory boosting result --
# the last is included only if xgboost was available to run it. Random Forest
# and the superseded first-stage XGBoost row are excluded here only for this
# narrower illustrative comparison -- both remain in `auc_spread` and in
# every table above, which cover all reported candidates.
key_approaches = ["Logistic Regression", "RBF SVC", "Stacking (LR+RF+SVC)"]
if "XGBoost (refined)" in indexed.index:
    key_approaches.append("XGBoost (refined)")
key_auc_values = indexed.loc[key_approaches, "roc_auc_mean"].to_numpy(dtype=float)
key_auc_spread = float(np.max(key_auc_values) - np.min(key_auc_values))

print(f"Top model by ROC-AUC : {leader} ({leader_auc:.4f})")
print(f"Logistic Regression  : {lr_auc:.4f}   (difference {leader_auc - lr_auc:+.4f})")
print(f"Spread across all {len(final_table)} reported candidates: {auc_spread:.4f} ROC-AUC")
print(f"Spread among the linear/kernel/stacking/refined-boosting approaches: "
      f"{key_auc_spread:.4f} ROC-AUC")
print(f"Logistic Regression's own fold-to-fold std: {lr_auc_std:.4f}")

print(f"""
PERFORMANCE PLATEAU
-------------------
Cross-validated discrimination is very similar across all {len(final_table)}
reported candidate configurations: the total spread between the best and
worst of them ({auc_spread:.4f} ROC-AUC) is smaller than the fold-to-fold
standard deviation of a single model ({lr_auc_std:.4f}). The {len(key_approaches)}
candidates that represent structurally different approaches -- a linear model,
a kernel method, a stacked ensemble of three models, and the refined
gradient-boosting result -- cluster even more tightly, within
{key_auc_spread:.4f} ROC-AUC of each other, all around ~0.79.

This suggests an empirical performance plateau for the current clinical
predictor set and evaluation design: additional model complexity appears to
provide limited incremental value, which points to the predictive information
available in these eight baseline features being a stronger bottleneck than
the choice of model family.

Stated carefully, this is an EMPIRICAL observation about this dataset, this
predictor set and this evaluation design. It is not a theoretical maximum, not
a claim that no better model exists, and not proof that the predictor set has
reached an absolute ceiling. A different feature set -- richer staging detail,
imaging, biomarkers, longitudinal information -- could well move it.

MODEL CHOICE, AND WHEN IT WAS MADE
----------------------------------
Logistic Regression is selected as the final model family NOW, on the
threshold-independent evidence above, and BEFORE any decision threshold is
chosen. The reasons are:

  - essentially equivalent discrimination: no alternative establishes a
    practically compelling advantage on ROC-AUC or Average Precision;
  - simplicity and interpretability: a single linear decision function over
    the frozen preprocessed features, which analysis/07 can describe directly;
  - parsimony: no gain that would justify the added complexity of a kernel
    method, a three-model stack, or hundreds of boosted trees;
  - no extra post-hoc adaptation: Logistic Regression was searched once, while
    the SVC and XGBoost each received two search stages.

Explicitly NOT a reason: Logistic Regression does not have the numerically
highest ROC-AUC here, and the operating-point work in analysis/05 is not
evidence for this choice. The threshold selected there optimises the operating
point of a model that has ALREADY been chosen; it is not a threshold-optimised
competition among finalists, and the Balanced Accuracy it achieves must not be
quoted retrospectively as the reason Logistic Regression was preferred.

STATUS OF EVERY NUMBER ABOVE
----------------------------
All of it is MODEL SELECTION on training folds, not an unbiased generalization
estimate. Each reported score is the best or the chosen candidate among several
evaluated on the same folds; the SVC and XGBoost each received two search
stages. Fold standard deviations describe spread across five overlapping
training folds -- not standard errors, not confidence intervals. No
significance test is performed anywhere in this script.
""")

final_specification = {
    "model_family": "Logistic Regression",
    "estimator": "sklearn.linear_model.LogisticRegression",
    "hyperparameters": {
        **{k: v for k, v in ensemble.FROZEN_LOGISTIC_REGRESSION_PARAMS.items()},
        "max_iter": 5000,
        "random_state": config.RANDOM_STATE,
    },
    "predictors": config.PRIMARY_FEATURES,
    "preprocessing": (
        "radcure.preprocessing.build_preprocessor() -- median imputation + StandardScaler "
        "for the numeric branch; OneHotEncoder(handle_unknown='ignore', drop=None) for the "
        "categorical branch. Fitted training-only, inside the pipeline."
    ),
    "selected_on": [
        "ROC-AUC (primary, threshold-independent)",
        "Average Precision (secondary, imbalance-aware)",
        "simplicity, interpretability and parsimony given equivalent discrimination",
    ],
    "selected_before_threshold_selection": True,
    "cv_roc_auc_mean": lr_auc,
    "cv_roc_auc_std": lr_auc_std,
    "cv_average_precision_mean": lr_ap_mean,
    "cv_average_precision_std": lr_ap_std,
    "plateau_note": (
        f"Spread across all {len(final_table)} reported candidates was {auc_spread:.4f} "
        f"ROC-AUC, smaller than a single model's fold-to-fold standard deviation "
        f"({lr_auc_std:.4f}). The linear, kernel, stacking and refined-boosting approaches "
        f"cluster within {key_auc_spread:.4f} ROC-AUC of each other. Empirical plateau for "
        f"this predictor set and evaluation design; not a theoretical ceiling."
    ),
}
spec_path = artifacts.save_json(final_specification, artifacts.FINAL_MODEL_SPECIFICATION)
print(f"[artifact] {spec_path.relative_to(PROJECT_ROOT)}")

print("""
Next: analysis/05_threshold_selection_and_model_freeze.py — choose one decision
threshold for the now-selected model from training-only out-of-fold
predictions, fit it on the complete training partition, and freeze both.
""")
