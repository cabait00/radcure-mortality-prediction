# %% [markdown-style header — see module docstring below for the full narrative]
"""
================================================================================
RADCURE ML — 03 Controlled Hyperparameter Tuning
================================================================================

Author:      Carlo Babo
Project:     Applied Machine Learning — RADCURE Clinical Dataset

--------------------------------------------------------------------------------
WHAT THIS SCRIPT COVERS
--------------------------------------------------------------------------------
`analysis/02` compared five default-hyperparameter model families and
shortlisted three: Logistic Regression (best default), Random Forest
(second-best on the primary metric), and RBF SVC (weaker ROC-AUC than Random
Forest but a better Average Precision, with `C`/`gamma` left at scikit-learn
defaults). This script runs one controlled `GridSearchCV` per shortlisted
family, under the IDENTICAL fixed CV design, and compares each tuned result
against its own default.

--------------------------------------------------------------------------------
INPUT AND OUTPUT
--------------------------------------------------------------------------------
Consumes  data/processed/modeling_cohort.csv   training partition only
          artifacts/02_baseline_cv_results.csv the default-hyperparameter
                                               benchmark, loaded rather than
                                               retyped as literals
Produces  artifacts/03_tuned_results.json      selected configuration + CV
                                               metrics per family
          artifacts/03_svc_grid_results.csv    the full first-stage SVC surface
          figures/03_default_vs_tuned.png
          figures/03_svc_gridsearch_heatmap.png

--------------------------------------------------------------------------------
DESIGN, FIXED BEFORE ANY SEARCH IS RUN
--------------------------------------------------------------------------------
  - Three families, each over a FIXED, pre-declared grid: Logistic Regression
    (10 combinations), Random Forest (96), RBF SVC (60). No grid is changed
    after seeing a result.
  - The SAME StratifiedKFold(5, shuffle=True, random_state=42) and the SAME
    three metrics as `analysis/02`, so tuned and default results are directly
    comparable.
  - `GridSearchCV(..., refit="roc_auc")`: selection uses the PRIMARY metric
    only. Average Precision and Balanced Accuracy are always read off the SAME
    selected row, never independently re-optimised.
  - Search-boundary hits on continuous hyperparameters are flagged for review.
    The SVC receives exactly one local refinement (Section 5b) because its
    first-stage `C` landed on a grid edge; the stopping rule is pre-declared
    and there is no third stage.

The held-out test partition is not loaded anywhere in this script.

Tested environment: see the pinned requirements.txt in the repository root.
"""

# %%
# =============================================================================
# SECTION 1 — Inputs: frozen training partition and the default benchmark
# =============================================================================
# Objective:
#   Load the training partition and the default-hyperparameter results from
#   `analysis/02`, which together are the fixed benchmark every tuned result
#   below is measured against.

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sklearn.model_selection import GridSearchCV  # noqa: E402

from radcure import artifacts, config, modeling, plots, tuning  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 100)
plots.apply_project_style()


def section(title: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n{title}\n{bar}")


section("SECTION 1 — Inputs: training partition and default benchmark")

X_train, y_train, id_train = artifacts.load_training_partition()
assert list(X_train.columns) == config.PRIMARY_FEATURES
assert len(X_train) == config.EXPECTED_TRAIN_N
assert int(y_train.sum()) == config.EXPECTED_TRAIN_EVENTS
print(f"Training partition: n={len(X_train)}  events={int(y_train.sum())}  "
      f"non-events={int((y_train == 0).sum())}")
print("The held-out partition is not loaded by this script.")

# The default benchmark comes from the persisted analysis/02 results, so the
# two scripts can never drift apart.
baseline_results = artifacts.load_table(artifacts.BASELINE_CV_RESULTS)
default_wide = baseline_results.pivot(index="model", columns="metric", values=["mean", "std"])
default_wide.columns = [f"{metric}_{stat}" for stat, metric in default_wide.columns]
DEFAULT_RESULTS = default_wide.to_dict(orient="index")

shortlist = artifacts.load_json(artifacts.BASELINE_SHORTLIST)
TUNED_FAMILIES = shortlist["shortlisted"]
print(f"\nShortlisted by analysis/02: {TUNED_FAMILIES}")
print("\nDefault-hyperparameter benchmark (loaded from artifacts/02_baseline_cv_results.csv):")
print(default_wide.loc[TUNED_FAMILIES].round(4).to_string())


# %%
# =============================================================================
# SECTION 2 — Tuning design and search spaces
# =============================================================================
# Objective:
#   Fix the CV design, the scoring metrics, the selection rule and the exact
#   search spaces for all three families, before any search is run.
#
# Rationale:
#   Reusing the IDENTICAL splitter and metric set as `analysis/02` (imported
#   from `radcure.modeling`, not re-implemented) means any change in a mean
#   score reflects the hyperparameter search, not a change in how models are
#   scored.

section("SECTION 2 — Tuning design and search spaces")

cv = tuning.build_cv_splitter()
cv_attrs = modeling.describe_cv_splitter(cv)
print(f"CV splitter: StratifiedKFold(n_splits={cv.get_n_splits()}, shuffle={cv_attrs.shuffle}, "
      f"random_state={cv_attrs.random_state}) -- identical to analysis/02.")
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
print("\n[OK] Grid sizes match the confirmed design: 10 / 96 / 60.")

print(
    "\nParallelism note (Random Forest): RandomForestClassifier is fixed with n_jobs=-1 "
    "and already parallelises tree-building across all cores on EACH fit, so its "
    "GridSearchCV uses n_jobs=1 (sequential over the 96 x 5 = 480 fits) to avoid two "
    "levels of parallelism oversubscribing the same cores. Logistic Regression and RBF "
    "SVC fit a single model per call with no internal parallelism, so their GridSearchCV "
    "uses n_jobs=-1."
)

results: dict[str, tuning.SearchSummary] = {}
searches: dict[str, GridSearchCV] = {}


def report_search(name: str, summary: tuning.SearchSummary) -> None:
    view = tuning.as_float_view(summary)
    for metric in tuning.CV_SCORING:
        print(f"   {metric:19s} mean={view[f'{metric}_mean']:.4f}  std={view[f'{metric}_std']:.4f}")


# %%
# =============================================================================
# SECTION 3 — Logistic Regression tuning
# =============================================================================
# Objective:
#   Search `C` (inverse regularisation strength) and `class_weight`, keeping L2
#   regularisation and the scikit-learn default solver unchanged.

section("SECTION 3 — Logistic Regression tuning")

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
report_search("Logistic Regression", results["Logistic Regression"])
for w in caught:
    print(f"   [WARNING] {w.category.__name__}: {w.message}")

lr_flags = tuning.flag_boundary_hits(
    lr_search.best_params_, tuning.LOGISTIC_REGRESSION_PARAM_GRID, ("classifier__C",)
)
print("   [BOUNDARY FLAG]", *lr_flags) if lr_flags else print("   [OK] Selected C is not at a grid boundary.")

# Result:
#   The search re-selected exactly the default cell (C=1.0,
#   class_weight=None), so Logistic Regression's tuned row is identical to its
#   default row -- the grid found nothing better than where it started.


# %%
# =============================================================================
# SECTION 4 — Random Forest tuning
# =============================================================================
# Objective:
#   Search `max_depth`, `min_samples_leaf`, `max_features` and `class_weight`,
#   with `n_estimators=500`, `random_state=42` and `n_jobs=-1` fixed.
#
# Runtime note:
#   96 candidates x 5 folds = 480 individual 500-tree forest fits, run
#   sequentially at the GridSearchCV level -- the slowest search here
#   (~8 minutes on 12 cores).

section("SECTION 4 — Random Forest tuning")

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
report_search("Random Forest", results["Random Forest"])
for w in caught:
    print(f"   [WARNING] {w.category.__name__}: {w.message}")

# Result:
#   The selected configuration (bounded max_depth, min_samples_leaf above 1,
#   max_features="sqrt") is more structurally regularised than the
#   unconstrained default AND achieved better CV metrics. Reduced overfitting
#   is NOT thereby demonstrated -- that would require train-vs-validation gaps,
#   which this script does not compute (`return_train_score=False`). The two
#   statements are kept separate.


# %%
# =============================================================================
# SECTION 5 — RBF SVC tuning
# =============================================================================
# Objective:
#   Search `C`, `gamma` and `class_weight`. `probability` is never set, so
#   ROC-AUC and Average Precision continue to be scored via
#   `decision_function`, matching the baseline configuration.

section("SECTION 5 — RBF SVC tuning")

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
report_search("RBF SVC", results["RBF SVC"])
for w in caught:
    print(f"   [WARNING] {w.category.__name__}: {w.message}")

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
svc_best_pipeline = tuning.get_search_best_pipeline(svc_search)
assert not hasattr(svc_best_pipeline.named_steps["classifier"], "predict_proba")
print("   [OK] Selected SVC has no predict_proba (probability was never enabled).")

first_stage_svc = results["RBF SVC"]

# --- Persist the full first-stage surface, for the heatmap below -------------
# `class_weight=None` is rendered as the lowercase label "none" rather than
# str(None) == "None": pandas treats the exact string "None" as a missing
# value on read, so the persisted column would come back as NaN and the
# artefact would not survive its own round-trip.
svc_grid_results = pd.DataFrame(
    {
        "C": [p["classifier__C"] for p in svc_search.cv_results_["params"]],
        "gamma": [str(p["classifier__gamma"]) for p in svc_search.cv_results_["params"]],
        "class_weight": [
            "none" if p["classifier__class_weight"] is None else str(p["classifier__class_weight"])
            for p in svc_search.cv_results_["params"]
        ],
        "mean_test_roc_auc": svc_search.cv_results_["mean_test_roc_auc"],
        "std_test_roc_auc": svc_search.cv_results_["std_test_roc_auc"],
        "mean_test_average_precision": svc_search.cv_results_["mean_test_average_precision"],
        "mean_test_balanced_accuracy": svc_search.cv_results_["mean_test_balanced_accuracy"],
    }
)
svc_grid_path = artifacts.save_table(svc_grid_results, artifacts.SVC_GRID_RESULTS)
print(f"\n[artifact] {svc_grid_path.relative_to(PROJECT_ROOT)}  ({len(svc_grid_results)} rows)")

# Result:
#   The first-stage search selected C at the UPPER edge of its grid, which
#   means the grid -- not the data -- may have limited the result. That is
#   resolved by exactly one local refinement in Section 5b.

# %%
# --- FIGURE: the SVC C x gamma search surface --------------------------------
# One panel per `class_weight` setting, because a single heatmap mixing both
# would be unreadable and would hide the fact that class_weight, not C or
# gamma, is what moves this model's default operating point.
class_weight_levels = sorted(svc_grid_results["class_weight"].unique())
gamma_levels = sorted(
    svc_grid_results["gamma"].unique(),
    key=lambda g: (g == "scale", float(g) if g != "scale" else 0.0),
)
c_levels = sorted(svc_grid_results["C"].unique())

# One shared colour scale across both panels: with per-panel autoscaling the
# best cell of the weaker panel would look identical to the best cell overall.
scale_min = float(svc_grid_results["mean_test_roc_auc"].min())
scale_max = float(svc_grid_results["mean_test_roc_auc"].max())

fig, axes = plt.subplots(1, len(class_weight_levels), figsize=(6.2 * len(class_weight_levels), 4.6))
axes = np.atleast_1d(axes)
for ax, class_weight in zip(axes, class_weight_levels):
    subset = svc_grid_results.loc[svc_grid_results["class_weight"] == class_weight]
    matrix = (
        subset.pivot(index="C", columns="gamma", values="mean_test_roc_auc")
        .reindex(index=c_levels, columns=gamma_levels)
        .to_numpy()
    )
    plots.annotated_heatmap(
        ax, matrix,
        row_labels=[f"C={c:g}" for c in c_levels],
        col_labels=[f"γ={g}" for g in gamma_levels],
        colorbar_label="mean CV ROC-AUC",
        vmin=scale_min, vmax=scale_max,
    )
    ax.set_title(f"class_weight = {class_weight}")
    ax.set_xlabel("gamma")
    ax.set_ylabel("C")

    best_idx = np.unravel_index(np.nanargmax(matrix), matrix.shape)
    ax.add_patch(
        Rectangle(
            (best_idx[1] - 0.5, best_idx[0] - 0.5), 1, 1,
            fill=False, edgecolor=plots.ACCENT_COLOR, linewidth=3,
        )
    )

fig.suptitle(
    "RBF SVC grid search — mean 5-fold CV ROC-AUC over C x gamma\n"
    "(shared colour scale across both panels; orange box = best cell in each panel; "
    "selection uses refit='roc_auc')",
    fontsize=12.5, fontweight="bold",
)
fig.tight_layout()
plots.save_figure(fig, "03_svc_gridsearch_heatmap.png")


# %%
# =============================================================================
# SECTION 5b — RBF SVC boundary refinement (second and FINAL stage)
# =============================================================================
# Objective:
#   Resolve the first-stage upper-boundary hit on `C` with ONE local search:
#   `C` extended upward past 100, `gamma` refined locally around 0.001.
#
# Rationale, and its cost stated openly:
#   A selected value on the edge of its grid means the search space, not the
#   data, may have bounded the result, so the first-stage number cannot be read
#   as "the best this family can do". One local extension is the proportionate
#   response. But this is a SECOND consultation of the same training folds for
#   one family only: the SVC therefore receives more post-hoc adaptation
#   opportunity than Logistic Regression and Random Forest, whose grids were
#   searched once each. Any SVC-vs-LR/RF gap below needs reading in that light.
#
#   Pre-declared stopping rule: there is NO third stage. If C=3000.0 (the new
#   upper edge) is selected, that is flagged as a residual limitation and the
#   search stops anyway.

section("SECTION 5b — RBF SVC boundary refinement (second and FINAL stage)")

print("Refinement grid:")
for param, values in tuning.RBF_SVC_REFINEMENT_PARAM_GRID.items():
    print(f"   {param:28s} {values}")
assert tuning.grid_size(tuning.RBF_SVC_REFINEMENT_PARAM_GRID) == 24

t0 = time.time()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    svc_refined_search = tuning.build_rbf_svc_refinement_search()
    svc_refined_search.fit(X_train, y_train)
elapsed = time.time() - t0
searches["RBF SVC (refined)"] = svc_refined_search
svc_refined = tuning.summarize_best(svc_refined_search)

print(f"\nFitted 24 candidates x {cv.get_n_splits()} folds in {elapsed:.1f}s.")
print(f"Refined best params: {svc_refined_search.best_params_}")
report_search("RBF SVC (refined)", svc_refined)
for w in caught:
    print(f"   [WARNING] {w.category.__name__}: {w.message}")

# --- Grid-local flags vs. the UNION of both searched grids --------------------
# `flag_boundary_hits` only ever sees ONE grid, so it cannot know that the
# refinement grid's lower C edge (100.0) is the first-stage grid's UPPER edge,
# i.e. that the region below 100 was already searched and found worse. Whether
# a flag is a genuine open question is therefore judged against the union.
selected_C = svc_refined_search.best_params_["classifier__C"]
union_C = sorted(
    set(tuning.RBF_SVC_PARAM_GRID["classifier__C"])
    | set(tuning.RBF_SVC_REFINEMENT_PARAM_GRID["classifier__C"])
)
print(f"\nCombined C range searched across BOTH stages: {union_C}")
if selected_C in (max(union_C), min(union_C)):
    print(f"   [OPEN] Selected C={selected_C} is still at an end of everything searched; "
          f"values beyond it remain untested. Accepted as a residual limitation, and the "
          f"search stops here per the pre-declared rule.")
else:
    print(f"   [RESOLVED] Selected C={selected_C} is INTERIOR to the combined searched range "
          f"({min(union_C)} ... {max(union_C)}): both smaller and larger C values were "
          f"evaluated and scored no better. The first-stage boundary flag is resolved -- the "
          f"grid, not the data, was the earlier constraint, and extending it changed nothing.")

print(f"\nFirst-stage -> refined (same selection rule, refit='{tuning.REFIT_METRIC}'):")
first_stage_view = tuning.as_float_view(first_stage_svc)
refined_view = tuning.as_float_view(svc_refined)
for metric in tuning.CV_SCORING:
    before, after = first_stage_view[f"{metric}_mean"], refined_view[f"{metric}_mean"]
    print(f"   {metric:19s} {before:.4f} -> {after:.4f}   (delta {after - before:+.4f})")

# The refined configuration REPLACES the first-stage SVC from here on; the
# first-stage row stays printed above so the tuning history remains visible.
results["RBF SVC"] = svc_refined
svc_refined_best_pipeline = tuning.get_search_best_pipeline(svc_refined_search)
assert not hasattr(svc_refined_best_pipeline.named_steps["classifier"], "predict_proba")
print("\n[FROZEN] The refined RBF SVC configuration is now frozen; no further SVC tuning.")


# %%
# =============================================================================
# SECTION 6 — Default vs. tuned, and the three tuned models against each other
# =============================================================================
# Objective:
#   Compare each tuned model against its own default, then rank the three
#   tuned models, as observed differences -- not as significance tests.

section("SECTION 6 — Default vs. tuned comparison")

print("The 'RBF SVC' tuned row below is the REFINED (Section 5b) configuration.\n")

comparison_rows = []
for name in TUNED_FAMILIES:
    default = DEFAULT_RESULTS[name]
    tuned = tuning.as_float_view(results[name])
    for metric in tuning.CV_SCORING:
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
default_vs_tuned = pd.DataFrame(comparison_rows)
print(default_vs_tuned.set_index(["model", "metric"]).round(4).to_string())
print(
    "\nDeltas are observed mean differences only. No formal significance test is "
    "performed; they should be read alongside each model's fold-to-fold standard "
    "deviation, not treated as proof that a change is real."
)

metric_columns = [f"{m}_{stat}" for m in tuning.CV_SCORING for stat in ("mean", "std")]
tuned_table = pd.DataFrame(
    {name: {col: tuning.as_float_view(results[name])[col] for col in metric_columns}
     for name in TUNED_FAMILIES}
).T.round(4)

print("\nTuned models:")
print(tuned_table.to_string())
ranking_auc = tuned_table["roc_auc_mean"].sort_values(ascending=False).index.tolist()
ranking_ap = tuned_table["average_precision_mean"].sort_values(ascending=False).index.tolist()
print(f"\nRanked by ROC-AUC (PRIMARY):      {ranking_auc}")
print(f"Ranked by Average Precision:      {ranking_ap}")
print(f"Same ranking? {'YES' if ranking_auc == ranking_ap else 'NO'}")

# %%
# --- FIGURE: default vs. tuned, on the two model-comparison metrics -----------
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
positions = np.arange(len(TUNED_FAMILIES))
bar_height = 0.36

for ax, metric, title in zip(
    axes,
    ["roc_auc", "average_precision"],
    ["ROC-AUC (PRIMARY)", "Average Precision (SECONDARY)"],
):
    rows = default_vs_tuned.loc[default_vs_tuned["metric"] == metric].set_index("model")
    default_means = np.asarray(
        [float(cast(float, rows.at[m, "default_mean"])) for m in TUNED_FAMILIES],
        dtype=float,
    )
    default_stds = np.asarray(
        [float(cast(float, rows.at[m, "default_std"])) for m in TUNED_FAMILIES],
        dtype=float,
    )
    tuned_means = np.asarray(
        [float(cast(float, rows.at[m, "tuned_mean"])) for m in TUNED_FAMILIES],
        dtype=float,
    )
    tuned_stds = np.asarray(
        [float(cast(float, rows.at[m, "tuned_std"])) for m in TUNED_FAMILIES],
        dtype=float,
    )
    axis_scale = float(max(default_means.max(), tuned_means.max()))

    ax.barh(positions + bar_height / 2, default_means, bar_height, xerr=default_stds,
            color=plots.NEUTRAL_COLOR, label="default hyperparameters (analysis/02)",
            error_kw={"ecolor": "black", "capsize": 3, "elinewidth": 1})
    ax.barh(positions - bar_height / 2, tuned_means, bar_height, xerr=tuned_stds,
            color=plots.NON_EVENT_COLOR, label="tuned (analysis/03)",
            error_kw={"ecolor": "black", "capsize": 3, "elinewidth": 1})
    ax.set_yticks(positions, labels=TUNED_FAMILIES)
    ax.invert_yaxis()
    ax.set_xlabel(title)
    ax.set_xlim(0, axis_scale * 1.38)
    for pos, value, err in zip(positions - bar_height / 2, tuned_means, tuned_stds):
        text_x = float(value + err + axis_scale * 0.03)
        ax.text(text_x, pos, f"{value:.4f}", va="center", fontsize=8.5)
    for pos, value, err in zip(positions + bar_height / 2, default_means, default_stds):
        text_x = float(value + err + axis_scale * 0.03)
        ax.text(text_x, pos, f"{value:.4f}", va="center", fontsize=8.5)

handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc="lower center", ncol=2, fontsize=9.5,
           bbox_to_anchor=(0.5, -0.06))
fig.suptitle(
    "Default vs. tuned — 5-fold CV on the training partition\n"
    "(whiskers show fold-to-fold standard deviation, not confidence intervals)",
    fontsize=12.5, fontweight="bold",
)
fig.text(
    0.5, -0.13,
    "Logistic Regression's two bars are identical because its grid re-selected exactly the "
    "default cell (C=1.0, class_weight=None) — tuning found nothing better than where it started.",
    ha="center", fontsize=9, style="italic",
)
fig.tight_layout()
plots.save_figure(fig, "03_default_vs_tuned.png")


# %%
# =============================================================================
# SECTION 7 — Reading the SVC's Balanced Accuracy correctly
# =============================================================================
# This section exists because the tuned SVC's Balanced Accuracy jumps far above
# the other two models', and that number is easy to misread as evidence of
# better discrimination. It is not.

section("SECTION 7 — Reading the SVC's Balanced Accuracy correctly")

lr_view = tuning.as_float_view(results["Logistic Regression"])
svc_view = tuning.as_float_view(results["RBF SVC"])

svc_native_positive_rate = float(
    np.mean(tuning.get_search_best_pipeline(svc_refined_search).predict(X_train) == 1)
)
lr_native_positive_rate = float(
    np.mean(tuning.get_search_best_pipeline(lr_search).predict(X_train) == 1)
)

print(f"Logistic Regression : ROC-AUC {lr_view['roc_auc_mean']:.4f}   "
      f"AP {lr_view['average_precision_mean']:.4f}   "
      f"Balanced Accuracy {lr_view['balanced_accuracy_mean']:.4f}")
print(f"RBF SVC (refined)   : ROC-AUC {svc_view['roc_auc_mean']:.4f}   "
      f"AP {svc_view['average_precision_mean']:.4f}   "
      f"Balanced Accuracy {svc_view['balanced_accuracy_mean']:.4f}")
print(f"\nSelected SVC class_weight : {svc_refined_search.best_params_['classifier__class_weight']!r}")
print(f"Selected LR  class_weight : {lr_search.best_params_['classifier__class_weight']!r}")
print("\nIn-sample positive-prediction rate at each model's NATIVE decision rule:")
print(f"   Logistic Regression (probability >= 0.5) : {lr_native_positive_rate:.4f}")
print(f"   RBF SVC (decision_function >= 0)         : {svc_native_positive_rate:.4f}")
print(f"   Training event prevalence                : {y_train.mean():.4f}")

auc_gap = svc_view["roc_auc_mean"] - lr_view["roc_auc_mean"]
ap_gap = svc_view["average_precision_mean"] - lr_view["average_precision_mean"]
auc_summary = (f"{svc_view['roc_auc_mean']:.4f} vs. {lr_view['roc_auc_mean']:.4f} "
               f"(delta {auc_gap:+.4f})")
ap_summary = (f"{svc_view['average_precision_mean']:.4f} vs. "
              f"{lr_view['average_precision_mean']:.4f} (delta {ap_gap:+.4f})")

print(f"""
Interpretation:

  - On the two THRESHOLD-INDEPENDENT metrics the two models are essentially
    tied: ROC-AUC {auc_summary}, Average Precision {ap_summary}.
    Differences in the third decimal are not an advantage for either model.

  - Balanced Accuracy is a THRESHOLD-DEPENDENT metric: it describes one hard
    operating point, not ranking quality. The tuned SVC's configuration
    includes class_weight="balanced", which re-weights the two classes during
    fitting and thereby moves where its native decision rule
    (decision_function >= 0) falls. Its positive-prediction rate is therefore
    far higher than the Logistic Regression's at that model's own default cut
    of 0.5, as the rates printed above show.

  - The comparison is between two DIFFERENT operating points, not between two
    levels of discrimination. Logistic Regression's Balanced Accuracy rises from
    ~{lr_view['balanced_accuracy_mean']:.3f} at the default 0.5 cut to ~{config.EXPECTED_TRAIN_OOF_SELECTED_BALANCED_ACCURACY:.3f} once a threshold is
    chosen on training-only out-of-fold predictions (analysis/05) -- without its
    ROC-AUC or Average Precision changing at all, because choosing a threshold
    does not change the underlying ranking.

  - Therefore the SVC's higher Balanced Accuracy here must NOT be read as
    evidence that it discriminates better. It reflects a different default
    operating point, which is a modelling choice, not a property of the
    ranking. Model selection in this project rests on the
    threshold-independent metrics (analysis/04); the operating point is chosen
    afterwards, once, for the already-selected model (analysis/05).

  - No threshold competition among finalists is run here or anywhere else in
    this project.
""")


# %%
# =============================================================================
# SECTION 8 — Status of these estimates, and the persisted artefact
# =============================================================================

section("SECTION 8 — Status of these estimates")

print("""
GridSearchCV best scores, and every training-only comparison in this script,
are part of MODEL SELECTION. They are NOT an unbiased final generalization
estimate:

  - each reported score is the best of many candidates evaluated on the same
    five training folds, and selecting the maximum over 10, 96, 60 (and, for
    the SVC, a further 24) candidates biases that maximum optimistically;
  - the RBF SVC was searched TWICE while Logistic Regression and Random Forest
    were searched once, so it received more post-hoc adaptation opportunity;
  - fold-to-fold standard deviations describe spread across overlapping
    training folds. They are not standard errors, not confidence intervals,
    and no significance claim is derived from them.

Full nested hyperparameter-tuning cross-validation, which would give an
approximately unbiased estimate of the whole tuning procedure, is deliberately
not introduced at this stage. The held-out test partition remains reserved for
the dedicated held-out evaluation stage of the finished model and never
informs model selection.
""")

tuned_payload = {
    "cv": {
        "splitter": "StratifiedKFold",
        "n_splits": cv.get_n_splits(),
        "shuffle": cv_attrs.shuffle,
        "random_state": cv_attrs.random_state,
    },
    "refit_metric": tuning.REFIT_METRIC,
    "models": {},
}
for name in TUNED_FAMILIES:
    view = tuning.as_float_view(results[name])
    tuned_payload["models"][name] = {
        "status": "PRESPECIFIED",
        "search_stages": 2 if name == "RBF SVC" else 1,
        "best_params": {
            key.removeprefix("classifier__"): value
            for key, value in results[name]["best_params"].items()
        },
        **{f"{metric}_{stat}": view[f"{metric}_{stat}"]
           for metric in tuning.CV_SCORING for stat in ("mean", "std")},
    }

tuned_path = artifacts.save_json(tuned_payload, artifacts.TUNED_RESULTS)
print(f"[artifact] {tuned_path.relative_to(PROJECT_ROOT)}")
print("\nSelected configurations carried forward:")
for name, entry in tuned_payload["models"].items():
    print(f"   {name:22s} {entry['best_params']}")

print("""
Next: analysis/04_model_complementarity_and_exploratory_models.py — do these
three tuned models actually make different errors, and does any ensemble or
boosting alternative add discrimination? No further tuning of Logistic
Regression, Random Forest or RBF SVC happens after this script.
""")
